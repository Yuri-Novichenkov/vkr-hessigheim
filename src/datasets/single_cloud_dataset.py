from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class CloudConfig:
    txt_path: str
    num_points: int = 2500
    steps_per_epoch: int = 2000  # "виртуальная" длина датасета (сколько семплов в эпоху)
    training: bool = True

    # Какие признаки брать из таблицы:
    # - "xyz" => только X,Y,Z (совместимо с твоим PointNet input = 3)
    # - "xyzrgb" => X,Y,Z,R,G,B (тогда тебе надо будет поменять dim=6 и Conv1d(6,...))
    features: str = "xyz"

    # Какой столбец является target (меткой на точку)
    label_col: str = "Classification"

    # Разделитель в txt
    sep: str = "\t"

    # Нормализация XYZ: центрирование + масштабирование в единичную сферу
    normalize_xyz: bool = True

    # Если хочешь deterministic семплирование при training=False
    seed: int = 42


class SingleCloudDataset(Dataset):
    """
    Делает много обучающих примеров из ОДНОГО point cloud файла.

    Возвращает:
      points: torch.FloatTensor (C, num_points)
      labels: torch.LongTensor (num_points,)   # если label_col существует

    Под твой текущий PointNet:
      C должно быть 3 (features="xyz"), чтобы вход был (B,3,N).
    """

    def __init__(self, cfg: CloudConfig):
        self.cfg = cfg

        df = pd.read_csv(cfg.txt_path, sep=cfg.sep)

        # Поддержим вариант, если в файле X может называться "//X" (как у тебя на скрине)
        # Приведём имена к удобным
        cols = {c: c.strip() for c in df.columns}
        df = df.rename(columns=cols)

        if "X" not in df.columns and "//X" in df.columns:
            df = df.rename(columns={"//X": "X"})

        required_xyz = ["X", "Y", "Z"]
        for c in required_xyz:
            if c not in df.columns:
                raise ValueError(f"Не нашёл колонку {c} в {cfg.txt_path}. Есть колонки: {list(df.columns)}")

        self.has_labels = cfg.label_col in df.columns

        # ===== признаки =====
        if cfg.features == "xyz":
            feat_cols = ["X", "Y", "Z"]
        elif cfg.features == "xyzrgb":
            for c in ["R", "G", "B"]:
                if c not in df.columns:
                    raise ValueError(f"features='xyzrgb', но нет колонки {c}")
            feat_cols = ["X", "Y", "Z", "R", "G", "B"]
        else:
            raise ValueError("cfg.features должен быть 'xyz' или 'xyzrgb'")

        feats = df[feat_cols].to_numpy(dtype=np.float32)  # (N_all, C)
        self.xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)  # (N_all, 3)
        self.feats = feats

        if self.has_labels:
            labels = df[cfg.label_col].to_numpy(dtype=np.int64)
            self.labels = labels
        else:
            self.labels = None

        self.N_all = self.feats.shape[0]
        self.C = self.feats.shape[1]

        # RNG для режима validation/test, чтобы выборка была повторяемой
        self._rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        # "виртуальная длина": сколько раз за эпоху мы хотим взять случайный семпл
        return self.cfg.steps_per_epoch

    @staticmethod
    def _normalize_xyz_inplace(pts_xyz: np.ndarray) -> np.ndarray:
        """
        pts_xyz: (N,3)
        Центрируем и масштабируем в единичную сферу.
        """
        centroid = pts_xyz.mean(axis=0, keepdims=True)
        pts_xyz = pts_xyz - centroid
        scale = np.max(np.linalg.norm(pts_xyz, axis=1))
        if scale > 1e-9:
            pts_xyz = pts_xyz / scale
        return pts_xyz

    def _sample_indices(self) -> np.ndarray:
        n = self.cfg.num_points
        if self.cfg.training:
            # случайно, можно с повторениями
            return np.random.choice(self.N_all, n, replace=(self.N_all < n))
        else:
            # детерминированно/стабильно
            if self.N_all >= n:
                # просто первые n (или можешь заменить на равномерную выборку)
                return np.arange(n)
            extra = self._rng.choice(self.N_all, n - self.N_all, replace=True)
            return np.concatenate([np.arange(self.N_all), extra])

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        ids = self._sample_indices()

        feats = self.feats[ids].copy()  # (num_points, C)

        # Нормализуем только XYZ компоненты (первые 3 фичи всегда XYZ)
        if self.cfg.normalize_xyz:
            feats[:, 0:3] = self._normalize_xyz_inplace(feats[:, 0:3])

        # -> (C, num_points)
        points = torch.from_numpy(feats).transpose(0, 1).contiguous().float()

        if self.has_labels:
            labels = torch.from_numpy(self.labels[ids]).long()  # (num_points,)
            return points, labels

        return points