# подавить предупреждения пользователей PyTorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class Tnet(nn.Module):
    ''' T-Net обучается матрице преобразования с заданным размером '''
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # размеры для матрицы преобразования
        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        # батчи
        bs = x.shape[0]

        # x: (B, 3, N)

        # проход через общие слои MLP
        x = self.bn1(F.relu(self.conv1(x)))  # (B, 3, N) → (B, 64, N)
        x = self.bn2(F.relu(self.conv2(x)))  # (B, 64, N) → (B, 128, N)
        x = self.bn3(F.relu(self.conv3(x)))  # (B, 128, N) → (B, 1024, N)

        # максимальный пул по количеству точек
        x = self.max_pool(x).view(bs, -1)  # (B, 1024, N) → (B, 1024, 1)
        # .view(bs, -1)=(B, 1024, 1) → (B, 1024)

        # проход через MLP
        x = self.bn4(F.relu(self.linear1(x)))  # (B, 1024) → (B, 512)
        x = self.bn5(F.relu(self.linear2(x)))  # (B, 512) → (B, 256)
        x = self.linear3(x)  # (B, 256) → (B, 9)

        # инициализация единичной матрицы
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(- 1, self.dim, self.dim) + iden  # (B, 9) → (B, 3, 3)

        return x  # A_input: (B, 3, 3)


class PointNetBackbone(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024,
                 local_feat=True):
        ''' Инициализаторы:
                num_points - количество точек в облаке точек
                num_global_feats - количество глобальных объектов для основного
                                   слоя максимального пулинга
                local_feat - если True, forward() возвращает конкатенацию
                             локальных и глобальных объектов
            '''
        super(PointNetBackbone, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points,
                                     return_indices=True)

    def forward(self, x):
        # получаем размер батча
        bs = x.shape[0]

        # Пропустить через первый Tnet для получения матрицы преобразования
        A_input = self.tnet1(x)  # (B, 3, 3)

        # выполнить первое преобразование для каждой точки в батче
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        # x.transpose(2,1): (B, N, 3)
        # A_input: (B, 3, 3)
        # bmm: (B, N, 3) @ (B, 3, 3) → (B, N, 3)
        # после .transpose(2, 1) x: (B, 3, N)

        # проходим через первый общий MLP
        x = self.bn1(F.relu(self.conv1(x)))  # (B, 3, N) → (B, 64, N)
        x = self.bn2(F.relu(self.conv2(x)))  # (B, 64, N) → (B, 64, N)

        # получить преобразование признаков
        A_feat = self.tnet2(x)  # A_feat: (B, 64, 64)

    # выполняем второе преобразование для каждого (64-мерного) признака в батче
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        # x.transpose: (B, N, 64)
        # A_feat: (B, 64, 64)
        # bmm: (B, N, 64)
        # transpose обратно x: (B, 64, N)

        # сохраняем локальные точечные признаки для сегментации
        local_features = x.clone()  # local_features: (B, 64, N)

        # проходим через второй MLP
        x = self.bn3(F.relu(self.conv3(x)))  # (B, 64, N) → (B, 64, N)
        x = self.bn4(F.relu(self.conv4(x)))  # (B, 64, N) → (B, 128, N)
        x = self.bn5(F.relu(self.conv5(x)))  # (B, 128, N) → (B, 1024, N)

        # получаем глобальный вектор признаков и критические индексы
        global_features, critical_indexes = self.max_pool(x)
        # global_features: (B, 1024, 1)
        # critical_indexes: (B, 1024, 1)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)
        # global_features: (B, 1024)
        # critical_indexes: (B, 1024)
        if self.local_feat:
            features = torch.cat((local_features,
                                  global_features.unsqueeze(-1).
                                  repeat(1, 1, self.num_points)), dim=1)
            # local_features: (B, 64, N)
            # global_repeated: (B, 1024, N)
            # concat → features: (B, 1088, N)

            return features, critical_indexes, A_feat
            # features: (B, 1088, N)
            # critical_indexes: (B, 1024)
            # A_feat: (B, 64, 64)

        else:
            return global_features, critical_indexes, A_feat
            # global_features: (B, 1024)
            # critical_indexes: (B, 1024)
            # A_feat: (B, 64, 64)


class PointNetClassHead(nn.Module):
    '''' Classification Head '''
    def __init__(self, num_points=2500, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()

# получаем базовую структуру(для классификации нужны только глобальные признаки
        self.backbone = PointNetBackbone(num_points, num_global_feats,
                                         local_feat=False)

        # MLP для классификации
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)

        # batchnorm для первых двух линейных слоёв
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

# В статье указано, что пакетная нормализация была добавлена ​​только к слою.
# перед слоем классификации, но в другой версии добавлен dropout.
        # для перых двух слоёв
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (B, 3, N)
        # получаем глобальные признаки
        x, crit_idxs, A_feat = self.backbone(x)
        # Backbone (local_feat=False) → x: (B, 1024)

        x = self.bn1(F.relu(self.linear1(x)))  # (B, 512)
        x = self.bn2(F.relu(self.linear2(x)))  # (B, 256)
        x = self.dropout(x)  # (B, 256)
        x = self.linear3(x)  # (B, k)

        return x, crit_idxs, A_feat


class PointNetSegHead(nn.Module):
    ''' Segmentation Head '''
    def __init__(self, num_points=2500, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        # получаем базовую структуру
        self.backbone = PointNetBackbone(num_points, num_global_feats,
                                         local_feat=True)

        # общий MLP
        num_features = num_global_feats + 64  # глобальные и локальные признаки
        self.conv1 = nn.Conv1d(num_features, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)

        # batch norms для общих MLP
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # вход (B, 3, N)
        # получаем комбинированные признаки
        x, crit_idxs, A_feat = self.backbone(x)
        # Backbone (local_feat=True) → x: (B, 1088, N)

        # пропускаем через общий MLP
        x = self.bn1(F.relu(self.conv1(x)))  # (B, 512, N)
        x = self.bn2(F.relu(self.conv2(x)))  # (B, 256, N)
        x = self.bn3(F.relu(self.conv3(x)))  # (B, 128, N)
        x = self.conv4(x)  # (B, m, N)

        x = x.transpose(2, 1)  # (B, N, m)

        return x, crit_idxs, A_feat


def main():
    test_data = torch.rand(32, 3, 2500)

    # test T-net
    tnet = Tnet(dim=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    # test backbone
    pointfeat = PointNetBackbone(local_feat=False)
    out, _, _ = pointfeat(test_data)
    print(f'Global Features shape: {out.shape}')

    pointfeat = PointNetBackbone(local_feat=True)
    out, _, _ = pointfeat(test_data)
    print(f'Combined Features shape: {out.shape}')

    # test on single batch (should throw error if there is an issue)
    pointfeat = PointNetBackbone(local_feat=True).eval()
    out, _, _ = pointfeat(test_data[0, :, :].unsqueeze(0))

    # test classification head
    classifier = PointNetClassHead(k=5)
    out, _, _ = classifier(test_data)
    print(f'Class output shape: {out.shape}')

    classifier = PointNetClassHead(k=5).eval()
    out, _, _ = classifier(test_data[0, :, :].unsqueeze(0))

    # test segmentation head
    seg = PointNetSegHead(m=3)
    out, _, _ = seg(test_data)
    print(f'Seg shape: {out.shape}')


if __name__ == '__main__':
    main()
