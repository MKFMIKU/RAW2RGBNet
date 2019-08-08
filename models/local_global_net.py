import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(opts):
    return LocalGloablNet(n_feats=64, n_blocks=3, n_resgroups=24)


class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            if nm == 'in':
                module_body.append(nn.InstanceNorm2d(n_feats))
            if nm == 'bn':
                module_body.append(nn.BatchNorm2d(n_feats))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


class RBGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RBGroup, self).__init__()
        module_body = [
            RB(n_feats, nm) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


class LocalGloablNet(nn.Module):
    def __init__(self, n_feats, n_blocks, n_resgroups, nm=None):
        super(LocalGloablNet, self).__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.n_resgroups = n_resgroups
        self.nm = nm
        self.__build_model()

    def __build_model(self):
        self.head = nn.Conv2d(4, self.n_feats, kernel_size=11, stride=1, padding=5, bias=True)

        # Build Local Path here use RCAN, while keeping the origin size of image
        local_path = [
            RBGroup(n_feats=self.n_feats, nm=self.nm, n_blocks=self.n_blocks) for _ in range(self.n_resgroups)
        ]
        local_path.append(nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1, bias=False))
        self.local_path = nn.Sequential(*local_path)

        # Build Global Path here
        global_path = []
        g_feats = self.n_feats
        for _ in range(4):
            global_path.append(nn.AvgPool2d(2))
            global_path.append(nn.Conv2d(g_feats, g_feats, kernel_size=3, stride=1, padding=1, bias=True))
            if self.nm == 'in':
                global_path.append(nn.InstanceNorm2d(g_feats))
            if self.nm == 'bn':
                global_path.append(nn.BatchNorm2d(g_feats))
            global_path.append(nn.ReLU())
        global_path.append(nn.AdaptiveAvgPool2d(1))
        self.global_path = nn.Sequential(*global_path)

        self.tail = nn.Conv2d(self.n_feats, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.head(x)

        local_fea = self.local_path(x)
        local_fea += x

        global_fea = self.global_path(x)

        fused_fea = F.relu((global_fea + local_fea), inplace=True)

        x = self.tail(fused_fea)
        return x
