import torch
import torch.nn as nn
import torch.nn.functional as F


class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            if nm=='in':
                module_body.append(nn.InstanceNorm2d(n_feats))
            if nm=='bn':
                module_body.append(nn.BatchNorm2d(n_feats))
            module_body.append(nn.ReLU(inplace=True))
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


class RBGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RBGroup).__init__()
        module_body = [
            RB(n_feats, nm) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, 3))
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


class LocalGloablNet(nn.Module):
    def __init__(self, n_feats, n_blocks, n_resgroups, nm='in'):
        super(LocalGloablNet, self).__init__()
        self.__build_model()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.n_resgroups = n_resgroups

        self.nm = nm

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
        g_feats = 4
        for i in range(1, self.n_feats // 4 + 1):
            global_path.append(nn.AvgPool2d(2))
            global_path.append(nn.Conv2d(g_feats, i*self.n_feats//4, kernel_size=3, stride=1, padding=1, bias=True))
            g_feats = i*self.n_feats//4
            if self.nm=='in':
                global_path.append(nn.InstanceNorm2d(g_feats))
            if self.nm=='bn':
                global_path.append(nn.BatchNorm2d(g_feats))
            global_path.append(nn.ReLU(inplace=True))
        global_path.append(nn.AdaptiveAvgPool2d(4))
        self.global_path = nn.Sequential(*global_path)

        global_fc = []
        for i in [1024, 512, 256, 128]:
            global_fc.append(nn.Linear(i, i//2))
            if i > 128:
                global_fc.append(nn.ReLU(inplace=True))
        self.global_fc = nn.Sequential(*global_fc)

        self.tail = nn.Conv2d(self.n_feats, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)

        local_fea = self.local_path(x)
        local_fea += x

        global_fea = self.global_path(x)
        global_fea = global_fea.view(global_fea.size(0), -1)
        global_fea = self.global_fc(global_fea)

        fused_fea = F.relu((global_fea + local_fea), inplace=True)

        x = self.tail(fused_fea)
        return x
