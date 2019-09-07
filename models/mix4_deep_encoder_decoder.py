import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(opts):
    return EncoderDecoderNet(n_feats=32, n_blocks=10, n_resgroups=16)


class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, stride=1, padding=kernel_size_1 // 2)
        self.conv_3_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_1, stride=1, padding=kernel_size_1 // 2)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size_2, stride=1, padding=kernel_size_2 // 2)
        self.conv_5_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_2, stride=1, padding=kernel_size_2 // 2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            if nm == 'in':
                module_body.append(nn.InstanceNorm2d(n_feats, affine=True))
            if nm == 'bn':
                module_body.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                module_body.append(nn.PReLU())
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


class EncoderDecoderNet(nn.Module):
    def __init__(self, n_feats, n_blocks, n_resgroups, nm=None):
        super(EncoderDecoderNet, self).__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.n_resgroups = n_resgroups
        self.nm = nm
        self.__build_model()

    def __build_model(self):
        self.head = nn.Sequential(
            nn.Conv2d(4, self.n_feats * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

        self.downer = nn.Sequential(
            nn.Conv2d(self.n_feats * 2, self.n_feats * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, stride=2, padding=1, bias=True)
        )
        local_path = [
            RBGroup(n_feats=self.n_feats * 4, nm=self.nm, n_blocks=self.n_blocks) for _ in range(self.n_resgroups)
        ]
        local_path.append(nn.Conv2d(self.n_feats * 4, self.n_feats * 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.local_path = nn.Sequential(*local_path)
        self.uper = nn.Sequential(
            nn.ConvTranspose2d(self.n_feats * 4, self.n_feats * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(self.n_feats * 2, self.n_feats * 2, kernel_size=4, stride=2, padding=1, bias=True)
        )

        self.global_path = nn.Sequential(
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
            MSRB(self.n_feats * 2),
        )
        self.global_down = nn.Conv2d(self.n_feats * 8 * 2, self.n_feats * 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.linear = nn.Sequential(
            nn.Conv2d(self.n_feats * 4, self.n_feats * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(self.n_feats * 2, self.n_feats * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

        self.tail = nn.Conv2d(self.n_feats * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.head(x)

        x_down = self.downer(x)
        local_fea = self.local_path(x_down)
        local_fea += x_down
        local_fea = self.uper(local_fea)

        out = x
        msrb_out = []
        for i in range(8):
            out = self.global_path[i](out)
            msrb_out.append(out)
        global_fea = torch.cat(msrb_out, 1)
        global_fea = self.global_down(global_fea)

        fused_fea = torch.cat([global_fea, local_fea], 1)
        fused_fea = self.linear(fused_fea)

        x = self.tail(fused_fea+x)
        return F.tanh(x)
