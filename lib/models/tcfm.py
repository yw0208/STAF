import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import BASE_DATA_DIR

from torch.autograd import Variable  ##

SMPL_MEAN_PARAMS = 'data/base_data/smpl_mean_params.npz'


class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores


class TemporalEncoder(nn.Module):
    def __init__(self, channel):
        super(TemporalEncoder, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        # self.conv_phi1 = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
        #                            padding=0, bias=False)
        # self.conv_theta1 = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
        #                              padding=0, bias=False)
        self.conv_g = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv1d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.conv_mask_forR = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        # self.attention = TemporalAttention(attention_size=2155, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        # NTF -> NFT
        x = x.permute(0, 2, 1)
        b, c, thw = x.size()  # N x 2048 x 16

        x_ = x.permute(0, 2, 1)
        xx = torch.matmul(x_, x)
        xx = self.softmax(xx)  # N x 16 x 16

        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)  # N x 2048/2 x 16
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # N x 16 x 2048/2
        # x_phi1 = self.conv_phi1(x).view(b, self.inter_channel, -1)  # N x 2048/2 x 16
        # x_theta1 = self.conv_theta1(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # N x 16 x 2048/2
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # N x 16 x 2048/2

        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)  # N x 16 x 16
        # mul_theta_phi1 = torch.matmul(x_theta1, x_phi1)
        # mul_theta_phi1 = self.softmax(mul_theta_phi1)  # N x 16 x 16

        R = torch.cat((xx, mul_theta_phi), dim=0).view(xx.size(0), -1, xx.size(1),
                                                                       xx.size(2))  # 2 x N x 16 x 16
        Y = self.conv_mask_forR(R).reshape(b, thw, thw)
        Y = self.softmax(Y)  # N x 16 x 16

        mul_theta_phi_g = torch.matmul(Y, x_g)  # N x 16 x 2048/2
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel,
                                                                             thw)  # N x 2048/2 x 16

        mask = self.conv_mask(mul_theta_phi_g)  # N x 2048 x 16

        out_ = mask + x  #

        return out_


class TCFM(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            channel=2048,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):
        super(TCFM, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.nonlocalblock = TemporalEncoder(channel=channel)

    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature = self.nonlocalblock(input, is_train=is_train)  #
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size * seqlen, -1)

        return feature
        # smpl_output_Dm
