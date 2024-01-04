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

        self.attention = TemporalAttention(attention_size=2155, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        out_ = x.permute(0, 2, 1)

        y_cur_2 = out_[:, :, 4]  #
        y_cur_1 = out_[:, :, 3]  #
        y_cur_3 = out_[:, :, 5]  #

        y_bef_2 = out_[:, :, 1]  #
        y_bef_1 = out_[:, :, 0]  #
        y_bef_3 = out_[:, :, 2]  #

        y_aft_2 = out_[:, :, 7]  #
        y_aft_1 = out_[:, :, 6]  #
        y_aft_3 = out_[:, :, 8]  #

        y_cur_ = torch.cat((y_cur_1[:, None, :], y_cur_2[:, None, :], y_cur_3[:, None, :]), dim=1)  #
        y_bef_ = torch.cat((y_bef_1[:, None, :], y_bef_2[:, None, :], y_bef_3[:, None, :]), dim=1)  #
        y_aft_ = torch.cat((y_aft_1[:, None, :], y_aft_2[:, None, :], y_aft_3[:, None, :]), dim=1)  #

        scores = self.attention(y_cur_)  #
        y_cur = torch.mul(y_cur_, scores[:, :, None])  #
        y_cur = torch.sum(y_cur, dim=1)  #

        scores = self.attention(y_bef_)  #
        y_bef = torch.mul(y_bef_, scores[:, :, None])  #
        y_bef = torch.sum(y_bef, dim=1)  #

        scores = self.attention(y_aft_)  #
        y_aft = torch.mul(y_aft_, scores[:, :, None])  #
        y_aft = torch.sum(y_aft, dim=1)  #

        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)

        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)  # N x 2048      

        if not is_train:
            return out, scores, out_
        else:
            y = torch.cat((out[:, None, :], out[:, None, :], out[:, None, :]), dim=1)
            return y, scores, out_


class SAFM(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(SAFM, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.nonlocalblock = TemporalEncoder(channel=2155)


    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature, scores, feature_seqlen = self.nonlocalblock(input, is_train=is_train) #
        feature = feature.reshape(-1, feature.size(-1))


        return feature,scores
        # smpl_output_Dm
