import torch
import torch.nn as nn
import numpy as np

from .tcfm import TCFM
from .safm import SAFM
from .pose_resnet import get_resnet_encoder
from ..core.config import cfg
from ..utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .iuv_predictor import IUV_predict_layer

import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, is_train=False, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }]
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }]
        return output


class STAF(nn.Module):
    """ STAF based Deep Regressor for Human Mesh Recovery
    STAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, seqlen=16, pretrained=True):
        super().__init__()
        self.seq_len = seqlen
        # self.feature_extractor = ResNet_Backbone(model=cfg.MODEL.PyMAF.BACKBONE, pretrained=pretrained)
        # deconv layers
        self.inplanes = 2048
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            self.maf_extractor.append(MAF_Extractor())
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]

        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            self.regressor.append(Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params))

        self.safm = SAFM(seqlen=self.seq_len)
        self.tcfm = TCFM(seqlen=self.seq_len, channel=grid_feat_len)
        # self.moca1 = TCFM(seqlen=self.seq_len,channel=ma_feat_len)
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 1))

        # dp_feat_dim = 256
        # if cfg.MODEL.PyMAF.AUX_SUPV_ON:
        #     self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, s_feat, J_regressor=None, is_train=False):
        mid_frame = self.seq_len // 2
        # print(s_feat.shape)
        mid_s_feat = s_feat.permute(0, 2, 3, 4, 1)
        # print(s_feat.shape)
        mid_s_feat = self.avgpool(mid_s_feat)
        # print(s_feat.shape)
        mid_s_feat = mid_s_feat.squeeze(-1)
        s_feat[:, mid_frame, :, :, :] = mid_s_feat

        batch_size = s_feat.shape[0]
        s_feat = s_feat.reshape(-1, 2048, 7, 7)
        assert 0 <= cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.PyMAF.N_ITER == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        smpl_output = self.regressor[0].forward_init(s_feat, J_regressor=J_regressor)
        # parameter predictions
        for rf_i in range(3):
            pred_cam = smpl_output[0]['pred_cam']
            pred_shape = smpl_output[0]['pred_shape']
            pred_pose = smpl_output[0]['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            if rf_i == 2:
                s_feat = s_feat.reshape(batch_size, self.seq_len, 256, 28, 28)
                s_feat = s_feat[:, self.seq_len // 2, :, :, :]
                if is_train:
                    s_feat = torch.cat([s_feat[:, None, :, :], s_feat[:, None, :, :], s_feat[:, None, :, :]], dim=1)
                s_feat = s_feat.reshape(-1, 256, 28, 28)

            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i

            # print(pred_cam.shape)
            # print(s_feat_i.shape)
            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size * self.seq_len, -1, -1), 1, 2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)
                ref_feature = ref_feature.reshape(batch_size, self.seq_len, -1)
                ref_feature = self.tcfm(ref_feature, is_train=is_train)
            elif rf_i == 1:
                pred_smpl_verts = smpl_output[0]['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0),
                                                  pred_smpl_verts)  # [B*N, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds)  # [B, 431 * n_feat]
                # ref_feature = ref_feature.reshape(batch_size, self.seq_len, -1)
                # ref_feature = self.moca1(ref_feature, is_train=is_train)
                ref_feature = ref_feature.reshape(batch_size, self.seq_len, -1)  # [B, N, 431 * n_feat]
                ref_feature, scores = self.safm(ref_feature, is_train=is_train)

                pred_cam = pred_cam.reshape(batch_size, self.seq_len, -1)
                pred_shape = pred_shape.reshape(batch_size, self.seq_len, -1)
                pred_pose = pred_pose.reshape(batch_size, self.seq_len, -1)
                if self.seq_len % 2 == 0:
                    pred_cam = pred_cam[:, int(self.seq_len / 2 - 1), :]
                    pred_shape = pred_shape[:, int(self.seq_len / 2 - 1), :]
                    pred_pose = pred_pose[:, int(self.seq_len / 2 - 1), :]
                else:
                    pred_cam = pred_cam[:, int(self.seq_len / 2), :]
                    pred_shape = pred_shape[:, int(self.seq_len / 2), :]
                    pred_pose = pred_pose[:, int(self.seq_len / 2), :]
                if is_train:
                    pred_cam = torch.cat((pred_cam[:, None, :], pred_cam[:, None, :], pred_cam[:, None, :]), dim=1)
                    pred_cam = pred_cam.reshape(-1, pred_cam.size(-1))
                    pred_shape = torch.cat((pred_shape[:, None, :], pred_shape[:, None, :], pred_shape[:, None, :]),
                                           dim=1)
                    pred_shape = pred_shape.reshape(-1, pred_shape.size(-1))
                    pred_pose = torch.cat((pred_pose[:, None, :], pred_pose[:, None, :], pred_pose[:, None, :]), dim=1)
                    pred_pose = pred_pose.reshape(-1, pred_pose.size(-1))
                pred_cam = pred_cam.detach()
                pred_shape = pred_shape.detach()
                pred_pose = pred_pose.detach()
            else:
                pred_smpl_verts = smpl_output[0]['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0),
                                                  pred_smpl_verts)  # [B*N, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds)  # [B*N, 431 * n_feat]


            smpl_output = self.regressor[rf_i](ref_feature, pred_pose, pred_shape, pred_cam, n_iter=1,
                                               J_regressor=J_regressor)

        scores=None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores

        else:
            repeat_num = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)
                s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3)
                s['scores'] = scores
        scores=None
        return smpl_output, scores


def staf(smpl_mean_params, pretrained=True):
    """ Constructs an STAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STAF(smpl_mean_params, pretrained)
    return model
