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


class PyMAF_ResNet(nn.Module):
    """ STAF based Deep Regressor for Human Mesh Recovery
    STAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.feature_extractor = ResNet_Backbone(model=cfg.MODEL.PyMAF.BACKBONE, pretrained=pretrained)

    def forward(self, x, J_regressor=None, is_train=False):
        s_feat,_ = self.feature_extractor(x)
        return s_feat


def pymaf_net(smpl_mean_params, pretrained=True):
    """ Constructs an STAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF_ResNet(smpl_mean_params, pretrained)
    return model
