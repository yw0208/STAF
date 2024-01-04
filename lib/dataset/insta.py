# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import os

import cv2
import h5py
import joblib
import torch
import logging
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import TCMR_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, split_into_chunks
from lib.dataset._dataset_3d import IMG_NORM_MEAN, IMG_NORM_STD

logger = logging.getLogger(__name__)


class Insta(Dataset):
    def __init__(self, load_opt, seqlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.mid_frame = int(seqlen / 2)
        self.stride = int(seqlen * (1 - overlap) + 0.5)
        self.h5_file = osp.join(TCMR_DB_DIR, 'insta_train_db.pt')
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

        self.db = joblib.load(self.h5_file)
        self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride)

        print(f'InstaVariety number of dataset objects {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index + 1]
        else:
            return data[start_index:start_index + 1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
        kp_2d = convert_kps(kp_2d, src='insta', dst='spin')
        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)

        vid_name = self.get_sequence(start_index, end_index, self.db['vid_name'])
        frame_id = self.get_sequence(start_index, end_index, self.db['frame_id']).astype(str)
        path_list = self.get_path_list(vid_name, frame_id)
        img_sequence_list = self.get_img_sequence(path_list)
        img_sequence = torch.from_numpy(np.array(img_sequence_list))

        for idx in range(self.seqlen):
            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        repeat_num = 3
        target = {
            'imgs': img_sequence,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1),
            # 2D keypoints transformed according to bbox cropping
            # 'instance_id': instance_id
        }

        return target

    def get_img_sequence(self, img_paths):
        imgs_tensor_list = []
        for path in img_paths:
            img = joblib.load(path)
            img = img['s_feat']
            imgs_tensor_list.append(img)
        return imgs_tensor_list

    def rgb_processing(self, rgb_img):
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def get_path_list(self, vid_names, frame_ids):
        img_path_list = []
        for vid_name, frame_id in zip(vid_names, frame_ids):
            path = '/opt/data/private/datasets/insta_img_feature/train'
            vid_name_list = vid_name.decode().split('/')[-1].split('-')
            videos = vid_name_list[0] + '-' + vid_name_list[1]
            video = vid_name_list[-1]
            imgname = str(frame_id)
            feature_path = os.path.join(path, videos, video, imgname)
            img_path_list.append(feature_path + '.pt')

        return img_path_list
