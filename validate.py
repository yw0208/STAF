import time

import joblib
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar
from torch.utils.data import DataLoader

from lib.core.config import BASE_DATA_DIR
from lib.dataset import ThreeDPW
from lib.models.staf import STAF
from lib.utils.eval_utils import batch_compute_similarity_transform_torch, compute_error_verts, compute_accel, \
    compute_error_accel
from lib.utils.utils import move_dict_to_device
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    generator = STAF(
        seqlen=9
    ).to('cuda')
    print('init weight...')
    checkpoint = torch.load('./data/base_data/3dpw_model_best.pth.tar')
    generator.load_state_dict(checkpoint['gen_state_dict'], strict=False)

    valid_db = ThreeDPW(load_opt='repr_table4_h36m_mpii3d_model', set='test', seqlen=9, overlap=8.0/9.0, debug=False)
    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=16,
        shuffle=False,
        num_workers=12
    )

    evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

    device = 'cuda'

    generator.eval()

    start = time.time()

    summary_string = ''

    bar = Bar('Validation', fill='#', max=len(valid_loader))

    if evaluation_accumulators is not None:
        for k, v in evaluation_accumulators.items():
            evaluation_accumulators[k] = []

    J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, target in enumerate(valid_loader):

        move_dict_to_device(target, device)

        # <=============
        with torch.no_grad():
            inp = target['imgs']
            batch = len(inp)

            preds,_ = generator(inp, J_regressor=J_regressor)  #

            # convert to 14 keypoint format for evaluation
            n_kp = preds[-1]['kp_3d'].shape[-2]
            pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
            target_theta = target['theta'].view(-1, 85).cpu().numpy()

            evaluation_accumulators['pred_verts'].append(pred_verts)
            evaluation_accumulators['target_theta'].append(target_theta)

            evaluation_accumulators['pred_j3d'].append(pred_j3d)
            evaluation_accumulators['target_j3d'].append(target_j3d)
        # =============>

        batch_time = time.time() - start

        summary_string = f'({i + 1}/{len(valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                         f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

        bar.suffix = summary_string
        bar.next()

    bar.finish()

    logger.info(summary_string)

    for k, v in evaluation_accumulators.items():
        evaluation_accumulators[k] = np.vstack(v)

    pred_j3ds = evaluation_accumulators['pred_j3d']
    target_j3ds = evaluation_accumulators['target_j3d']

    pred_j3ds = torch.from_numpy(pred_j3ds).float()
    target_j3ds = torch.from_numpy(target_j3ds).float()

    print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
    pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
    target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    pred_verts = evaluation_accumulators['pred_verts']
    target_theta = evaluation_accumulators['target_theta']

    m2mm = 1000

    pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
    # accel = np.mean(compute_accel(pred_j3ds)) * m2mm
    accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
    joblib.dump(errors,'staf-mpjpe')
    mpjpe = np.mean(errors) * m2mm
    pa_mpjpe = np.mean(errors_pa) * m2mm

    eval_dict = {
        'mpjpe': mpjpe,
        'pa-mpjpe': pa_mpjpe,
        # 'accel': accel,
        'pve': pve,
        'accel_err': accel_err
    }
    print(eval_dict)