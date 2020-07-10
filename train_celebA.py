import configargparse
import os, time, datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np

from dataset.face_dataset import CelebAMaskDataset

from torch.utils.data import DataLoader, SubsetRandomSampler
from modeling import SRNsModel
import util
import random
import re

import shutil

from datetime import datetime

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    s = s.replace('"', '')

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Multi-resolution training: Instead of passing only a single value, each of these command-line arguments take comma-
# separated lists. If no multi-resolution training is required, simply pass single values (see default values).
p.add_argument('--img_sidelengths', type=_parse_num_range, default='128', required=False,
               help='Progression of image sidelengths.'
                    'If comma-separated list, will train on each sidelength for respective max_steps.'
                    'Images are downsampled to the respective resolution.')
p.add_argument('--output_sidelength', type=int, default=512, required=False,
               help='Target resolution.')
p.add_argument('--out_channels', type=int, default=3, required=False,
               help='Output channels.')
p.add_argument('--max_steps_per_img_sidelength', type=_parse_num_range, default="200000",
               help='Maximum number of optimization steps.'
                    'If comma-separated list, is understood as steps per image_sidelength.')
p.add_argument('--batch_size_per_img_sidelength', type=_parse_num_range, default="8",
               help='Training batch size.'
                    'If comma-separated list, will train each image sidelength with respective batch size.')

# Training options
p.add_argument('--num_instance', type=int, default="8", help='num instance in code dict.')
p.add_argument('--data_root', type=str, default='/data/anpei/facial-data/seg_face_8000', help='Path to directory with training data.')
p.add_argument('--logging_root', type=str, default='/mnt/new_disk2/liury/log/SRNs',
               required=False, help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--data_type', type=str, default='seg', help='Identifier to image files.')

p.add_argument('--lr    ', type=float, default=5e-5, help='learning rate. default=5e-5')

p.add_argument('--geo_weight', type=float, default=-1.0, help='weight for depth')
p.add_argument('--l1_weight', type=float, default=200,
               help='Weight for l1 loss term (lambda_img in paper).')
p.add_argument('--kl_weight', type=float, default=1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--reg_weight', type=float, default=1e-3,
               help='Weight for depth regularization term (lambda_depth in paper).')

p.add_argument('--steps_til_ckpt', type=int, default=10000,
               help='Number of iterations until checkpoint is saved.')
p.add_argument('--steps_til_val', type=int, default=1000,
               help='Number of iterations until validation set is run.')
p.add_argument('--no_validation', action='store_true', default=False,
               help='If no validation set should be used.')

p.add_argument('--preload', action='store_true', default=False,
               help='Whether to preload data to RAM.')

p.add_argument('--opt_cam', action='store_true', default=False,
               help='Whether to optimaze camera pose.')

p.add_argument('--checkpoint_path', default=None,
               help='Checkpoint to trained model.')
p.add_argument('--overwrite_embeddings', action='store_true', default=False,
               help='When loading from checkpoint: Whether to discard checkpoint embeddings and initialize at random.')
p.add_argument('--start_step', type=int, default=0,
               help='If continuing from checkpoint, which iteration to start counting at.')

p.add_argument('--sample_observations_train', type=_parse_num_range, default=None,
               help='Only pick a subset of specific observations for each instance.')
p.add_argument('--sample_observations_val', type=_parse_num_range, default=None,
               help='Only pick a subset of specific observations for each instance.')

p.add_argument('--sample_instances_train', type=_parse_num_range, default=None,
               help='Only pick a subset of all instances.')
p.add_argument('--sample_instances_val', type=_parse_num_range, default=None,
               help='Only pick a subset of all instances.')

# Model options
p.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester.')
p.add_argument('--freeze_networks', action='store_true',
               help='Whether to freeze weights of all networks in SRN (not the embeddings!).')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--use_encoder', action='store_true',
               help='Whether to use a resnet based image encoder')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')

opt = p.parse_args()

# print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    torch.autograd.set_detect_anomaly(True)
    img_sidelengths = opt.img_sidelengths
    output_sidelength = opt.output_sidelength

    batch_size_per_sidelength = opt.batch_size_per_img_sidelength
    max_steps_per_sidelength = opt.max_steps_per_img_sidelength

    # random pick tran/val dataset
    print('Loading train dataset: %s'%(opt.data_root))

    train_dataset = CelebAMaskDataset(
        root_dir=opt.data_root,
        img_sidelength=output_sidelength)

    print('[DONE] load train dataset.', len(train_dataset))

    assert (len(img_sidelengths) == len(batch_size_per_sidelength)), \
        "Different number of image sidelengths passed than batch sizes."
    assert (len(img_sidelengths) == len(max_steps_per_sidelength)), \
        "Different number of image sidelengths passed than max steps."

    print('Init SRN model.')
    
    # if opt.animated > 0:
    #     batch_size = batch_size_per_sidelength[0] * 2
    #     sample_size = len(opt.sample_observations_train) // batch_size
    #     sample_frames = np.random.randint(batch_size, size=sample_size) + np.arange(sample_size) * batch_size
    # else:
    #     sample_frames = None

    model = SRNsModel(num_instances=opt.num_instance,
                        latent_dim=opt.embedding_size,
                        tracing_steps=opt.tracing_steps,       
                        freeze_networks=opt.freeze_networks,
                        out_channels=opt.out_channels,
                        img_sidelength=img_sidelengths[0],
                        output_sidelength=output_sidelength,
                        opt_cam=opt.opt_cam)

    if opt.checkpoint_path is not None:
        print("> Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=opt.overwrite_embeddings)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.train()

    print('[DONE] Init SRN model.')

    now = datetime.now()
    task_name = os.path.splitext(os.path.basename(opt.config_filepath))[0]
    log_dir = os.path.join(opt.logging_root, now.strftime('%m%d%H')+task_name)

    print('Logging dir = ', log_dir)

    ckpt_dir = os.path.join(log_dir, 'latent_codes')
    events_dir = os.path.join(log_dir, 'events')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)

    if os.path.exists(os.path.join(opt.data_root, 'indexing.txt')):
        shutil.move(
            os.path.join(opt.data_root, 'indexing.txt'), 
            os.path.join(ckpt_dir, 'indexing.txt'))

    # Save command-line parameters log directory.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(log_dir, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(events_dir)

    print('Beginning training...')
    # This loop implements training with an increasing image sidelength.
    cum_max_steps = 0  # Tracks max_steps cumulatively over all image sidelengths.
    for img_sidelength, max_steps, batch_size in zip(img_sidelengths, max_steps_per_sidelength,
                                                     batch_size_per_sidelength):
        print("\n" + "#" * 10)
        print("Training with sidelength %d for %d steps with batch size %d" % (img_sidelength, max_steps, batch_size))
        print("#" * 10 + "\n")
        train_dataset.set_img_sidelength(output_sidelength)

        # Need to instantiate DataLoader every time to set new batch size.
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      collate_fn=train_dataset.collate_fn,
                                      pin_memory=opt.preload)
        cum_max_steps += max_steps

        prev_loss = 0.0

        while True:
            for model_input, ground_truth in train_dataloader:
                step = 0
                ins_idx = model_input['instance_idx'].cpu().numpy()
                print('Process instances: %s.'%'\t'.join(model_input['data_id']))

                pose = model_input['pose']
                intrinsics = model_input['intrinsics']
                uv = model_input['uv']

                try:
                    z = model.get_embedding(model_input)
                except AttributeError:
                    z = model.module.get_embedding(model_input)

                optimizer.zero_grad()
                model_outputs = model(pose, z, intrinsics, uv)
                # print('*** model_output = ', model_outputs[0].shape, model_outputs[1].shape)

                # calc mIoU
                pred_seg = F.one_hot(torch.argmax(model_outputs[0], dim=2, keepdim=False).long())
                trgt_seg = F.one_hot(ground_truth['rgb'].squeeze(-1).long())

                mIoU = torch.mean(torch.div(
                    torch.sum(pred_seg&trgt_seg, dim=0).float()+1e-8,
                    torch.sum(pred_seg|trgt_seg, dim=0).float()+1e-8)).cpu().numpy()
                print('*** mIoU = ', mIoU)

                try:
                    total_loss = model.get_loss(model_outputs, model_input, opt)
                except AttributeError:
                    total_loss = model.module.get_loss(model_outputs, model_input, opt)

                total_loss.backward()
                optimizer.step()

                if (step % 50) == 0:
                    print("> [%06d] Loss %.4f  Prev_Loss %.4f Camera " % (step, total_loss, prev_loss), z.data[0, :3])

                if (step % 1000) == 0:
                    try:
                        model.write_updates(
                            writer, model_outputs, ground_truth, step, mode="train", color_map=train_dataset.color_map)
                    except AttributeError:
                        model.module.write_updates(
                            writer, model_outputs, ground_truth, step, mode="train", color_map=train_dataset.color_map)

            if (total_loss < 10.0): 
                print("> [DONE] Step %07d  Loss %.4f  Prev_Loss %.4f. Saving to files:" %
                        (step, total_loss, prev_loss))
            
                batch_params = z.clone()
                batch_params = batch_params.detach().cpu().numpy()
                for idx in range(batch_params.shape[0]):
                    output_fp = os.path.join(
                        ckpt_dir, 'param_%s.npy' % (model_input['data_id'][idx]))
                    print('\t %02d: %s' % (idx, output_fp))
                    np.save(output_fp, batch_params[idx])

                step = 0
                try:
                    model.reset_cam()
                except AttributeError:
                    model.module.reset_cam()

                optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
                break

                prev_loss = total_loss
                step += 1


def main():
    train()


if __name__ == '__main__':
    main()
