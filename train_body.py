import configargparse
import os, time, datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset.body_dataset import BodyPartDataset

from torch.utils.data import DataLoader

from modeling import SIFModel

from modeling import util
import random
import re

from datetime import datetime

_NUM_OBSERVATIONS = 25

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
p.add_argument('--data_root', type=str, default='/data/anpei/facial-data/seg_face_8000', help='Path to directory with training data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--data_type', type=str, default='seg', help='Identifier to image files.')

p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')

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
    no_validation = opt.no_validation

    # random pick tran/val dataset
    print('Loading train dataset: %s'%(opt.data_root))

    if opt.sample_observations_train == [-1]:
        opt.sample_observations_train = [random.choice(list(range(_NUM_OBSERVATIONS)))]

    train_dataset = BodyPartDataset(
        root_dir=opt.data_root, 
        data_type=opt.data_type,
        img_sidelength=output_sidelength, 
        sample_observations=opt.sample_observations_train, 
        sample_instances=opt.sample_instances_train)

    print('[DONE] load train dataset.', len(train_dataset))

    assert (len(img_sidelengths) == len(batch_size_per_sidelength)), \
        "Different number of image sidelengths passed than batch sizes."
    assert (len(img_sidelengths) == len(max_steps_per_sidelength)), \
        "Different number of image sidelengths passed than max steps."

    if not no_validation:
        if opt.sample_instances_val is None:
            opt.sample_instances_val = opt.sample_instances_train

        if opt.sample_observations_val is None:
            opt.sample_observations_val = list(set(range(_NUM_OBSERVATIONS)) - set(opt.sample_observations_train))
            opt.sample_observations_val = [random.choice(opt.sample_observations_val)]

        val_dataset = BodyPartDataset(
            root_dir=opt.data_root, 
            data_type=opt.data_type,
            img_sidelength=output_sidelength,
            sample_observations=opt.sample_observations_val,
            sample_instances=opt.sample_instances_val)
        print('[DONE] load val dataset.', len(val_dataset))
        
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=8,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=val_dataset.collate_fn)

    print('Init SRN model.')
    model = SRNsModel(num_instances=train_dataset.num_instances,
                      latent_dim=opt.embedding_size,
                      fit_single_srn=opt.fit_single_srn,
                      use_unet_renderer=opt.use_unet_renderer,
                      use_encoder=opt.use_encoder,
                      tracing_steps=opt.tracing_steps,
                      freeze_networks=opt.freeze_networks,
                      out_channels=opt.out_channels,
                      img_sidelength=img_sidelengths[0],
                      output_sidelength=output_sidelength)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    model.train()

    if opt.checkpoint_path is not None:
        print("> Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=opt.overwrite_embeddings)

    print('[DONE] Init SRN model.')

    now = datetime.now()
    task_name = os.path.splitext(os.path.basename(opt.config_filepath))[0]
    log_dir = os.path.join(opt.logging_root, now.strftime('%m%d%H')+task_name)
    
    print('Logging dir = ', log_dir)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    events_dir = os.path.join(log_dir, 'events')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)

    # Save command-line parameters log directory.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(log_dir, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(events_dir)
    iter = opt.start_step
    epoch = iter // len(train_dataset)
    step = 0

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
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=train_dataset.collate_fn,
                                      pin_memory=opt.preload)
        cum_max_steps += max_steps

        # Loops over epochs.
        while True:
            for model_input, ground_truth in train_dataloader:
                
                pose = model_input['pose']
                intrinsics = model_input['intrinsics']
                uv = model_input['uv']
                if 'params' in model_input.keys() and model_input['params'] is not None:
                    z = model_input['params']
                else:
                    z = model.get_embedding(model_input)

                model_outputs = model(pose, z, intrinsics, uv)

                optimizer.zero_grad()

                if opt.out_channels in [1, 3]:
                    dist_loss = model.get_image_loss(model_outputs, ground_truth)
                else:
                    dist_loss = model.get_cls_loss(model_outputs, ground_truth)

                reg_loss = model.get_regularization_loss(model_outputs, ground_truth)
                latent_loss = model.get_latent_loss()

                weighted_dist_loss = opt.l1_weight * dist_loss
                weighted_reg_loss = opt.reg_weight * reg_loss
                weighted_latent_loss = opt.kl_weight * latent_loss

                total_loss = (weighted_dist_loss
                              + weighted_reg_loss
                              + weighted_latent_loss)

                total_loss.backward()

                optimizer.step()

                # print("Iter %07d   Epoch %03d   L_img %0.4f   L_latent %0.4f   L_depth %0.4f" %
                #       (iter, epoch, weighted_dist_loss, weighted_latent_loss, weighted_reg_loss))

                model.write_updates(writer, model_outputs, ground_truth, iter, mode="train", color_map=train_dataset.color_map)
                writer.add_scalar("Loss/scaled_distortion_loss", weighted_dist_loss, iter)
                writer.add_scalar("Loss/scaled_regularization_loss", weighted_reg_loss, iter)
                writer.add_scalar("Loss/scaled_latent_loss", weighted_latent_loss, iter)
                writer.add_scalar("Loss/total_loss", total_loss, iter)

                if iter % opt.steps_til_val == 0 and not no_validation:
                    print("[%06d] Running validation set..."%(iter))

                    model.eval()
                    with torch.no_grad():
                        dist_losses = []
                        geo_losses = []
                        for model_input, ground_truth in val_dataloader:
                            pose = model_input['pose']
                            intrinsics = model_input['intrinsics']
                            uv = model_input['uv']
                            if 'params' in model_input.keys() and model_input['params'] is not None:
                                z = model_input['params']
                            else:
                                z = model.get_embedding(model_input)

                            model_outputs = model(pose, z, intrinsics, uv)

                            if opt.out_channels in [1, 3]:
                                dist_loss = model.get_image_loss(model_outputs, ground_truth)
                            else:
                                dist_loss = model.get_cls_loss(model_outputs, ground_truth)

                            dist_loss = dist_loss.cpu().numpy()
                            dist_losses.append(dist_loss)

                            model.write_updates(writer, model_outputs, ground_truth, iter, mode='val', color_map=val_dataset.color_map)
                        writer.add_scalar("val/dist_loss", np.mean(dist_losses), iter)
                        if geo_losses: writer.add_scalar("val/geo_loss", np.mean(geo_losses), iter)

                    model.train()

                iter += 1
                step += 1

                if iter == cum_max_steps:
                    break

                if iter % opt.steps_til_ckpt == 0:
                    util.custom_save(model,
                                     os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                                     discriminator=None,
                                     optimizer=optimizer)

            if iter == cum_max_steps:
                break
            epoch += 1

    util.custom_save(model,
                     os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                     discriminator=None,
                     optimizer=optimizer)


def main():
    train()


if __name__ == '__main__':
    main()
