import torch
import torch.nn as nn
import numpy as np
import bisect

import torchvision
import util

import skimage.measure
from torch.nn import functional as F

import torchvision.models as models

from layers.pytorch_prototyping import *
from layers import geometry
from layers import hyperlayers
from layers.ray_sampler import SIFRenderer

from itertools import chain


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class SIFModel(nn.Module):

    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 feat_dim=None,
                 use_unet_renderer=False,
                 use_encoder=False,
                 freeze_networks=False,
                 animated=False,
                 sample_frames=None,
                 opt_cam=False,
                 out_channels=3,
                 img_sidelength=128,
                 output_sidelength=128):
        super(SIFModel, self).__init__()

        self.latent_dim = latent_dim
        self.opt_cam = opt_cam

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.freeze_networks = freeze_networks
        self.out_channels = out_channels
        self.img_sidelength = img_sidelength
        self.output_sidelength = output_sidelength
        self.num_instances = num_instances

        # List of logs
        self.logs = list()

        if self.opt_cam:
            self.latent_codes = nn.Embedding(
                self.num_instances, self.latent_dim).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
            self.cam_pose = nn.Embedding(self.num_instances, 12).cuda()
            self.cam_pose.weight.data.copy_(
                torch.eye(4)[:3, :].view(-1).unsqueeze(0).expand(self.num_instances, -1))
            self.get_embedding = lambda x: torch.cat([
                self.cam_pose(x['instance_idx'].long().cuda()),
                self.latent_codes(x['instance_idx'].long().cuda())], axis=1)

        else:
            self.latent_codes = nn.Embedding(
                self.num_instances, self.latent_dim).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
            self.get_embedding = lambda x: self.latent_codes(
                x['instance_idx'].long().cuda())
            self.logs.append(("embedding", "", self.latent_codes.weight, 1000))

        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=self.latent_dim,
                                             hidden_ch=self.num_hidden_units_phi,
                                             num_hidden_layers=self.phi_layers - 2,
                                             in_ch=3,
                                             out_ch=self.num_hidden_units_phi)

        self.pixel_generator = SIFRenderer(
            self.num_instances, self.out_channels, self.latent_dim)

        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.hyper_phi.parameters()))

            for param in all_network_params:
                param.requires_grad = False

        # Losses
        self.loss_fn = self.pixel_generator.loss_fn

    
    def get_cls_loss(self, prediction, ground_truth, re_map=None):

        pred_imgs, _ = prediction
        pred_imgs = pred_imgs.permute(0, 2, 1)
        trgt_imgs = ground_truth['rgb'].squeeze(-1).long()
        trgt_imgs = trgt_imgs.cuda()

        return self.loss_fn(pred_imgs, trgt_imgs)


    def write_updates(self, writer, predictions, ground_truth=None, iter=0, mode="", color_map=None):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        predictions, depth_maps = predictions
        batch_size, _, channels = predictions.shape

        if not channels in [1, 3]:
            # classification
            predictions = torch.argmax(predictions, dim=2).long().unsqueeze(2)

        if ground_truth is not None:
            trgt_imgs = ground_truth['rgb']
            trgt_imgs = trgt_imgs.cuda()
            if not channels in [1, 3]:
                trgt_imgs = trgt_imgs.long()
            assert predictions.shape == trgt_imgs.shape
        else:
            trgt_imgs = None

        prefix = mode + '/'

        # Module"s own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type == "image":
                    writer.add_image(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "figure":
                    writer.add_figure(name, content, iter, close=True)
                elif type == "histogram":
                    writer.add_histogram(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "scalar":
                    writer.add_scalar(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "embedding" and (mode == 'train'):
                    writer.add_embedding(mat=content, global_step=iter)
                elif type == "mesh":
                    vert, color = util.mat2mesh(content.detach().cpu().numpy())
                    writer.add_mesh(name, vertices=vert, colors=color)

        if (iter % 500 == 0) or (mode == 'test'):
            output_vs_gt = torch.cat(
                (predictions, trgt_imgs), dim=0) if trgt_imgs is not None else predictions
            output_vs_gt = util.lin2img(output_vs_gt, color_map)

            # print('*** output_vs_gt = ', output_vs_gt.shape, output_vs_gt.dtype)

            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            if trgt_imgs is not None:
                rgb_loss = ((predictions.float().cuda(
                ) - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
                rgb_loss = util.lin2img(rgb_loss)

                fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
                                        for i in range(batch_size)])
                writer.add_figure(prefix + "rgb_error_fig",
                                  fig,
                                  iter,
                                  close=True)

                # writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
                # writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

            depth_maps_plot = util.lin2img(depth_maps)
            writer.add_image(prefix + "pred_depth",
                             torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                                         scale_each=True,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)
            if 'depth' in ground_truth.keys():
                trgt_depth = ground_truth['depth'].float().cuda()
                pred_depth = depth_maps * (predictions != 0).float().cuda()
                geo_loss = torch.abs(pred_depth - trgt_depth)

                geo_loss = util.lin2img(geo_loss)
                fig = util.show_images([geo_loss[i].detach().cpu().numpy().squeeze()
                                        for i in range(batch_size)])
                writer.add_figure(prefix + "depth_error_fig",
                                  fig,
                                  iter,
                                  close=True)

        # writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        # writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        if iter and hasattr(self, 'latent_reg_loss'):
            writer.add_scalar(prefix + "latent_reg_loss",
                              self.latent_reg_loss, iter)


    def forward(self, pose, z, intrinsics, uv, device=None, auc_input=None):
        pose = pose.cuda()
        intrinsics = intrinsics.cuda()
        uv = uv.cuda().float()
        self.z = z.cuda()
        self.latent_reg_loss = torch.mean(self.z ** 2)

        if self.opt_cam:
            pose.requires_grad = True
            rel_pose = torch.eye(4).expand_as(pose).cuda()
            rel_pose[:, :3, :] = (self.z[:, :12]).view(-1, 3, 4)
            pose = torch.bmm(rel_pose, pose)
            self.z = self.z[:, 12:]

        phi = self.hyper_phi(self.z)
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)

        v = phi(points_xyz)

        novel_views = self.pixel_generator(v)

        if self.output_sidelength != self.img_sidelength:
            novel_views = novel_views.permute(0, 2, 1).view(
                -1, self.out_channels, self.img_sidelength, self.img_sidelength)
            novel_views = F.interpolate(
                novel_views, size=(self.output_sidelength, self.output_sidelength), mode='bilinear', align_corners=False)
            novel_views = novel_views.view(-1, self.out_channels,
                                           self.output_sidelength**2).permute(0, 2, 1)

            # print('*** forward - up_scaling ', torch.cuda.memory_allocated()-cur_mem)
            cur_mem = torch.cuda.memory_allocated()

        # Calculate normal map
        if self.training:
            # log saves tensors that"ll receive summaries when model"s write_updates function is called
            self.logs = list()
            self.logs.extend(log)
            with torch.no_grad():
                batch_size = uv.shape[0]
                x_cam = uv[:, :, 0].view(batch_size, -1)
                y_cam = uv[:, :, 1].view(batch_size, -1)
                z_cam = depth_maps.view(batch_size, -1)

                normals = geometry.compute_normal_map(
                    x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
                self.logs.append(("image", "normals",
                                  torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

            self.logs.append(("histogram", "embedding", self.z, 1000))

        return novel_views, depth_maps
