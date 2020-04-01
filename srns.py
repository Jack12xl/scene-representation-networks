import torch
import torch.nn as nn
import numpy as np

import torchvision
import util

import skimage.measure
from torch.nn import functional as F

import torchvision.models as models

from pytorch_prototyping import *
import custom_layers
import geometry
import hyperlayers

from itertools import chain

class SegEncoder(nn.Module):
    def __init__(   self, 
                    img_sidelength=128,
                    num_classes=19,
                    latent_dim=256,
                    kernel_size=7,
                    shortcut=None):
        
        super().__init__()

        self.num_classes = num_classes
        self.img_sidelength = img_sidelength
        self.latent_dim = latent_dim

        if shortcut is not None:
            self.shortcut = shortcut
        else:
            self.shortcut = torch.empty(self.latent_dim).cuda()
            nn.init.normal_(self.shortcut, mean=0.0, std=1.0)
        
        n_hidden = 128

        ks = kernel_size
        pw = ks // 2

        self._emb = nn.Sequential(
            nn.Conv2d(
                num_classes, n_hidden, kernel_size=ks, padding=pw),
            nn.InstanceNorm2d(num_features=n_hidden,affine=False),
            nn.ReLU()
        )

        self._gamma = nn.Sequential(
            nn.Conv2d(n_hidden, self.latent_dim, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self._beta = nn.Sequential( 
            nn.Conv2d(n_hidden, self.latent_dim, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        seg = x
        # print('**** seg = ', seg.shape)
        seg = F.one_hot(seg.squeeze().long(), num_classes=self.num_classes).permute(0, 2, 1)
        # print('**** seg_one_hot = ', seg.shape)

        seg = seg.view(-1, self.num_classes, self.img_sidelength, self.img_sidelength).float()

        embedding = self._emb(seg)
        gamma = self._gamma(embedding).squeeze()
        beta = self._beta(embedding).squeeze()

        # print('*** gamma = ', gamma.shape)
        # print('*** beta = ', beta.shape)

        out = self.shortcut * (1 + gamma) + beta

        # print('**** out = ', out.shape)

        return out

class ConvEncoder(nn.Module):
    def __init__(   self, 
                    img_sidelength=128,
                    num_classes=19,
                    latent_dim=256):
        
        super().__init__()

        self.img_sidelength = img_sidelength
        self.latent_dim = latent_dim  
        self.num_classes = num_classes      

        self.encoder = models.resnet18(pretrained=False)
        self.activation = nn.ReLU()
        self.encoder.fc = nn.Identity()

        if latent_dim < 512:
            self.encoder.layer4= nn.Identity()
        
        self.encoder.conv1 = nn.Conv2d(num_classes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.activation = nn.Tanh()

    def forward(self, x):
        seg = x
        seg = F.one_hot(seg.squeeze().long(), num_classes=self.num_classes).permute(0, 2, 1)
        seg = seg.view(-1, self.num_classes, self.img_sidelength, self.img_sidelength).float()

        embedding = self.encoder(seg)
        embedding = self.activation(embedding)

        return embedding


class SRNsModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 has_params=False,
                 fit_single_srn=False,
                 use_unet_renderer=False,
                 use_image_encoder=False,
                 freeze_networks=False,
                 out_channels=3,
                 img_sidelength=128,
                 output_sidelength=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.has_params = has_params

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps
        self.freeze_networks = freeze_networks
        self.fit_single_srn = fit_single_srn
        self.out_channels = out_channels
        self.img_sidelength = img_sidelength
        self.output_sidelength = output_sidelength
        self.num_instances = num_instances

        # List of logs
        self.logs = list()

        if self.fit_single_srn:  # Fit a single scene with a single SRN (no hypernetworks)
            self.phi = FCBlock( hidden_ch=self.num_hidden_units_phi,
                                num_hidden_layers=self.phi_layers - 2,
                                in_features=3,
                                out_features=self.num_hidden_units_phi)
        else:
            # Auto-decoder: each scene instance gets its own code vector z
            if use_image_encoder:
                self.image_encoder = ConvEncoder(img_sidelength=output_sidelength)
                self.get_embedding = lambda x: self.image_encoder(x['rgb'].cuda())
            else:
                self.latent_codes = nn.Embedding(self.num_instances, self.latent_dim).cuda()
                nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
                self.get_embedding = lambda x: self.latent_codes(x['instance_idx'].long().cuda())
                if not self.fit_single_srn: self.logs.append(("embedding", "", self.latent_codes.weight, 500))

            self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                                 hyper_num_hidden_layers=1,
                                                 hyper_hidden_ch=self.latent_dim,
                                                 hidden_ch=self.num_hidden_units_phi,
                                                 num_hidden_layers=self.phi_layers - 2,
                                                 in_ch=3,
                                                 out_ch=self.num_hidden_units_phi)

        self.ray_marcher = custom_layers.Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                                    raymarch_steps=self.sphere_trace_steps)

        if use_unet_renderer:
            self.pixel_generator = custom_layers.DeepvoxelsRenderer(
                nf0=32, in_channels=self.num_hidden_units_phi,
                input_resolution=self.img_sidelength, img_sidelength=self.img_sidelength,
                out_channels=self.out_channels)
        else:
            self.pixel_generator = FCBlock(hidden_ch=self.num_hidden_units_phi,
                        num_hidden_layers=self.rendering_layers - 1,
                        in_features=self.num_hidden_units_phi,
                        out_features=self.out_channels,
                        outermost_linear=True)

        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.ray_marcher.parameters())
                                  + list(self.hyper_phi.parameters()))

            for param in all_network_params:
                param.requires_grad = False

        # Losses
        self.l2_loss = nn.MSELoss(reduction="mean")

        print(self)
        print("Number of parameters:")
        util.print_network(self)

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        """
        _, depth = prediction

        # print('*** depth = ', depth.shape, torch.zeros_like(depth).shape)
        # print(torch.max(depth), torch.min(depth))

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_cls_loss(self, prediction, ground_truth):
        
        xent_loss = nn.CrossEntropyLoss(reduction='mean')

        # print('**** get_cls_loss: ')
        # print(prediction[0].shape, ground_truth['rgb'].shape, np.unique(ground_truth['rgb'].detach().cpu().numpy()))

        pred_imgs, _ = prediction
        pred_imgs = pred_imgs.permute(0, 2, 1)

        trgt_imgs = ground_truth['rgb'].squeeze(-1).long()
        trgt_imgs = trgt_imgs.cuda()

        # print(pred_imgs.shape, trgt_imgs.shape, np.unique(trgt_imgs.detach().cpu().numpy()))

        # compute softmax_xent loss
        loss = xent_loss(pred_imgs, trgt_imgs)

        return loss

    def get_geo_loss(self, prediction, ground_truth):
        assert 'depth' in ground_truth.keys(), 'GT depth does not exist.'

        geo_loss = nn.SmoothL1Loss(reduction='none')

        trgt_depth = ground_truth['depth'].cuda()

        pred_img, pred_depth = prediction

        nonzero_cnt = 1.0
        
        if pred_img.shape[-1] in [1, 3]:
            pred_img = torch.argmax(pred_img, dim=2).long().unsqueeze(2)
            pred_depth = pred_depth * (pred_img != 0).float()
        
        loss = geo_loss(pred_depth, trgt_depth)
        nonzero_cnt = (loss > 1e-12).sum(axis=1).clamp(min=1).float()

        # return torch.mean(loss.sum(axis=1) / nonzero_cnt) 
        return torch.mean(loss.sum() / nonzero_cnt)

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        if self.fit_single_srn:
            self.latent_reg_loss = 0
        else:
            self.latent_reg_loss = torch.mean(self.z ** 2)

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        """Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, depth_maps = prediction

        batch_size = predictions.shape[0]

        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode="constant", value=1.)

        predictions = util.lin2img(predictions)

        if ground_truth is not None:
            trgt_imgs = ground_truth["rgb"]
            trgt_imgs = util.lin2img(trgt_imgs)

            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).numpy()

    def get_output_img(self, prediction):
        pred_imgs, _ = prediction
        return util.lin2img(pred_imgs)

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
            if not channels in [1, 3]: trgt_imgs = trgt_imgs.long()
            assert predictions.shape == trgt_imgs.shape
        else:
            trgt_imgs = None

        prefix = mode + '/'

        # Module"s own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type == "image":
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                elif type == "figure":
                    writer.add_figure(name, content, iter, close=True)
                elif type == "histogram":
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type == "scalar":
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type == "embedding" and (mode == 'train'):
                        writer.add_embedding(mat=content, global_step=iter)

        if (iter % 1000 == 0) or (mode == 'test'):
            output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0) if trgt_imgs is not None else predictions
            output_vs_gt = util.lin2img(output_vs_gt, color_map)

            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            if trgt_imgs is not None:
                rgb_loss = ((predictions.float().cuda() - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
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
            writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)

    def forward(self, pose, z, intrinsics, uv, device=None):
        self.logs = list() # log saves tensors that"ll receive summaries when model"s write_updates function is called

        # Parse model input.
        # instance_idcs = input["instance_idx"].long().cuda()
        # pose = input["pose"].cuda()
        # intrinsics = input["intrinsics"].cuda()
        # uv = input["uv"].cuda().float()

        pose = pose.cuda()
        intrinsics = intrinsics.cuda()
        uv = uv.cuda().float()
        self.z = z.cuda()

        # print('*** 0: process input - ', instance_idcs.shape, pose.shape, intrinsics.shape, uv.shape)

        if self.fit_single_srn:
            phi = self.phi
        else:
            # if z is not None:
            #     self.z = z
            # elif 'param' in input.keys():

            # if self.has_params: # If each instance has a latent parameter vector, we"ll use that one.
            #     if z is None:
            #         self.z = input["param"].cuda()
            #     else:
            #         self.z = z
            # else: # Else, we"ll use the embedding.
            #     # print('*** 1: embedding - ', self.latent_dim, self.num_instances, instance_idcs)
            #     self.z = self.latent_codes(instance_idcs)

            # print('*** 1: build phi - ', self.z.shape, torch.max(self.z), torch.min(self.z), self.fit_single_srn)
            phi = self.hyper_phi(self.z) # Forward pass through hypernetwork yields a (callable) SRN.

        # print('*** 2: ray_marcher.')

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)
        self.logs.extend(log)

        # Sapmle phi a last time at the final ray-marched world coordinates.
        v = phi(points_xyz)

        # Translate features at ray-marched world coordinates to RGB colors.
        novel_views = self.pixel_generator(v)
        # print('***** novel_views = ', novel_views.shape)

        if self.output_sidelength != self.img_sidelength:
            novel_views = novel_views.permute(0,2,1).view(
                -1, self.out_channels, self.img_sidelength, self.img_sidelength) 
            novel_views = F.interpolate(
                novel_views, size=(self.output_sidelength, self.output_sidelength), mode='bilinear')
            novel_views = novel_views.view(-1, self.out_channels, self.output_sidelength**2).permute(0,2,1)
            
        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(("image", "normals",
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

        if not self.fit_single_srn:
            self.logs.append(("scalar", "embed_min", z.min(), 1))
            self.logs.append(("scalar", "embed_max", z.max(), 1))

        # print('***** depth_map = ', depth_maps.shape)

        return novel_views, depth_maps
