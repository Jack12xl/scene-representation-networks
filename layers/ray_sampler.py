
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch

from torchsearchsorted import searchsorted

from layers import geometry

def intersection(rays, bbox):
    n = rays.shape[0]
    left_face = bbox[:, 0, 0]
    right_face = bbox[:, 6, 0]
    front_face = bbox[:, 0, 1]
    back_face = bbox[:, 6, 1]
    bottom_face = bbox[:, 0, 2]
    up_face = bbox[:, 6, 2]
    # parallel t 无穷大
    left_t = ((left_face - rays[:, 3]) / (rays[:, 0] +
                                          np.finfo(float).eps.item())).reshape((n, 1))
    right_t = ((right_face - rays[:, 3]) / (rays[:, 0] +
                                            np.finfo(float).eps.item())).reshape((n, 1))
    front_t = ((front_face - rays[:, 4]) / (rays[:, 1] +
                                            np.finfo(float).eps.item())).reshape((n, 1))
    back_t = ((back_face - rays[:, 4]) / (rays[:, 1] +
                                          np.finfo(float).eps.item())).reshape((n, 1))
    bottom_t = ((bottom_face - rays[:, 5]) / (rays[:, 2] +
                                              np.finfo(float).eps.item())).reshape((n, 1))
    up_t = ((up_face - rays[:, 5]) / (rays[:, 2] +
                                      np.finfo(float).eps)).reshape((n, 1))

    left_point = left_t * rays[:, :3] + rays[:, 3:]
    right_point = right_t * rays[:, :3] + rays[:, 3:]
    front_point = front_t * rays[:, :3] + rays[:, 3:]
    back_point = back_t * rays[:, :3] + rays[:, 3:]
    bottom_point = bottom_t * rays[:, :3] + rays[:, 3:]
    up_point = up_t * rays[:, :3] + rays[:, 3:]

    left_mask = (left_point[:, 1] >= bbox[:, 0, 1]) & (left_point[:, 1] <= bbox[:, 7, 1]) \
        & (left_point[:, 2] >= bbox[:, 0, 2]) & (left_point[:, 2] <= bbox[:, 7, 2])
    right_mask = (right_point[:, 1] >= bbox[:, 1, 1]) & (right_point[:, 1] <= bbox[:, 6, 1]) \
        & (right_point[:, 2] >= bbox[:, 1, 2]) & (right_point[:, 2] <= bbox[:, 6, 2])

    # compare x, z
    front_mask = (front_point[:, 0] >= bbox[:, 0, 0]) & (front_point[:, 0] <= bbox[:, 5, 0]) \
        & (front_point[:, 2] >= bbox[:, 0, 2]) & (front_point[:, 2] <= bbox[:, 5, 2])

    back_mask = (back_point[:, 0] >= bbox[:, 3, 0]) & (back_point[:, 0] <= bbox[:, 6, 0]) \
        & (back_point[:, 2] >= bbox[:, 3, 2]) & (back_point[:, 2] <= bbox[:, 6, 2])

    # compare x,y
    bottom_mask = (bottom_point[:, 0] >= bbox[:, 0, 0]) & (bottom_point[:, 0] <= bbox[:, 2, 0]) \
        & (bottom_point[:, 1] >= bbox[:, 0, 1]) & (bottom_point[:, 1] <= bbox[:, 2, 1])

    up_mask = (up_point[:, 0] >= bbox[:, 4, 0]) & (up_point[:, 0] <= bbox[:, 6, 0]) \
        & (up_point[:, 1] >= bbox[:, 4, 1]) & (up_point[:, 1] <= bbox[:, 6, 1])

    tlist = -torch.ones_like(rays, device=rays.device)*1e3
    tlist[left_mask, 0] = left_t[left_mask].reshape((-1,))
    tlist[right_mask, 1] = right_t[right_mask].reshape((-1,))
    tlist[front_mask, 2] = front_t[front_mask].reshape((-1,))
    tlist[back_mask, 3] = back_t[back_mask].reshape((-1,))
    tlist[bottom_mask, 4] = bottom_t[bottom_mask].reshape((-1,))
    tlist[up_mask, 5] = up_t[up_mask].reshape((-1,))
    tlist = tlist.topk(k=2, dim=-1)

    return tlist[0]


def gen_weight(sigma, delta, act_fn=F.relu):
    """Generate transmittance from predicted density
    """
    alpha = 1.-torch.exp(-act_fn(sigma.squeeze(-1))*delta)
    weight = 1.-alpha+1e-10
    #weight = alpha * torch.cumprod(weight, dim=-1) / weight # exclusive cum_prod

    weight = alpha * torch.cumprod(torch.cat(
        [torch.ones((alpha.shape[0], 1), device=alpha.device), weight], -1), -1)[:, :-1]

    return weight


def sample_pdf(rays, z_vals, weights, N_samples, det=False, pytest=False):
    # Get pdf
    bins = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=z_vals.device), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=z_vals.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) +
                       [N_samples], device=z_vals.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1, device=inds.device), inds-1)
    above = torch.min(
        cdf.shape[-1]-1 * torch.ones_like(inds, device=inds.device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(
        denom < 1e-5, torch.ones_like(denom, device=denom.device), denom)
    t = (u-cdf_g[..., 0])/denom
    samples_z = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    pts = samples_z.unsqueeze(-1) * \
        rays[:, :3].unsqueeze(1) + rays[:, 3:].unsqueeze(1)

    return samples_z, pts


def sample_near_far(rays, near_far, num_samples=16):
    
    n = rays.size(0)
    ray_d = rays[:, :3]
    ray_o = rays[:, 3:]

    near = 1
    far = 5.5

    t_vals = torch.linspace(
        0., 1., steps=num_samples, device=rays.device)
    #print(near_far[:,0:1].repeat(1, self.sample_num).size(), t_vals.unsqueeze(0).repeat(n,1).size())
    z_vals = near_far[:, 0:1].repeat(1, num_samples) * (1.-t_vals).unsqueeze(0).repeat(
        n, 1) + near_far[:, 1:2].repeat(1, num_samples) * (t_vals.unsqueeze(0).repeat(n, 1))

    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)

    t_rand = torch.rand(z_vals.size(), device=rays.device)

    z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]

    return z_vals.unsqueeze(-1), pts
    

def render(depth, rgb, sigma, noise, border_weight=1e10):
    delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
    #pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
    pad = border_weight * \
        torch.ones(delta[..., :1].size(), device=delta.device)
    delta = torch.cat([delta, pad], dim=-1)   # [N, L]

    if noise > 0.:
        sigma += (torch.randn(size=sigma.size(),
                                device=delta.device) * noise)

    weights = gen_weight(sigma, delta).unsqueeze(-1)  # [N, L, 1]

    color = torch.sum(torch.sigmoid(rgb) * weights, dim=1)  # [N, 3]
    depth = torch.sum(weights * depth, dim=1)   # [N, 1]
    acc_map = torch.sum(weights, dim=1)

    return color, depth, acc_map, weights


class MLP(nn.Module):
    def __init__(   self,
                    in_ch,
                    hidden_ch=[256],
                    skip_dim=3,
                    activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        self.skip_dim = skip_dim
        
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.activation_fn = activation_fn

        self.net = []

        for h_dim in self.hidden_ch:
            self.net.append(nn.Sequential(
                nn.Linear(in_ch, h_dim),
                self.activation_fn(inplace=True)
            ))
            in_ch = h_dim + in_ch

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat([out, x], dim=-1)
        return out


class SIFRenderer(nn.Module):
    def __init__(self,
                 num_samples,
                 num_fine_samples=0,
                 num_classes=19,
                 backbone_dim=256):

        super().__init__()
        self.num_samples = num_samples
        self.num_fine_samples = num_fine_samples

        self.near_far = (0.0, 5.0)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.hidden_dim = [2048, 1024, 512, 256, 128]
        self.feat_dim = self.backbone_dim + 3       # feature + pose
        
        act_fn = nn.Softmax(dim=-1) if num_classes > 2 else nn.Sigmoid(dim=-1)
        self.net = nn.Sequential(
            MLP(self.feat_dim, self.hidden_dim, activation_fn=nn.LeakyReLU),
            nn.Linear(self.hidden_dim[-1], num_classes),
            act_fn)

    def get_loss(self, output, gt):
        return self.loss_fn(output, gt)

    def forward(self,
                cam2world,
                phi,
                uv,
                intrinsics,
                encoder,        # imoplicit field encoder : R^3 -> R^k
                near_far=None):

        # get rays
        ray_dirs = geometry.get_ray_directions(
            uv, cam2world=cam2world, intrinsics=intrinsics)
        pos = cam2world[:, :3, 3].unsqueeze(1).expand(-1,ray_dirs.shape[1],-1)
        rays = torch.cat([ray_dirs, pos], dim=3)

        # sample coarse
        if near_far is None:    near_far = self.near_far
        dpt, pts = sample_near_far(rays, near_far, self.num_samples)

        # calc confidence for each ray (PointRend)

        # encoding
        feat = encoder(pts)
        prob = self.net(feat)   # (N, k) for k classes 

        if self.num_fine_samples > 0:
            weight = prob[:, :1]        # (B*H*W, 1) cls 0 as bg.
            dpt_fine, pts = sample_pdf(rays, dpt, weight)
            feat = encoder(pts)
            prob = self.net(feat)

        # reshape to image
        prob = prob.view(B, H, W, self.num_classes)

        return prob
