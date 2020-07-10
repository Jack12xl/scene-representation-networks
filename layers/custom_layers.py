import geometry
import torchvision
import util

from layers.pytorch_prototyping import Unet, UpsamplingNet, Conv2dSame

import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable


def init_recurrent_weights(self):
	for m in self.modules():
		if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					nn.init.kaiming_normal_(param.data)
				elif 'weight_hh' in name:
					nn.init.orthogonal_(param.data)
				elif 'bias' in name:
					param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
	for name, parameter in lstm_layer.named_parameters():
		if not "bias" in name: continue
		n = parameter.size(0)
		start, end = n // 4, n // 2
		parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
	total_norm = x.norm()
	total_norm = total_norm ** (1 / 2.)
	clip_coef = max_norm / (total_norm + 1e-6)
	if clip_coef < 1:
		return x * clip_coef


class DepthSampler(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,
				xy,
				depth,
				cam2world,
				intersection_net,
				intrinsics):
		self.logs = list()

		batch_size, _, _ = cam2world.shape

		intersections = geometry.world_from_xy_depth(xy=xy, depth=depth, cam2world=cam2world, intrinsics=intrinsics)

		depth = geometry.depth_from_world(intersections, cam2world)

		if self.training:
			print(depth.min(), depth.max())

		return intersections, depth


class Raymarcher(nn.Module):
	def __init__(self,
				 num_feature_channels,
				 raymarch_steps):
		super().__init__()

		self.n_feature_channels = num_feature_channels
		self.steps = raymarch_steps

		hidden_size = 16
		self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
								hidden_size=hidden_size)

		self.lstm.apply(init_recurrent_weights)
		lstm_forget_gate_init(self.lstm)

		self.out_layer = nn.Linear(hidden_size, 1)
		self.counter = 0

	def forward(self,
				cam2world,
				phi,
				uv,
				intrinsics,
				dpt=None):
		batch_size, num_samples, _ = uv.shape
		log = list()

		ray_dirs = geometry.get_ray_directions(uv,
											   cam2world=cam2world,
											   intrinsics=intrinsics)
		
		if dpt is not None:
			initial_depth = dpt * torch.ones((batch_size, num_samples, 1)).cuda()
			world_coords = geometry.world_from_xy_depth(
				uv, initial_depth, intrinsics=intrinsics, cam2world=cam2world)
			return world_coords, initial_depth, log

		initial_depth = torch.zeros((batch_size, num_samples, 1)).normal_(mean=0.05, std=5e-4).cuda()
		init_world_coords = geometry.world_from_xy_depth(uv,
														 initial_depth,
														 intrinsics=intrinsics,
														 cam2world=cam2world)

		world_coords = [init_world_coords]
		depths = [initial_depth]
		states = [None]

		for step in range(self.steps):
			v = phi(world_coords[-1])

			state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])

			if state[0].requires_grad:
				state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

			signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
			new_world_coords = world_coords[-1] + ray_dirs * signed_distance

			states.append(state)
			world_coords.append(new_world_coords)

			depth = geometry.depth_from_world(world_coords[-1], cam2world)                
			
			# log.append(('scalar', 'dmax_s%02d'%(step), depths[-1].max(), 10))
			# log.append(('scalar', 'dmin_s%02d'%(step), depths[-1].min(), 10))

			# if self.training:
			#     print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
			#           (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

			depths.append(depth)

		log.append(('scalar', 'dmin_final', depths[-1].min(),10))
		log.append(('scalar', 'dmax_final', depths[-1].max(),10))

		# if z_range is not None:
		#     dpt_min = torch.min(z_range)
		#     dpt_max = torch.max(z_range)
		#     depths[-1] = depths[-1] + dpt_min.expand_as(initial_depth).cuda()
		#     log.append(('scalar', 'z_range_min', dpt_min, 10))

		if self.counter > 0 and (not self.counter % 1000):
			# Write tensorboard summary for each step of ray-marcher.
			drawing_depths = torch.stack(depths, dim=0)[:, 0, :, :]
			drawing_depths = util.lin2img(drawing_depths).repeat(1, 3, 1, 1)
			log.append(('image', 'raycast_progress',
						torch.clamp(torchvision.utils.make_grid(drawing_depths, scale_each=False, normalize=True), 0.0,
									5),
						100))

			# # Visualize residual step distance (i.e., the size of the final step)
			# fig = util.show_images([util.lin2img(signed_distance)[i, :, :, :].detach().cpu().numpy().squeeze()
			#                         for i in range(batch_size)])
			# log.append(('figure', 'stopping_distances', fig, 100))
		self.counter += 1

		return world_coords[-1], depths[-1], log


class DeepvoxelsRenderer(nn.Module):
	def __init__(self,
				 nf0,
				 in_channels,
				 input_resolution,
				 img_sidelength,
				 out_channels=3):
		super().__init__()

		self.nf0 = nf0
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.input_resolution = input_resolution
		self.img_sidelength = img_sidelength

		self.num_down_unet = util.num_divisible_by_2(input_resolution)
		self.num_upsampling = util.num_divisible_by_2(img_sidelength) - self.num_down_unet

		self.build_net()

	def build_net(self):
		self.net = [
			Unet(   in_channels=self.in_channels,
					out_channels=self.out_channels if self.num_upsampling <= 0 else 4 * self.nf0,
					outermost_linear=True if self.num_upsampling <= 0 else False,
					use_dropout=True,
					dropout_prob=0.1,
					nf0=self.nf0 * (2 ** self.num_upsampling),
					norm=nn.BatchNorm2d,
					max_channels=8 * self.nf0,
					num_down=self.num_down_unet)
		]

		if self.num_upsampling > 0:
			self.net += [
				UpsamplingNet(per_layer_out_ch=self.num_upsampling * [self.nf0],
												  in_channels=4 * self.nf0,
												  upsampling_mode='transpose',
												  use_dropout=True,
												  dropout_prob=0.1),
				Conv2dSame(self.nf0, out_channels=self.nf0 // 2, kernel_size=3, bias=False),
				nn.BatchNorm2d(self.nf0 // 2),
				nn.ReLU(True),
				Conv2dSame(self.nf0 // 2, out_channels=self.out_channels, kernel_size=3)
			]

		self.net += [nn.Tanh()]
		self.net = nn.Sequential(*self.net)

	def forward(self, input):
		batch_size, _, ch = input.shape
		input = input.permute(0, 2, 1).view(batch_size, ch, self.img_sidelength, self.img_sidelength)
		out = self.net(input)
		return out.view(batch_size, self.out_channels, -1).permute(0, 2, 1)


class ImAEDecoder(nn.Module):
	def __init__(self, in_dim, out_dim, pos_dim=3, gf_dim=128, outermost_linear=True):
		super(ImAEDecoder, self).__init__()

		self.in_dim = in_dim
		self.pos_dim = pos_dim
		self.out_dim = out_dim
		self.gf_dim = gf_dim

		self.outermost_linear = outermost_linear

		self.linear_1 = nn.Linear(self.in_dim+self.pos_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, self.out_dim, bias=True)

		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)

	def forward(self, in_feat):

		l1 = self.linear_1(in_feat)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)


		if not self.outermost_linear:
			#l7 = torch.clamp(l7, min=0, max=1)
			l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
			
		return l7