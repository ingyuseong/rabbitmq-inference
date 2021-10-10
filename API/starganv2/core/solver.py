"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from os.path import join as ospj

import torch
import torch.nn as nn
import torch.nn.functional as F

from starganv2.core.model import build_model
from starganv2.core.checkpoint import CheckpointIO
from starganv2.core.data_loader import InputFetcher
import starganv2.core.utils as utils


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            # utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                # print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)
    
    @torch.no_grad()
    def sample(self, loaders, result_dir):
        args = self.args
        nets_ema = self.nets_ema
        # os.makedirs(result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim))
        
        fname = ospj(result_dir)
        print('Working on {}...'.format(fname))
        return utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)