import os
import argparse

from torchvision import models
import torchvision.transforms as transforms

from munch import Munch
from torch.backends import cudnn
import torch

from starganv2.core.data_loader import get_test_loader
from starganv2.core.solver import Solver
from starganv2.core.wing import align_face

def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def starGAN_inference(try_id, gender, length, image_bytes):
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')

    parser.add_argument('--resume_iter', type=int, default=30000, help='Iterations to resume training/testing')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator')

    # directory for training
    parser.add_argument('--checkpoint_dir', type=str, default='pretrained/starganv2/2domain', help='Directory for saving network checkpoints')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='static/result_image', help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='image/user', help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='image/2domain/ref', help='Directory containing input reference images')
    
    # face alignment
    parser.add_argument('--wing_path', type=str, default='pretrained/starganv2/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='pretrained/starganv2/celeba_lm_mean.npz')

    args = parser.parse_args()
    
    # align image
    out_dir = "image/try/{}/input".format(try_id) # to S3
    align_face(args, out_dir, length, image_bytes)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    solver = Solver(args)

    source_dir = "image/2domain/input"
    reference_dir = "image/2domain/ref"
    result_dir = "image/try/{}/result".format(try_id) # to S3

    assert len(subdirs(source_dir)) == args.num_domains
    assert len(subdirs(reference_dir)) == args.num_domains

    loaders = Munch(src=get_test_loader(root=source_dir,
                                    img_size=args.img_size,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1),
                ref=get_test_loader(root=reference_dir,
                                    img_size=args.img_size,
                                    batch_size=8, # ref count handling
                                    shuffle=False,
                                    num_workers=1))
    img_list = solver.sample(loaders, result_dir)
    
    os.remove("image/2domain/input/female/input.png")
    os.remove("image/2domain/input/male/input.png")

    return img_list
    