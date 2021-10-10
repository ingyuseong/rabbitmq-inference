"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import json

import torch
import torch.nn as nn
import torchvision.utils as vutils

import io
import numpy as np
from PIL import Image
import boto3
import config


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def load_image(x):
    x = denormalize(x)
    result = vutils.make_grid(x.cpu())
    return result.permute(1,2,0)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, file_path):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]

    print("Save images to S3...")
    image_list = []
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
        # 이미지 각각 저장
        id = str(i+1)
        if len(id) == 1: 
            id = '0' + id
        key = file_path+"/style{}.jpg".format(id)
        image_list.append({'id': id, 'img_src': key, 'title': '스타일 {}'.format(id)})
        save_to_S3(x_fake, key)
    return image_list

def save_to_S3(image, key):
    # S3 upload setting
    s3_client = boto3.client(
        's3',
        aws_access_key_id = config.S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key = config.S3_CONFIG['aws_secret_access_key'],
        region_name = config.S3_CONFIG['region_name']
    )

    img_array = load_image(image).squeeze().numpy().reshape(256,256,3)
    image = Image.fromarray(np.uint8(img_array*255)).convert('RGB')

    buff = io.BytesIO()
    image.save(buff, format='png')
    s3_client.put_object(
        Body=buff.getvalue(), 
        Bucket=config.S3_CONFIG['bucket'], 
        Key=key, 
        ACL='public-read', 
        ContentType='image/jpeg'
    )