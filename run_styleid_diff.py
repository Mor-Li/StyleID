import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

import matplotlib.pyplot as plt  # 新增导入 matplotlib 库
from tqdm import tqdm  # 新增导入 tqdm 库

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default='./data/cnt', help='Directory path to content images')
    parser.add_argument('--sty', default='./data/sty', help='Directory path to style images')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM inversion steps')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='Steps to save features')
    parser.add_argument('--start_step', type=int, default=49, help='Start step for feature injection')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='Image height in pixels')
    parser.add_argument('--W', type=int, default=512, help='Image width in pixels')
    parser.add_argument('--C', type=int, default=4, help='Latent channels')
    parser.add_argument('--f', type=int, default=8, help='Downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='Attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='Query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='Injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='Model config file')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='Save path for precomputed features')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='Model checkpoint file')
    parser.add_argument('--precision', type=str, default='autocast', help='Precision mode: "full" or "autocast"')
    parser.add_argument('--output_path', type=str, default='output', help='Directory to save output images')
    parser.add_argument("--without_init_adain", action='store_true', help='Do not apply initial AdaIN')
    parser.add_argument("--without_attn_injection", action='store_true', help='Do not apply attention injection')
    opt = parser.parse_args()

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    begin = time.time()

    # 使用 tqdm 为风格图像和内容图像添加进度条
    for sty_name in tqdm(sty_img_list, desc='Processing styles'):
        sty_name_ = os.path.join(opt.sty, sty_name)
        # 加载风格图像的原始PIL图像
        try:
            sty_pil = Image.open(sty_name_).convert("RGB")
        except Exception as e:
            print(f"Error loading style image {sty_name_}: {e}")
            continue  # 跳过无法加载的图像

        init_sty = load_img(sty_name_).to(device)
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(
                init_sty.clone(),
                num_steps=ddim_inversion_steps,
                unconditional_conditioning=uc,
                end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                callback_ddim_timesteps=save_feature_timesteps,
                img_callback=ddim_sampler_callback
            )
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']

        # 使用 tqdm 为内容图像添加进度条
        for cnt_name in tqdm(cnt_img_list, desc='  Processing contents', leave=False):
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            # 生成输出文件名
            base_cnt = os.path.basename(cnt_name).split('.')[0]
            base_sty = os.path.basename(sty_name).split('.')[0]
            output_image_name = f"{base_cnt}_stylized_{base_sty}_output.png"
            combined_image_name = f"{base_cnt}_stylized_{base_sty}_compare.png"
            output_image_path = os.path.join(output_path, output_image_name)
            combined_image_path = os.path.join(output_path, combined_image_name)

            # 检查组合图像文件是否已经存在
            if os.path.exists(combined_image_path):
                print(f"Combined output already exists for {cnt_name} and {sty_name}, skipping...")
                continue  # 跳过当前循环，处理下一个内容-风格对

            # 加载内容图像的原始PIL图像
            try:
                cnt_pil = Image.open(cnt_name_).convert("RGB")
            except Exception as e:
                print(f"Error loading content image {cnt_name_}: {e}")
                continue  # 跳过无法加载的图像

            init_cnt = load_img(cnt_name_).to(device)
            cnt_feat_name = os.path.join(feat_path_root, base_cnt + '_cnt.pkl')
            cnt_feat = None

            # ddim inversion encoding
            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(
                    init_cnt.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=ddim_sampler_callback
                )
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # inversion
                        print(f"Inversion end: {time.time() - begin:.2f} seconds")
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                            adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                        feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                        if opt.without_attn_injection:
                            feat_maps = None

                        # inference
                        samples_ddim, intermediates = sampler.sample(
                            S=ddim_steps,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=adain_z_enc,
                            injected_features=feat_maps,
                            start_step=start_step,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        # 单独保存效果图
                        img.save(output_image_path)

                        # 将内容图像、风格图像和输出图像转换为numpy数组
                        content_np = np.array(cnt_pil).astype(np.float32) / 255.0
                        style_np = np.array(sty_pil).astype(np.float32) / 255.0
                        output_np = np.array(img).astype(np.float32) / 255.0

                        # 创建一个1行3列的子图
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                        # 显示内容图像
                        axes[0].imshow(content_np)
                        axes[0].set_title('Content Image')
                        axes[0].axis('off')

                        # 显示风格图像
                        axes[1].imshow(style_np)
                        axes[1].set_title('Style Image')
                        axes[1].axis('off')

                        # 显示输出图像
                        axes[2].imshow(output_np)
                        axes[2].set_title('Output Image')
                        axes[2].axis('off')

                        # 调整布局并保存组合图像
                        plt.tight_layout()
                        plt.savefig(combined_image_path)
                        plt.close()

                        if len(feat_path_root) > 0:
                            print("Save features")
                            if not os.path.isfile(cnt_feat_name):
                                with open(cnt_feat_name, 'wb') as h:
                                    pickle.dump(cnt_feat, h)
                            if not os.path.isfile(sty_feat_name):
                                with open(sty_feat_name, 'wb') as h:
                                    pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin:.2f} seconds")

if __name__ == "__main__":
    main()
