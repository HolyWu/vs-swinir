import math
import os

import numpy as np
import torch
import torch.nn as nn
import vapoursynth as vs

from .network_swinir import SwinIR as net


def SwinIR(clip: vs.VideoNode,
           task: str = 'classical_sr',
           scale: int = 2,
           tile_x: int = 0,
           tile_y: int = 0,
           tile_pad: int = 8,
           device_type: str = 'cuda',
           device_index: int = 0) -> vs.VideoNode:
    '''
    SwinIR: Image Restoration Using Swin Transformer

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        task: Task to perform. Must be 'classical_sr', 'lightweight_sr', 'real_sr', or 'real_sr_large'.

        scale: Scale factor. The supported factors depend on the task.
            classical_sr: 2, 3, 4, 8
            lightweight_sr: 2, 3, 4
            real_sr, real_sr_large: 4

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SwinIR: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('SwinIR: Only RGBS format is supported')

    task = task.lower()
    device_type = device_type.lower()

    if task not in ['classical_sr', 'lightweight_sr', 'real_sr', 'real_sr_large']:
        raise vs.Error("SwinIR: task must be 'classical_sr', 'lightweight_sr', 'real_sr', or 'real_sr_large'")

    if task == 'classical_sr' and scale not in [2, 3, 4, 8]:
        raise vs.Error('SwinIR: scale must be 2, 3, 4 or 8 for classical_sr')
    elif task == 'lightweight_sr' and scale not in [2, 3, 4]:
        raise vs.Error('SwinIR: scale must be 2, 3 or 4 for lightweight_sr')
    elif task in ['real_sr', 'real_sr_large'] and scale != 4:
        raise vs.Error('SwinIR: scale must be 4 for real_sr and real_sr_large')

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("SwinIR: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('SwinIR: CUDA is not available')

    if os.path.getsize(os.path.join(os.path.dirname(__file__), '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth')) == 0:
        raise vs.Error("SwinIR: model files have not been downloaded. run 'python -m vsswinir' first")

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if task == 'classical_sr':
        model_name = f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth'
        model = net(upscale=scale,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    resi_connection='1conv')
        param_key_g = 'params'
    elif task == 'lightweight_sr':
        model_name = f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth'
        model = net(upscale=scale,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='pixelshuffledirect',
                    resi_connection='1conv')
        param_key_g = 'params'
    elif task == 'real_sr':
        model_name = '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        model = net(upscale=4,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='nearest+conv',
                    resi_connection='1conv')
        param_key_g = 'params_ema'
    else:
        model_name = '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
        model = net(upscale=4,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                    embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2,
                    upsampler='nearest+conv',
                    resi_connection='3conv')
        param_key_g = 'params_ema'

    model_path = os.path.join(os.path.dirname(__file__), model_name)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model.to(device)

    @torch.inference_mode()
    def swinir(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img_lq = frame_to_tensor(f[0])

        if tile_x > 0 and tile_y > 0:
            output = tile_process(img_lq, scale, tile_x, tile_y, tile_pad, device, model)
        else:
            img_lq = img_lq.to(device)
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * scale, :w_old * scale]

        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=swinir)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f


def tile_process(img: torch.Tensor, scale: int, tile_x: int, tile_y: int, tile_pad: int, device: torch.device, model: nn.Module) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_x)
    tiles_y = math.ceil(height / tile_y)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_x
            ofs_y = y * tile_y

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_x, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_y, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            input_tile = input_tile.to(device)
            _, _, h_old, w_old = input_tile.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            input_tile = torch.cat([input_tile, torch.flip(input_tile, [2])], 2)[:, :, :h_old + h_pad, :]
            input_tile = torch.cat([input_tile, torch.flip(input_tile, [3])], 3)[:, :, :, :w_old + w_pad]
            output_tile = model(input_tile)
            output_tile = output_tile[..., :h_old * scale, :w_old * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output
