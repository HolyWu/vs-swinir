import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(__file__), filename), 'wb') as f:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename, total=int(r.headers.get('content-length', 0))) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == '__main__':
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    download_model('https://github.com/HolyWu/vs-swinir/releases/download/model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
