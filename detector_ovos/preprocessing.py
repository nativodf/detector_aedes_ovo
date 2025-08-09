from PIL import Image, ImageChops
import numpy as np
import os
import urllib.request
import urllib.error
import logging
from tqdm import tqdm

def trim(im: Image.Image) -> Image.Image:
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def dividir_em_imagens(img: np.ndarray, count: int = 4):
    h = img.shape[0]
    step = h // count
    parts = []
    for i in range(count):
        start = i*step
        end = h if i == count-1 else (i+1)*step
        parts.append(img[start:end, :, :])
    return parts

def download_file(url: str, directory: str, overwrite: bool = False):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, url.split("/")[-1])
    if os.path.exists(filename) and not overwrite:
        return
    try:
        resp = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        logging.error(f"HTTP {e.code} ao baixar {url}")
        return
    total = int(resp.getheader('content-length', 0))
    with open(filename, 'wb') as f, tqdm(total=total, unit='iB', unit_scale=True) as bar:
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))