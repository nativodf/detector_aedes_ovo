# main.py

# =========================
#       IMPORTS
# =========================
import os
import random
import time
import logging
import numpy as np
import torch
from functools import partial
from glob import glob
import urllib.request
import urllib.error
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageChops
from distinctipy import distinctipy
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2 as transforms
import rembg

# =========================
#   CONFIGURAÇÃO DE LOGS
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

# =========================
#   FUNÇÕES AUXILIARES
# =========================
def set_seed(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)

def resize_img(img: Image.Image, target_sz: int = 512, divisor: int = 32) -> Image.Image:
    min_dim = np.argmin(img.size)
    max_dim = np.argmax(img.size)
    ratio = min(img.size) / target_sz
    new_sz = []
    new_sz.insert(min_dim, target_sz)
    new_sz.insert(max_dim, int(max(img.size) / ratio))
    img = img.resize(new_sz)
    if divisor > 0:
        w, h = img.size
        w -= w % divisor
        h -= h % divisor
        img = img.crop((0, 0, w, h))
    return img

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)

def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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

def move_data_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_data_to_device(x, device) for x in data)
    return data

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

# =========================
#   INITIALIZAÇÃO GLOBAL
# =========================
set_seed(1234567)
device = get_torch_device()
logging.info(f"Device inicializado em: {device}")

class_names = ["background", "ovo"]
model = torch.load(
    "mrcnn_model.pth",
    map_location=torch.device('cpu'),
    weights_only=False
)

colors = distinctipy.get_colors(len(class_names))
int_colors = [tuple(int(c*255) for c in col) for col in colors]

font_file = "KFOlCnqEu92Fr1MmEU9vAw.ttf"
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=12)

Image.MAX_IMAGE_PIXELS = None

# =========================
#   FUNÇÃO DE INFERÊNCIA
# =========================
def predict_nn(img: np.ndarray, train_sz: int = 512, threshold: float = 0.85):
    t0 = time.time()
    pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    t1 = time.time()
    logging.info(f"[predict_nn] start resize: {(t1-t0):.2f}s")

    input_img = resize_img(pil, target_sz=train_sz, divisor=1)
    t2 = time.time()
    logging.info(f"[predict_nn] resize done: {(t2-t1):.2f}s")

    model.eval()
    model.to(device)
    tensor_in = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])(input_img)[None].to(device)
    t3 = time.time()
    logging.info(f"[predict_nn] preprocess done: {(t3-t2):.2f}s")

    with torch.no_grad():
        out = model(tensor_in)
    t4 = time.time()
    logging.info(f"[predict_nn] inference done: {(t4-t3):.2f}s")

    out = move_data_to_device(out, torch.device('cpu'))
    mask = out[0]['scores'] > threshold

    bboxes = BoundingBoxes(
        out[0]['boxes'][mask] * (min(pil.size) / min(input_img.size)),
        format='xyxy',
        canvas_size=input_img.size[::-1]
    )
    t5 = time.time()
    logging.info(f"[predict_nn] postprocess start: {(t5-t4):.2f}s")

    labels = [class_names[int(l)] for l in out[0]['labels'][mask]]
    scores = out[0]['scores'][mask]
    masks = F.interpolate(out[0]['masks'][mask], size=pil.size[::-1])

    try:
        masks = torch.concat([Mask((m>=threshold).to(torch.bool)) for m in masks])
    except:
        return pil, 0

    cols = [int_colors[class_names.index(l)] for l in labels]
    img_t = transforms.PILToTensor()(pil)
    seg = draw_segmentation_masks(img_t, masks=masks, alpha=0.3, colors=cols)
    seg = draw_bounding_boxes(
        seg, boxes=bboxes,
        labels=[f"{l}\n{s*100:.1f}%" for l, s in zip(labels, scores)],
        colors=cols
    )
    t6 = time.time()
    logging.info(f"[predict_nn] postprocess done: {(t6-t5):.2f}s total: {(t6-t0):.2f}s")

    return tensor_to_pil(seg), mask.sum().item()

# =========================
#   FUNÇÃO PARA GRADIO
# =========================
def predict_image(pil_img: Image.Image, threshold: float = 0.85):
    t0 = time.time()
    arr = np.array(pil_img.convert("RGBA"))
    arr = rembg.remove(arr)
    t1 = time.time()
    logging.info(f"[predict_image] rembg done: {(t1-t0):.2f}s")

    fg = Image.fromarray(arr)
    bg = Image.new("RGB", fg.size, (255,255,255))
    bg.paste(fg, mask=fg.split()[3])
    img_trim = trim(bg)
    t2 = time.time()
    logging.info(f"[predict_image] trim done: {(t2-t1):.2f}s")

    parts = dividir_em_imagens(np.array(img_trim), count=4)
    logging.info(f"[predict_image] divided into {len(parts)} parts in {(time.time()-t2):.2f}s")

    outputs, total = [], 0
    for i, p in enumerate(parts):
        ti = time.time()
        im_out, cnt = predict_nn(p, threshold=threshold)
        outputs.append((im_out, cnt))
        total += cnt
        logging.info(f"[predict_image] part {i+1} processed in {(time.time()-ti):.2f}s")

    tf = time.time()
    logging.info(f"[predict_image] TOTAL: {(tf-t0):.2f}s")
    return outputs, total

# =========================
#   TESTE LOCAL
# =========================
if __name__ == "__main__":
    img = Image.open("papel20250404_17320654.png")
    outs, tot = predict_image(img)
    print("Total de ovos:", tot)
