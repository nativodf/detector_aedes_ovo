import logging
import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from distinctipy import distinctipy
from functools import partial
import config
from preprocessing import download_file


def logs_config():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
    
def global_init():
    config.device = get_torch_device()
    set_seed(1234567)

    logging.info(f"Device inicializado em: {config.device}")

    config.base_dir = os.path.dirname(os.path.abspath(__file__)) 
    model_path = os.path.join(config.base_dir, '..', 'mrcnn_model.pth')

    logging.info(f"Tentando carregar modelo em: {model_path}")


    try:
        config.model = torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )
        logging.info("Modelo carregado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar modelo: {e}")
        config.model = None

    colors = distinctipy.get_colors(len(config.class_names))
    config.int_colors = [tuple(int(c*255) for c in col) for col in colors]

    font_file = "KFOlCnqEu92Fr1MmEU9vAw.ttf"
    download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
    config.draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=12)

    Image.MAX_IMAGE_PIXELS = None

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

def move_data_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_data_to_device(x, device) for x in data)
    return data

