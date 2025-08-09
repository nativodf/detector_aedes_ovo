import time
from PIL import Image
import logging
import numpy as np
import rembg
from config import class_names
from preprocessing import trim, dividir_em_imagens
from inference import predict_nn


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