import numpy as np


def predict_nn(img: np.ndarray, train_sz: int = 512, threshold: float = 0.85):
    import time
    import logging
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2 as transforms
    from torchvision.tv_tensors import BoundingBoxes, Mask
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
    from PIL import Image
    import numpy as np
    from config import model, device, class_names, int_colors, draw_bboxes
    from utils import resize_img, move_data_to_device, tensor_to_pil
    
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