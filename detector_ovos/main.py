import base64
from io import BytesIO
import os
import json
from PIL import Image
import config
from utils import logs_config, global_init
from gradio import predict_image


def main():
    logs_config()
    global_init()

    image_path = os.path.join(config.base_dir, "..", "ovitrampa.jpg")
    img = Image.open(image_path)

    outs, tot = predict_image(img)
    #print("Total de ovos:", tot)

    # convert images to base 64
    base64_imgs = []
    for im, cnt in outs:
        buffer = BytesIO()
        im.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_imgs.append({
            "image": img_str,
            "count": cnt
        })

    # turns the python dictionary into a JSON string
    print(json.dumps({
        "total": tot,
        "results": base64_imgs
    }))


if __name__ == "__main__":
    main()