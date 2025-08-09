import torch
from functools import partial
from distinctipy import distinctipy
from torchvision.utils import draw_bounding_boxes

# global variables
device = None
model = None
class_names = ["background", "ovo"]
int_colors = None
base_dir = None
draw_bboxes = None