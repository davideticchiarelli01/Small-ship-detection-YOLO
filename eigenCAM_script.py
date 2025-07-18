import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import os
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

# Lista dei path relativi
image_paths = [
    'val/boat72_so.png',
]
input_dir = '/app/datasets/D3_NoSentinel1/images/'
output_dir = '/app/yolo_cam_result/'
os.makedirs(output_dir, exist_ok=True)
# Carica modello
model = YOLO('/app/results/YOLOv12mLight-SPD-SPANet_D3_NoSentinel1_dataset11/weights/best.pt')
#model = YOLO('/app/best.pt')
model = model.cpu()
target_layers = [model.model.model[-2]]
#target_layers = [model.model.model[-2]]

for rel_path in image_paths:
    input_path = os.path.join(input_dir, rel_path)
    img = cv2.imread(input_path)
    if img is None:
        continue
    img = cv2.resize(img, (640, 640))
    rgb_img = img.copy()
    img_float = np.float32(img) / 255.

    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f'{filename}_cam_result.png')
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, cam_image_bgr)
