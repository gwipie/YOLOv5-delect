import os
import cv2
import numpy as np
import glob

model_input_height = 640
model_input_width = 640

src_dir = '/data/datasets/prepare_data'
dst_dir = '/data/yolov5/calibration_data_rgb_f32'

os.makedirs(dst_dir, exist_ok=True)

image_files = glob.glob(os.path.join(src_dir, '*.jpg'))

for src_file in image_files:
    image = cv2.imread(src_file)
    image = cv2.resize(image, (model_input_width, model_input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32) / 255.0
    
    filename = os.path.basename(src_file)
    short_name = os.path.splitext(filename)[0]
    pic_name = os.path.join(dst_dir, short_name + '.rgb')
    image.tofile(pic_name)
    print(f"write:{pic_name}")

print(f"Done! Total: {len(image_files)} images")
