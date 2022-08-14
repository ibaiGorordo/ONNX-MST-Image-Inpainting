import cv2
import numpy as np
from imread_from_url import imread_from_url

from MST import MST

# Simulate image mask
def get_masked_img(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[1300:1500, 1300:1500] = 255
    masked_img = img.copy()
    masked_img[mask == 255] = 255

    return masked_img, mask

mst_path = "models/MST_P2M_simp.onnx"
hawp_path = "models/hawp_simp.onnx"

# Initialize model
imageInpainting = MST(mst_path, hawp_path, hawp_threshold=0.9)

masked_img = imread_from_url("https://github.com/DQiaole/ZITS_inpainting/raw/main/test_imgs/img2.png")
mask = imread_from_url("https://github.com/DQiaole/ZITS_inpainting/raw/main/test_imgs/mask2.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
output_img, output_edge_map, output_lines_map = imageInpainting(masked_img, mask)

# Draw Inpaint
output_img = imageInpainting.draw()
cv2.namedWindow("Inpaint", cv2.WINDOW_NORMAL)
cv2.imshow("Inpaint", output_img)
cv2.waitKey(0)
