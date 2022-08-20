import cv2
import numpy as np
from imread_from_url import imread_from_url

from MST import MST, MaskSketcher

mst_path = "models/MST_P2M_simp.onnx"
hawp_path = "models/hawp_simp.onnx"

# Initialize model
imageInpainting = MST(mst_path, hawp_path, hawp_threshold=0.9)

img_url = "https://upload.wikimedia.org/wikipedia/commons/6/6d/Perro_Pipper_en_Merida_-_Pipperontour.jpg"
img = imread_from_url(img_url)

maskSketcher = MaskSketcher(img)
mask = maskSketcher.mask

output_img, output_edge_map, output_lines_map = imageInpainting(img, mask)

# Draw Inpaint
output_img = imageInpainting.draw()
combined_img = np.hstack((img, output_img))
# cv2.namedWindow("Mask image", cv2.WINDOW_NORMAL)
cv2.imshow("Mask image", combined_img)
cv2.imwrite("../doc/img/inpainted.png", output_img)
cv2.waitKey(0)
