import cv2
from imread_from_url import imread_from_url

from ModelName import ModelName

# Initialize inference model
model_path = "models/xxxxx.onnx"
modelName = ModelName(model_path)

# Read image
img_url = ""
img = imread_from_url(img_url)

# Perform the inference in the image
outputs = modelName(img)

# Draw Model Output
output_img = modelName.draw(img)
cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", output_img)
cv2.imwrite("doc/img/output.jpg", output_img)
cv2.waitKey(0)
