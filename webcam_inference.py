import cv2

from ModelName import ModelName

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize Inference model
model_path = "models/xxxxxx.onnx"
modelName = ModelName(model_path)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform the inference in the current frame
    outputs = modelName(frame)

    output_img = modelName.draw(frame)
    cv2.imshow("Model Output", output_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
