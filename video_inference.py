import cv2
import pafy

from ModelName import ModelName

# # Initialize video
# cap = cv2.VideoCapture("input.avi")

videoUrl = 'https://youtu.be/xxxxxxxxx'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))

# Initialize inference model
model_path = "models/xxxxxx.onnx"
modelName = ModelName(model_path)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Perform the inference in the current frame
    outputs = modelName(frame)

    output_img = modelName.draw(frame)
    cv2.imshow("Model Output", output_img)
    # out.write(output_img)

# out.release()
