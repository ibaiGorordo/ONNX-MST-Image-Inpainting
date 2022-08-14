import time
import cv2
import numpy as np
import onnxruntime

from .hawp.HAWP import HAWP


class MST:

    def __init__(self, model_path, hawp_path, hawp_threshold=0.98):
        # Initialize model
        self.initialize_model(model_path)
        self.line_detector = HAWP(hawp_path, hawp_threshold)

    def __call__(self, image, mask):
        return self.update(image, mask)

    def initialize_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def update(self, image, mask):
        self.img_height, self.img_width = image.shape[:2]
        self.image = image.copy()
        self.mask = mask.copy()

        img_tensor = self.prepare_image()
        self.mask_tensor = self.prepare_mask()

        self.input_lines = self.get_lines_map()
        line_tensor = self.prepare_tensor(self.input_lines)

        self.input_edges = self.get_edge_map()
        edge_tensor = self.prepare_tensor(self.input_edges)

        # Perform inference on the image
        outputs = self.inference(img_tensor, self.mask_tensor, line_tensor, edge_tensor)

        # Process output data
        self.output_img, \
        self.output_edge_map, \
        self.output_lines_map = self.process_output(outputs)

        return self.output_img, self.output_edge_map, self.output_lines_map

    def prepare_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        input_norm = cv2.resize(image, (self.input_width, self.input_height),
                                interpolation=cv2.INTER_AREA) / 255.0
        input_norm = input_norm.transpose(2, 0, 1)
        return input_norm[np.newaxis, :, :, :].astype(np.float32)

    def prepare_mask(self):
        mask_norm = self.mask / 255
        mask_norm = cv2.resize(mask_norm, (self.input_width, self.input_height),
                               interpolation=cv2.INTER_NEAREST)
        return mask_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def prepare_tensor(self, input):
        input_norm = cv2.resize(input, (self.input_width, self.input_height),
                                interpolation=cv2.INTER_AREA) / 255.0
        return input_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def inference(self, img_tensor, mask_tensor, line_tensor, edge_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: img_tensor,
                                    self.input_names[1]: mask_tensor,
                                    self.input_names[2]: line_tensor,
                                    self.input_names[3]: edge_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, outputs):
        output_img, edges_pred, lines_preds = [output.squeeze() for output in outputs]

        output_img = self.process_output_img(output_img)
        output_edge_map = (edges_pred.squeeze() * 255).astype(np.uint8)
        output_lines_map = (lines_preds.squeeze() * 255).astype(np.uint8)

        return output_img, output_edge_map, output_lines_map

    def get_lines_map(self):
        hawp_input = self.image.copy()
        hawp_input[self.mask==255] = 127.5
        line_input_img = cv2.resize(hawp_input, (self.input_width, self.input_height))
        lines, scores = self.line_detector(line_input_img)
        line_img = np.zeros(line_input_img.shape[:2])

        for line in lines:
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]),
                     (255, 255, 255), 1, cv2.LINE_AA)

        line_img = line_img.astype(np.uint8)
        return line_img

    def get_edge_map(self):
        edge_input_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edge_input_img = cv2.resize(edge_input_img, (self.input_width, self.input_height))
        smoothed_input = cv2.GaussianBlur(edge_input_img, (7, 7), 2)
        edge_img = cv2.Canny(smoothed_input, 35, 70)
        return edge_img

    def process_output_img(self, output_img):
        output_img = output_img.transpose(1, 2, 0)
        output_img = output_img * 255
        output_img = output_img.astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        return output_img

    def draw(self):
        output_img = cv2.resize(self.output_img, (self.img_width, self.img_height))
        output_img[self.mask == 0] = self.image[self.mask == 0]
        return output_img


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url


    def get_masked_img(img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[1300:1500, 1300:1500] = 255
        masked_img = img.copy()
        masked_img[mask == 255] = 255

        return masked_img, mask


    mst_path = "../models/MST_P2M_simp.onnx"
    hawp_path = "../models/hawp_simp.onnx"

    # Initialize model
    imageInpainting = MST(mst_path, hawp_path, hawp_threshold=0.9)

    img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/0/0d/Bedroom_Mitcham.jpg")

    masked_img, mask = get_masked_img(img)
    output_img, output_edge_map, output_lines_map = imageInpainting(masked_img, mask)

    # Draw Inpaint
    output_img = imageInpainting.draw()
    cv2.namedWindow("Inpaint", cv2.WINDOW_NORMAL)
    cv2.imshow("Inpaint", output_img)
    cv2.waitKey(0)
