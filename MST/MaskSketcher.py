from enum import Enum
import cv2
import numpy as np


class MouseState(Enum):
    Hover = 0
    Drawing = 1
    Removing = 2


class MaskSketcher:
    def __init__(self, img, color=(255, 255, 255)):

        self._mask: np.array = np.zeros(img.shape[:2], np.uint8)
        self._masked_img: np.array = img.copy()
        self.original_img: np.array = img

        self.color: tuple = color
        self.tool_size = self._mask.shape[0] // 20

        self.prev_pt: tuple = None
        self.state: MouseState = MouseState.Hover

        self.title_window = 'Mask image'
        self.create_window()
        self.draw()
        cv2.waitKey(0)

    @property
    def masked_img(self):
        return self._masked_img

    @property
    def mask(self):
        return self._mask

    def create_window(self):

        cv2.namedWindow(self.title_window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Drawing size: ', self.title_window,
                           self.tool_size, self._mask.shape[1],
                           self.on_trackbar)
        cv2.setMouseCallback(self.title_window, self.on_mouse)

    def paint(self, pt: tuple):
        self.update_mask(pt, 255)

    def remove(self, pt: tuple):
        self.update_mask(pt, 0)

    def update_mask(self, pt: tuple, value: int = 0):
        cv2.line(self._mask, self.prev_pt, pt, value, self.tool_size)
        self._masked_img[self._mask == 255] = self.color
        self._masked_img[self._mask == 0] = self.original_img[self._mask == 0]
        self.prev_pt = pt
        self.draw()

    def draw(self):
        cv2.imshow(self.title_window, self._masked_img)

    def on_trackbar(self, size):
        self.tool_size = size

    def on_mouse(self, event, x, y, flags, param):

        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state = MouseState.Drawing
        elif event == cv2.EVENT_LBUTTONUP:
            self.state = MouseState.Hover
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.state = MouseState.Removing
        elif event == cv2.EVENT_MBUTTONUP:
            self.state = MouseState.Hover

        if self.state == MouseState.Hover:
            self.prev_pt = None
            return

        if not self.prev_pt:
            self.prev_pt = pt
            return

        if self.state == MouseState.Drawing:
            self.paint(pt)
        elif self.state == MouseState.Removing:
            self.remove(pt)


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/6/6d/Perro_Pipper_en_Merida_-_Pipperontour.jpg")
    maskSketcher = MaskSketcher(img)

    inpainted = cv2.inpaint(img, maskSketcher.mask, 3, cv2.INPAINT_TELEA)
    combined_img = np.hstack((img, inpainted))
    cv2.namedWindow("Inpaint", cv2.WINDOW_NORMAL)
    cv2.imshow("Inpaint", combined_img)
    cv2.waitKey(0)

    cv2.imwrite("../masks/mask.png", maskSketcher.mask)
