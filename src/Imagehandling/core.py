"""We encapsulate the image and its associated data in a class."""

from __future__ import annotations
import cv2
from matplotlib import pyplot as plt
from typing import Sequence, Union
from copy import copy, deepcopy

class ColorState:
    RGB = {}
    GRAY = {'cmap': 'gray', 'vmin': 0, 'vmax': 255}

class Img:
    img:             cv2.Mat
    contours:        Union[Sequence[cv2.Mat], None]
    src_path:        str
    original:        cv2.Mat
    transformations: list[cv2.Mat]
    track:           bool

    CONTOURSCOLOR = (255, 0, 0)

    def __init__(self, src, track=True) -> None:
        self.img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        self.original = cv2.imread(src)
        self.src_path = src
        self.contours = None
        self.transformations = []
        self.track = track

    def write(self, path: str) -> None:
        """Store img in PATH."""
        cv2.imwrite(path, self.img)

    def _update(self, new_img) -> Img:
        """Replace image data with new-img with tracking."""
        if self.track:
            self.transformations.append(self.img)
        self.img = new_img
        return self
    
    def _scaled_shape(self, scale_w: float, scale_h: float) -> tuple:
        """Get scaled dimensions of the image."""
        w, h = self.img.shape
        return (round(w*scale_w), round(h*scale_h))
        
    def to_contours(self, use_original = True) -> Img:
        """Return a copy of the original image with contours.        
        Draw contours on the original image as background and store it as the current image. If
        use_original is False, use the current image instead.
        Note that this also change the colorspace to RGB."""
        new_self = deepcopy(self)
        img = new_self.img
        if new_self.contours:
            if use_original:
                img = new_self.original
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.drawContours(img, new_self.contours, -1, color=new_self.CONTOURSCOLOR,
                                   thickness=2)
            new_self.img = img
        return new_self

    def _prepare_plt(self, **kwargs) -> None:
        """Prepare image for 'plotting' with matplotlib."""
        plt.imshow(self.img, **kwargs)
        plt.xticks([])
        plt.yticks([])
    
    def show(self, color_state=ColorState.GRAY, **kwargs) -> Img:
        """Print image with matplotlib (for Jupyter notebooks)."""
        self._prepare_plt(**(kwargs | color_state))
        plt.show()
        return self

    def show_transformations(self, **kwargs) -> Img:
        """Show all transformations of the image."""
        imgs = self.transformations + [self.img]
        n = len(imgs)
        cols = 3
        rows = (n // cols) + 1
        for i in range(len(imgs)):
            plt.subplot(rows, cols, i+1)
            self._prepare_plt(imgs[i], **kwargs)
            plt.xticks([])
            plt.yticks([])
        plt.show()
        return self

    def binarize(self) -> Img:
        """Convert IMG to an inverted binary black-white-picture."""
        _, img = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        return self._update(img)

    def morph_dilate(self) -> Img:
        """Smear the image horizontally."""
        kernel_size = self._scaled_shape(0.10, 0.01)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        print(f"Image size {self.img.shape}, using kernel {kernel_size}")
        img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)
        return self._update(img)

    def morph_open(self) -> Img:
        """Apply erosion and dilation on the picture."""
        kernel_size = (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        print(f"Image size {self.img.shape}, using kernel {kernel_size}")
        # TODO Probieren: Erode mit kleinerem Kernel, dann dilate mit jetzigem
        img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel=kernel)
        return self._update(img)

    def find_contours(self) -> Img:
        """Find contours."""
        img = self.img.copy()
        cntrs, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = cntrs
        return self
