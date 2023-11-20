"""Segment an image of handwritten text into lines and words."""

from __future__ import annotations
import cv2
from pathlib import Path
import sys
from PIL import Image, ImageShow
#from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola
from skimage.morphology import isotropic_dilation
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter, rectangle, set_color
import numpy as np
from typing import TypeAlias, Union, Tuple, Callable, Any, Optional
import logging

ImageArray: TypeAlias = np.ndarray
PathOrStr: TypeAlias = Union[Path, str]
BBox: TypeAlias = Tuple[int, int, int, int]
Color: TypeAlias = Tuple[int, int, int]

Y_MIN = 0  # Offsets for the bounding box tuple
X_MIN = 1
Y_MAX = 2
X_MAX = 3

# TODO Why is there no documentation on the imported functions from scikit?

# -----------------------------------------------------------
# Logging


def activate_logger() -> logging.Logger:
    """Activate logging."""
    name = __name__ if __name__ != '__main__' else sys.argv[0]
    log = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s - %(name)s: %(levelname)s- %(funcName)s - %(message)s")
    fh = logging.FileHandler(f"{name}.log", mode='w')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    #log.addHandler(fh)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)  # all log messages to the handler
    fh.setLevel(logging.DEBUG)      # Log everything to the file
    ch.setLevel(logging.DEBUG)      # as well as to the console
    return log


# A little hack to allow re-loading the file into ipython w/o
# multiplying the logger instances
if 'log' not in globals():
    log = activate_logger()


# -----------------------------------------------------------
# Image convenience Functions


def read_image(file: PathOrStr) -> ImageArray:
    """Read an image at FILE."""
    src_path = Path(file)
    # opencv2 honors the image rotation
    # TODO Try pillow instead
    # Array = (h, w, 3)
    img = cv2.imread(f"{src_path}")[:, :, :: -1]
    log.info(f"Read image {src_path} {img.shape[:2]}")
    return img


def write_image(file: PathOrStr, img) -> None:
    """Store the image (RGB or binary mask)."""
    dest = Path(file)
    imgtype = " "
    if img.dtype == bool:
        img = img_as_ubyte(img, force_copy=True)
        imgtype = " binary "
    log.info(f"Storing{imgtype}image into {dest}")
    imsave(f"{dest}", img, check_contrast=False)


def dimensions2d(img: ImageArray) -> Tuple[int, int]:
    """Return the 2d dimensions of the original image (w,h)."""
    h, w = img.shape[:2]
    return (w, h)


# -----------------------------------------------------------
# The actual transformation


def create_binary_mask(img, radius: int = 20) -> ImageArray:
    """Transform IMG using grayscale, binarizing, dilation."""
    log.info("Creating binary mask")
    img = rgb2gray(img)
    thresh = threshold_sauvola(img, window_size=101)
    img = img <= thresh
    img = isotropic_dilation(img, radius=radius)  # 40
    return img

# -----------------------------------------------------------
# Segmenter object:
# Initialize with an untransformed image. It creates two lists of bounding
# boxes which can be accessed: word_boxes and line_boxes.
# As a rule of thumb, all functions prefixed with _ do not modify the
# state of the object.


class Segmenter:
    """
    Initialized with an image, the object transforms it, finds bounding boxes of the binarized image
    and builds clusters of all boxes who are within a certain distance to each other.

    Two results are stored in the object's instance as attributes:

    word_boxes - A list of bounding boxes for 'words' (contours of a certain size).
    line_boxes - A list of bounding boxes for 'lines' (bounding boxes of clusters of words)

    Note that a 'bounding box', here, is defined as (y_min, x_min, y_max, x_max), inverting the
    usual order of x and y. This is the way scikit image uses it (calling it then 'rows' and 'columns'),
    and we adopt this use.
    """

    img: ImageArray
    word_boxes: list[BBox]
    line_boxes: list[BBox]

    def __init__(self,
                 img: ImageArray,
                 transformer: Optional[Callable] = None,
                 cluster_padding: Tuple[int, int] = (150, 150)) -> None:
        """
        Create a binary mask using TRANSFORMER and find boxes, storing them in the object.
        TRANSFORMER is a function accepting an ImageArray as its sole argument.
        """
        if transformer is None:
            transformer = create_binary_mask
        self.img = transformer(img)
        self.word_boxes = []
        self.line_boxes = []
        self.find_word_boxes()
        self.find_line_boxes(cluster_padding)

    def _overlap(self, box1: BBox, box2: BBox) -> bool:
        """Check if BOX1 and BOX2 overlap.
        Boxes are scikit image regions (y_top,x_top,y_bot,y_bot)."""
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2
        # No overlap if area = 0
        if y1_min == y1_max or x1_min == x1_max \
           or y2_min == y2_max or x2_min == x2_max:
            return False
        # No overlap if one rectangle is on left side of the other
        if x1_min > x2_max or x2_min > x1_max:
            return False
        # No overlap if one rectangle is above another:
        if y1_min > y2_max or y2_min > y1_max:
            return False
        return True

    def _pad_box(self, box: BBox, dist: Tuple[int, int]) -> BBox:
        """Return a box padded by dist and dist."""
        x_dist, y_dist = dist
        y_min, x_min, y_max, x_max = box
        w, h = dimensions2d(self.img)
        return (max(y_min - y_dist, 0),
                max(x_min - x_dist, 0),
                min(y_max + y_dist, h),
                min(x_max + x_dist, w))

    def _touching_boxes(self, box: BBox, candidates: list[BBox], dist: Tuple[int, int]) -> list[BBox]:
        """Return all boxes in CANDIDATES which are touched by the padding box around SEED."""
        padded_box = self._pad_box(box, dist)
        return [cand for cand in candidates if self._overlap(padded_box, cand)]

    def _bounding_box(self, boxes: list[BBox]) -> BBox:
        """Return the box bounding all BOXES."""
        y_min = min([b[Y_MIN] for b in boxes])
        x_min = min([b[X_MIN] for b in boxes])
        y_max = max([b[Y_MAX] for b in boxes])
        x_max = max([b[X_MAX] for b in boxes])
        return (y_min, x_min, y_max, x_max)

    def _greedy_collect(self,
                        seed: BBox,
                        candidates: list[BBox],
                        dist: Tuple[int, int]) -> list[BBox]:
        """Starting from SEED, return all boxes in CANDIDATES which overlap
        with the padding box around SEED defined by DIST."""
        if not seed:
            return []
        if not candidates:
            return [seed]
        res = [seed]
        new_seeds = self._touching_boxes(seed, candidates, dist)
        for new_seed in new_seeds:
            res += self._greedy_collect(new_seed,
                                        list(set(candidates) - set(res)),
                                        dist)
        return list(set(res))

    def find_word_boxes(self) -> None:
        """Find bounding boxes on the object's img."""
        log.info("Looking for word boxes")
        x_dim, y_dim = dimensions2d(self.img)
        # Find regions with 1s
        label_img, count = label(self.img, connectivity=self.img.ndim, return_num=True)
        props = regionprops(label_img)

        boxes = []   # Return value
        for i, region in enumerate(props):
            # only take regions which:
            #  - do not take up more than one third of the image in
            #    either dimension
            #  - which have a certain extent (area of whole image /
            #     area of box: the closer to 1, the smaller the extent)
            y_min, x_min, y_max, x_max = region.bbox
            height = y_max - y_min
            width = x_max - x_min
            one_third_x = width > (x_dim / 3)
            one_third_y = height > (y_dim / 3)
            if round(region.extent, 2) < 0.75 and not one_third_x and not one_third_y:
                boxes += [region.bbox]
        n = len(boxes)
        log.info(f"Found {n} word boxes")
        self.word_boxes = boxes

    def find_line_boxes(self,
                        padding: Tuple[int, int] = (150, 150),
                        boxes: Optional[list[BBox]] = None) -> None:
        """Find clusters where BOXES all overlap within PADDING distance.
        Store the result in self.line_boxes."""
        log.info("Looking for line boxes")
        if boxes is None:
            boxes = self.word_boxes
        joined_boxes = []
        while boxes:
            group = self._greedy_collect(boxes[0], boxes, padding)
            joined_boxes += [self._bounding_box(group)]
            boxes = list(set(boxes) - set(group))
        n = len(joined_boxes)
        log.info(f"Found {n} line boxes")
        self.line_boxes = joined_boxes


class WordSegmenter(Segmenter):
    """Segmenter specialized for identifying words."""

    def __init__(self, img: ImageArray) -> None:
        """Initialize a segmenter using a specialized binary mask for word.."""
        def my_transformer(img):
            create_binary_mask(img, radius=20)
        super().__init__(img, my_transformer)


class LineSegmenter(Segmenter):
    """Segmenter specialized for identifying lines."""

    def __init__(self, img: ImageArray) -> None:
        """Initialize a segmenter using a specialized binary mask for lines."""
        def my_transformer(img):
            create_binary_mask(img, radius=40)
        super().__init__(img, my_transformer, (180, 180))


# -----------------------------------------------------------
class ImageWorker:

    img: ImageArray

    def __init__(self, img: ImageArray) -> None:
        self.img = img.copy()

    def write(self, file: PathOrStr) -> None:
        """Store the image."""
        write_image(file, self.img)

    def foreach_box(self, boxes: list[BBox], fn: Callable,
                    with_index=False) -> Any:
        """Call fn for each box, accumulating the results.
        FN must have the the signature fn(box) or, if
        WITH_INDEX is set to True, fn(box, i)"""
        if with_index:
            return [fn(box, i) for i, box in enumerate(boxes)]
        else:
            return [fn(box) for box in boxes]

    def get_slice(self, box: BBox) -> ImageArray:
        """Get the image slice bounded by BOX from the original image."""
        y_min, x_min, y_max, x_max = box
        return self.img[y_min:y_max, x_min:x_max]

    def boxes_as_images(self, boxes: list[BBox]) -> list[ImageArray]:
        """Return slices of the image defined by BOXES."""
        return self.foreach_box(boxes, self.get_slice)

    def draw_rectangle(self, box: BBox, color: Color) -> None:
        """Draw a rectangle around BOX on the img."""
        start = (box[Y_MIN], box[X_MIN])
        end = (box[Y_MAX], box[X_MAX])
        rr, cc = rectangle_perimeter(start, end)
        set_color(self.img, (rr, cc), color)

    def draw_box(self, box: BBox, color: Color) -> None:
        """Fill BOX with COLOR."""
        start = (box[Y_MIN], box[X_MIN])
        end = (box[Y_MAX], box[X_MAX])
        rr, cc = rectangle(start, end)
        set_color(self.img, (rr, cc), color)

    def draw_boxes(self, boxes: list[BBox],
                   color: Color = (255, 0, 0)) -> None:
        """Draw filled BOXES."""
        def color_box(box):
            self.draw_box(box, (255, 0, 0))
        self.foreach_box(boxes, color_box)

    def draw_rectangles(self, boxes: list[BBox],
                        color: Color = (255, 0, 0)) -> None:
        """Draw BOXES as rectangles."""
        def color_rectangle(box):
            self.draw_rectangle(box, color)
        self.foreach_box(boxes, color_rectangle)

    def write_box(self, box: BBox, filename: PathOrStr):
        """Store content of BOX in FILENAME."""
        box_for_output = (box[X_MIN], box[Y_MIN],
                          box[X_MAX], box[Y_MAX])
        dest = Path(filename)
        log.info(f"Storing box {box_for_output} into {dest}")
        imsave(f"{dest}", self.get_slice(box), check_contrast=False)

    def write_boxes(self, boxes: list[BBox],
                    filename: PathOrStr) -> None:
        """Store all BOXES in FILENAME, appending a number."""
        path = Path(filename)
        basename = path.stem

        def write_indexed_box(box, i):
            self.write_box(box, path.with_stem(f"{basename}_{i:03}"))
        self.foreach_box(boxes, write_indexed_box, with_index=True)

    def show(self) -> None:
        """Show Image using PIL."""
        ImageShow.register(ImageShow.DisplayViewer, 0)
        with Image.fromarray(self.img) as im:
            im.thumbnail((1000, 1000))
            im.show()

# -----------------------------------------------------------
# Example use:


def example_use():
    """Example usage."""
    input_img = Path('/home/jv/Bilder/022.jpg')
    output_path = Path('/home/jv/Bilder/splittest/')
    img = read_image(input_img)
    seg = Segmenter(img)

    worker = ImageWorker(img)
    worker.write_boxes(seg.word_boxes, output_path / "word.png")
    worker.write_boxes(seg.line_boxes, output_path / "line.png")

    worker = ImageWorker(img)
    worker.draw_rectangles(seg.word_boxes)
    worker.write(output_path / "original_with_word_boxes.png")
    worker = ImageWorker(img)
    worker.draw_boxes(seg.word_boxes)
    worker.draw_rectangles(seg.line_boxes)
    worker.write(output_path / "original_with_line_boxes.png")
