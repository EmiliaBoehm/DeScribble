"""Segment an image of handwritten text into lines and words."""

from __future__ import annotations
import cv2
from pathlib import Path
import sys
from PIL import Image, ImageShow
#from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches
from skimage.io import imsave
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
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(funcName)s() - %(message)s", "%Y-%m-%d %H:%M:%S")
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


def image_is_a_mask(img: ImageArray) -> bool:
    """Return True if image `img` is a boolean mask."""
    return (img.ndim == 2) and (img.dtype == bool)


def read_image(file: PathOrStr) -> ImageArray:
    """
    Read the image at `file`.

    Args:

       file: String or Path object

    Returns:

       RGB ImageArray
    """
    src_path = Path(file)
    # opencv2 honors the image rotation
    # TODO Try pillow instead
    # Array = (h, w, 3)
    img = cv2.imread(f"{src_path}")[:, :, :: -1]
    log.info(f"Read image {src_path} {img.shape[:2]}")
    return img


def write_image(file: PathOrStr, img: ImageArray) -> None:
    """
    Store the image (either RGB or binary mask).
    """
    dest = Path(file)
    imgtype = " "
    if image_is_a_mask(img):
        img = mask2rgb(img.copy())
        imgtype = " binary "
    log.info(f"Storing{imgtype}image into {dest}")
    imsave(f"{dest}", img, check_contrast=False)


def dimensions2d(img: ImageArray) -> Tuple[int, int]:
    """
    Return the 2d dimensions of the original image `img` (w,h),
    regardless of its dimensions.
    """
    h, w = img.shape[:2]
    return (w, h)


# -----------------------------------------------------------
# Binary Masks


def mask2rgb(mask: ImageArray, invert: bool = False) -> ImageArray:
    """
    Convert binary image `mask` to a b/w RGB image with three channels.

    Args:

        mask: ImageArray of the shape (h,w)
        invert: If True, invert the image colors.

    Returns:

        An ImageArray of shape (H,W,3)

    """
    if invert:
        mask = np.invert(mask)
    # Stack the mask as RGB Layers: True -> (True, True True)
    new_img = np.stack((mask, mask, mask), axis=2)
    # Converting to uint returns 0 or 1; then multiply by 255
    return new_img.astype('uint8') * 255


def binarize(img: ImageArray) -> ImageArray:
    """
    Turn RGB image `img` into a binary image using Sauvola threshold algorithm.
    """
    if image_is_a_mask(img):
        log.error("Asked to convert image into a mask, but it is already one.")
        return img
    log.info("Creating binary mask")
    img = rgb2gray(img)
    thresh = threshold_sauvola(img, window_size=101)
    return (img <= thresh)


def create_binary_mask(img, radius: int = 20) -> ImageArray:
    """
    Transform `img`  using grayscale, binarizing, dilation.
    """
    img = isotropic_dilation(binarize(img), radius=radius)  # 40
    return img


# -----------------------------------------------------------
# Segmenter object:
# Initialize with an untransformed image. It creates two lists of bounding
# boxes which can be accessed: word_boxes and line_boxes.
# As a rule of thumb, all functions prefixed with _ do not modify the
# state of the object.


class Segmenter:
    """
    Initialized with an image, the object transforms it, finds bounding
    boxes of the binarized image and builds clusters of all boxes who are
    within a certain distance to each other. The results are stored in the
    object's instance as attributes:

          word_boxes - A list of bounding boxes for 'words'
                       (contours of a certain size).
          line_boxes - A list of bounding boxes for 'lines'
                       (bounding boxes of clusters of words)

    Note that a 'bounding box', here, is defined as
    (y_min, x_min, y_max, x_max), inverting theusual order of x and y.
    This is the way scikit image uses it (calling y 'rows' and x 'columns').
    """

    img: ImageArray
    word_boxes: list[BBox]
    line_boxes: list[BBox]

    def __init__(self,
                 img: ImageArray,
                 transformer: Optional[Callable] = None,
                 cluster_padding: Tuple[int, int] = (150, 150)) -> None:
        """
        Create a binary mask using `transformer` and find boxes, storing them in the object.

        Args:

          transformer: A function accepting an ImageArray as its sole argument. The fucntion must
                       also return an ImageArray (either binary or RGB).
        """
        if transformer is None:
            transformer = create_binary_mask
        self.img = transformer(img)
        if self.img is None:
            log.fatal("Could not get a binary mask, canceling.")
            sys.exit(1)
        self.word_boxes = []
        self.line_boxes = []
        self.find_word_boxes()
        self.find_line_boxes(cluster_padding)

    def _overlap(self, box1: BBox, box2: BBox) -> bool:
        """
        Check if `box1` and `box2` overlap.

        Args:

           box1, box2: boxes (y,x,y,x)
        """
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
        """
        Return a box padded by dist and dist. Clip the resulting
        box at the image borders, if necessary.

        Args:

            box: box (y,x,y,x)
            dist: tuple (x-pad, y-pad)

        Returns:

            padded box

        """
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
        """
        Return the box bounding all `boxes`.

        Args:
            boxes: list of boxes (y,x,y,x)

        Returns:

           box surrounding all `boxes`.
        """
        y_min = min([b[Y_MIN] for b in boxes])
        x_min = min([b[X_MIN] for b in boxes])
        y_max = max([b[Y_MAX] for b in boxes])
        x_max = max([b[X_MAX] for b in boxes])
        return (y_min, x_min, y_max, x_max)

    def _greedy_collect(self,
                        seed: BBox,
                        candidates: list[BBox],
                        dist: Tuple[int, int]) -> list[BBox]:
        """
        Starting from `seed`, return all boxes in `candidates` which overlap
        with the padding box around `seed` defined by `dist`.

        Args:
            seed:   Box (y,x,y,x)
            candidates: list of boxes
            dist:  tuple (x-pad, y-pad)

        Returns:

            List of boxes
        """
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
        """
        Find bounding boxes on the object's img.

        Store the result in `self.word_boxes`.
        """
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
        """
        Find clusters where `boxes` all overlap within `padding` distance.
        Store the result in `self.line_boxes`.
        """
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
        def my_transformer(img) -> ImageArray:
            return create_binary_mask(img, radius=20)
        super().__init__(img, my_transformer)


class LineSegmenter(Segmenter):
    """Segmenter specialized for identifying lines."""

    def __init__(self, img: ImageArray) -> None:
        """Initialize a segmenter using a specialized binary mask for lines."""
        def my_transformer(img) -> ImageArray:
            return create_binary_mask(img, radius=40)
        super().__init__(img, my_transformer, (180, 180))


# -----------------------------------------------------------
class ImageWorker:
    """
    'Work' with image boxes by drawing them on an image or storing them as slices.

    """

    img: ImageArray

    def __init__(self, img: ImageArray, invert=True) -> None:
        """
        Load `img` in order to work with it.

        If `img` is a binary mask, automatically convert it to b/w RGB.
        Invert the mask unless `invert` is False.
        """
        self.img = img.copy()
        if self.img.ndim == 2:
            self.img = mask2rgb(self.img, invert)
            log.info("Transformed binary image to RGB")

    def write(self, file: PathOrStr) -> None:
        """Store the image."""
        write_image(file, self.img)

    def foreach_box(self, boxes: list[BBox], fn: Callable,
                    with_index=False) -> Any:
        """
        Call `fn` for each box, accumulating the results.

        Function `fn` must have the the signature `fn(box)` or, if
        WITH_INDEX is set to True, `fn(box, i)`.
        """
        if with_index:
            return [fn(box, i) for i, box in enumerate(boxes, 1)]
        else:
            return [fn(box) for box in boxes]

    def get_slice(self, box: BBox) -> ImageArray:
        """
        Get the image slice bounded by `box` from the original image.

        Args:

            box: tuple (y,x,y,x)

        Returns:

            ImageArray of shape (H,W,3)
        """
        y_min, x_min, y_max, x_max = box
        return self.img[y_min:y_max, x_min:x_max]

    def boxes_as_images(self, boxes: list[BBox]) -> list[ImageArray]:
        """
        Return slices of the image defined by `boxes`.

        Args:

            boxes: list of boxes (y,x,y,x)

        Returns:

            Lists of ImageArrays of shape (H,W,3)
        """
        return self.foreach_box(boxes, self.get_slice)

    def draw_rectangle(self, box: BBox, color: Color) -> None:
        """
        Draw a rectangle around BOX on the img.

         Args:

            box:  tuple (y,x,y,x)
            color: RGB tuple
        """
        start = (box[Y_MIN], box[X_MIN])
        end = (box[Y_MAX], box[X_MAX])
        rr, cc = rectangle_perimeter(start, end)
        set_color(self.img, (rr, cc), color)

    def draw_box(self, box: BBox, color: Color) -> None:
        """Fill BOX with COLOR.

        Args:

            box:  tuple (y,x,y,x)
            color: RGB tuple
        """
        start = (box[Y_MIN], box[X_MIN])
        end = (box[Y_MAX], box[X_MAX])
        rr, cc = rectangle(start, end)
        set_color(self.img, (rr, cc), color)

    def draw_boxes(self, boxes: list[BBox],
                   color: Color = (255, 0, 0)) -> None:
        """Draw filled BOXES.

        Args:

            boxes: list of boxes (y,x,y,x)
            color: RGB tuple
        """
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
        """Store content of BOX in FILENAME.

        Args:

              box: Tuple (y,x,y,x)
              filename: Full path to the file, including suffix.
        """
        box_for_output = (box[X_MIN], box[Y_MIN],
                          box[X_MAX], box[Y_MAX])
        dest = Path(filename)
        log.info(f"Storing box {box_for_output} into {dest}")
        imsave(f"{dest}", self.get_slice(box), check_contrast=False)

    def write_boxes(self, boxes: list[BBox],
                    filepattern: PathOrStr) -> None:
        """Store all BOXES in FILEPATTERN with index appended.

        Args:

              boxes: List of boxes (y,x,y,x)
              filename: Full path pattern for generating the filenames.

        Example:

              `worker.write_boxes(boxes, '../src/images/lines.png')`
                 will write the boxes in the files:
                              `../src/images/lines-001.png`
                              `../src/images/lines-002.png`
                              ....
        """
        path = Path(filepattern)
        basename = path.stem
        if not basename:
            log.fatal("Filename missing")
            sys.exit(0)

        if not path.parent.exists():
            log.info(f"Creating non-existing path {path.parent} on the fly")
            path.parent.mkdir(parents=True)

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


TESTBILD_BW = '/home/jv/Bilder/022_bw.jpg'
TESTBILD = '/home/jv/Bilder/022.jpg'


def example_test():
    img = read_image(TESTBILD)
    wseg = WordSegmenter(img)
    lseg = LineSegmenter(img)
    # Draw lines on the mask!
    wworker = ImageWorker(wseg.img)
    wworker.draw_rectangles(wseg.word_boxes)
    lworker = ImageWorker(lseg.img)
    lworker.draw_rectangles(lseg.line_boxes)
    wworker.show()
    lworker.show()


def example_use():
    """Example usage."""
    input_img = Path(TESTBILD_BW)
    output_path = Path('/home/jv/Bilder/splittest2/')
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
