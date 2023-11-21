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
import yaml

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
        return img
    log.info("Creating binary mask")
    img = rgb2gray(img)
    thresh = threshold_sauvola(img, window_size=101)
    return (img <= thresh)


def create_binary_mask(img, radius: int = 20) -> ImageArray:
    """
    Transform `img`  using grayscale, binarizing, dilation.

    Args:
          img: A binary or an RGB image. If it is a binary image,
               skip the binarization.
    """
    if not image_is_a_mask(img):
        img = binarize(img)
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

    word_mask: ImageArray
    binary_img: ImageArray
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
        self.binary_img = binarize(img)
        if self.binary_img is None:
            log.critical("Could not binarize the image, canceling.")
            sys.exit(1)
        self.word_mask = transformer(self.binary_img)
        if self.word_mask is None:
            log.critical("Could not transform the binary image, canceling.")
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
        w, h = dimensions2d(self.word_mask)
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
        x_dim, y_dim = dimensions2d(self.word_mask)
        # Find regions with 1s
        label_img, count = label(self.word_mask, connectivity=self.word_mask.ndim, return_num=True)
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
        log.info("Segmenting for words")

        def my_transformer(img) -> ImageArray:
            return create_binary_mask(img, radius=20)
        super().__init__(img, my_transformer)


class LineSegmenter(Segmenter):
    """Segmenter specialized for identifying lines."""

    def __init__(self, img: ImageArray) -> None:
        """Initialize a segmenter using a specialized binary mask for lines."""
        log.info("Segmenting for lines")

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
        if image_is_a_mask(self.img):
            self.img = mask2rgb(self.img, invert)
            log.info("Transforming binary image to b/w RGB")

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
                    dest_dir: PathOrStr,
                    filepattern: str,
                    **pattern_map) -> None:
        """Store all BOXES in FILEPATTERN with index appended.

        Args:

              boxes: List of boxes (y,x,y,x)
              dest_dir: Directory to store the files in. If it does not exist,
                        it will be created on the fly.
              filepattern: A string which will be passed to str.format.
                           Use {i} for the running index.
              pattern_map: Any other values passed to `filepattern`,
                           which will be expanded with `str.format`.

        Example:

              `worker.write_boxes(boxes, 'src/images/',
                                  "lines-{i:03}.png")`
                 will write the boxes in the files:
                              `src/images/lines-001.png`
                              `src/images/lines-002.png`
                              ....
        """
        path = Path(dest_dir)
        if not path.exists():
            log.info(f"Creating non-existing path {path} on the fly")
            path.mkdir(parents=True)

        def write_indexed_box(box, i):
            filename = filepattern.format(**pattern_map | {'i': i})
            self.write_box(box, path / filename)
        self.foreach_box(boxes, write_indexed_box, with_index=True)

    def show(self) -> None:
        """Show Image using PIL."""
        ImageShow.register(ImageShow.DisplayViewer, 0)
        with Image.fromarray(self.img) as im:
            im.thumbnail((1000, 1000))
            im.show()

# -----------------------------------------------------------


class Pipeline:
    """
    Initialize and run a preprocessing pipeline.
    """

    config: dict
    root_path: Path

    def __init__(self,
                 yaml_file: Optional[PathOrStr] = None,
                 root_node: str = 'preprocess') -> None:
        """
        Initialize a pipeline using `yaml_file`.

        If no argument is given, look for a YAML file called `params.yaml`
        in the project's root directory.  The root directory
        is determined by locating the directory `.git`.
        """
        if not yaml_file:
            root = self.get_root_path()
            results = list(root.glob('params.yaml'))
            if results:
                yaml_file = results[0]
            if not yaml_file:
                log.error(f"Could not locate pipeline configuration file {yaml_file}")
                sys.exit(1)
        config_path = Path(yaml_file)
        self.root_path = config_path.parent
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        if not config[root_node]:
            log.error(f"YAML configuration file {config_path}' has no root node '{root_node}'")
            sys.exit(1)
        self.config = config[root_node]

    def get_root_path(self) -> Path:
        """
        Find the root directory which as a `.git` directory.
        If none is found, return the current directory.
        """
        path = Path.cwd()
        for dir in path.resolve().parents:
            if (dir / ".git").exists():
                path = dir
        return path

    def validate_image_sources(self) -> bool:
        """
        Check if image sources are valid file paths.
        """
        pass
        return True

    def get_param(self, tree_path: str, separator: str = '/',
                  log_not_found: bool = False) -> Any:
        """
        Find the configuration parameter defined by the 'tree-path'.

        Example:
                    Pipeline.get_param('image-sources/raw')

        Args:

           tree-path: String where  a slash designats a child node.
           separator: string separating the nodes, defaults to '/'
           log_not_found: Log an error if `tree_path` yields None.

        Returns:

           Any value associated with this path, or None.
        """
        paths = tree_path.strip(separator).split(separator)
        val = None
        config = self.config
        while paths and type(config) is dict:
            val = config.get(paths[0])
            if type(val) is dict:
                config = val
            paths = paths[1:]
        if not val and log_not_found:
            log.error(f"Could not find configuration value associated with {tree_path}")
        return val

    def pump_boxes(self,
                   src_path: Optional[PathOrStr] = None,
                   dest_path: Optional[PathOrStr] = None,
                   use_file: Union[None, PathOrStr, list[PathOrStr]] = None) -> None:
        """
        Read all images in the path stored in the config file, find boxes,
        and write the boxes to the corresponding subdirectory also defined in
        the config file.

        Args:
             src_path:   Source path or None. Defaults to the value stored in
                         `params.yaml`.

             dest_path:  Destination path or None. Defaults to the value stored
                         `params.yaml`. Subfoldes are created on the fly, if
                         necessary.

             use_file:   Apply the pipeline only on this particular file or list of files.
        """
        # First check the paths passed
        root = self.root_path
        src_path = root / self.get_param("images/bw", log_not_found=True) if src_path is None else src_path
        dest_path = root / self.get_param("images/segmented", log_not_found=True) if dest_path is None else dest_path
        if not src_path or not dest_path:
            log.fatal('Missing paths, cannot proceed.')
            sys.exit(1)
        src_path = Path(src_path).resolve()
        if not src_path.exists():
            log.fatal(f"Source path {src_path} not found")
            sys.exit(1)
        if not src_path.is_dir():
            log.fatal(f"Source path {src_path} must be a directory")
            sys.exit(1)
        dest_path = Path(dest_path).resolve()
        if dest_path.exists() and dest_path.is_file():
            log.fatal(f"Destination path {dest_path} seems to point to a file, quitting")
            sys.exit(1)
        if not dest_path.exists():
            log.info(f"Creating destination file {dest_path} on the fly")
            dest_path.mkdir(parents=True)
        # Define source files
        # (this should be generator, ideally)
        files = []
        if use_file is None:
            for suffix in ['jpg', 'jpeg']:
                files.extend([file for file in Path(src_path).glob(f"*.{suffix}")])
        else:
            use_file = use_file if isinstance(use_file, list) else [use_file]
            for file in use_file:
                files.append(src_path / Path(file))
        # and go!
        count = 0
        for file in files:
            log.info(f"Converting {file}:")
            path = Path(file)   # make sure it's a Path object
            if not path.is_file():
                log.error(f"Source file {path} does not exist")
                break      # skip this file
            count += 1
            stem = path.stem
            # TODO do some sanity check on stem
            dest = dest_path / stem
            dest.mkdir(exist_ok=True)
            # TEMP Delete files in target direcetory for debuging:
            for del_file in dest.glob('*'):
                del_file.unlink()
            img = read_image(file)
            wseg = WordSegmenter(img)
            lseg = LineSegmenter(wseg.binary_img)
            bw_worker = ImageWorker(wseg.binary_img)
            bw_worker.write_boxes(wseg.word_boxes, dest,
                                  "word-{stem}-{i:03}.png", stem=stem)
            bw_worker.write_boxes(lseg.line_boxes, dest,
                                  "line-{stem}-{i:03}.png", stem=stem)
            # For debugging: Add masks and original image
            write_image(dest / "0_binary_img.png", wseg.binary_img)
            write_image(dest / "0_original_img.png", img)
            write_image(dest / "0_word_mask.png", wseg.word_mask)
            write_image(dest / "0_line_mask.png", lseg.word_mask)
            bw_worker.draw_rectangles(wseg.word_boxes, (0, 255, 0))
            bw_worker.draw_rectangles(lseg.line_boxes)
            write_image(dest / "0_bw_worker.png", bw_worker.img)

        log.info(f"Segemented {count} files")







# -----------------------------------------------------------
# Example use:


TESTBILD_BW = '/home/jv/Bilder/022_bw.jpg'
TESTBILD = '/home/jv/Bilder/022.jpg'


def example_test():
    img = read_image(TESTBILD)
    wseg = WordSegmenter(img)
    lseg = LineSegmenter(img)
    # Draw lines on the mask!
    wworker = ImageWorker(wseg.binary_img)
    wworker.draw_rectangles(wseg.word_boxes)
    lworker = ImageWorker(lseg.binary_img)
    lworker.draw_rectangles(lseg.line_boxes)
    wworker.show()
    lworker.show()


def example_use():
    """Example usage."""
    input_img = Path(TESTBILD_BW)
    output_path = Path('/home/jv/Bilder/splittest3/')
    img = read_image(input_img)
    seg = Segmenter(img)

    worker = ImageWorker(img)
    worker.write_boxes(seg.word_boxes, output_path, "word-{i:03}.png")
    worker.write_boxes(seg.line_boxes, output_path, "line-{i:03}.png")

    worker = ImageWorker(img)
    worker.draw_rectangles(seg.word_boxes)
    worker.write(output_path / "original_with_word_boxes.png")
    worker = ImageWorker(img)
    worker.draw_boxes(seg.word_boxes)
    worker.draw_rectangles(seg.line_boxes)
    worker.write(output_path / "original_with_line_boxes.png")
