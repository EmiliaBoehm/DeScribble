"""Prepare and load the data for the model."""

import pandas
import numpy as np
import sys
from PIL import ImageShow
#  from PIL import Image as PIL_Image  # conflicts with datasets.features.Image
from pathlib import Path
from datasets import Dataset
from datasets.features import Features, Image, ClassLabel
from torchvision.transforms import RandomResizedCrop, Compose, ToTensor, ToPILImage
from torch import Tensor
from transformers import AutoImageProcessor
from typing import Union, TypeAlias, Callable, Optional
#import logger
import logging
import config as cfg

#global log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PathOrStr: TypeAlias = Union[Path, str]

# FIELD NAMES OF THE CSV RAW INDEX FILE
CSV_GENDER = 'Geschlecht'
CSV_INDEX = 'Datei_Index'

# LABELS
LABELS = ['Männlich', 'Weiblich']
label2id = {'Männlich': 0,
            'Weiblich': 1}
id2label = {0: 'Männlich',
            1: 'Weiblich'}

# CHECKPOINTS
#CHECKPOINT="microsoft/trocr-base-handwritten"
#CHECKPOINT="google/vit-base-patch16-384"
CHECKPOINT_VIT_ONLINE = "google/vit-base-patch16-224-in21k"


# TRANSFORM PIPELINE
_transforms = None


def get_index_table() -> pandas.DataFrame:
    """Return a pandas dataframe object with labels and participant index."""
    config = cfg.Config()
    path = config.get_index()
    cfg.assert_path(path)
    df = pandas.read_csv(path)
    # Delete trailing whitespace and convert blanks to underscores:
    df = df.rename(columns=lambda s: s.strip().replace(" ", "_"))
    # Remove columns without 'Geschlecht' value:
    df = df[~df['Geschlecht'].isna()]
    log.debug(f"Loaded raw index table {path.name} with shape {df.shape}")
    return df


def get_image_list() -> list[dict]:
    """Return a dictionary list with the keys 'index', 'gender', 'img').

    Iterate over all segmented images and associate those where
    image_filter(filename) returns True."""
    index_table = get_index_table()
    config = cfg.Config()
    src = config.get_image_segmented().resolve()
    cfg.assert_path(src)
    result = []
    for idx, row in index_table.iterrows():
        participant_index = row[CSV_INDEX]
        subdir = src / f"{participant_index:03}"
        if not subdir.exists():
            log.info(f"Segmented images for participant {participant_index} not found")
            continue
        for file in subdir.glob('*'):
            if file.name.endswith(('jpg', 'jpeg', 'png')) \
               and file.name.startswith('word'):
                result.append({'index': participant_index,
                               'gender': row[CSV_GENDER],
                               'img': f"{file}"})
    return result


def collect_statistics(image_list: list[dict]) -> dict:
    """Return some statistics on IMAGE_LIST."""
    n_total = len(image_list)
    n_male = len([x for x in image_list if x['gender'] == 'Männlich'])
    n_female = len([x for x in image_list if x['gender'] == 'Weiblich'])
    n_per_participant: dict = {}
    for row in image_list:
        idx = row['index']
        count = n_per_participant.get(idx, 0)
        n_per_participant[idx] = count + 1
    return {'n_total': n_total,
            'n_male': n_male,
            'n_female': n_female,
            'n_per_participant': n_per_participant}


def get_dataset(image_list: list[dict]) -> Dataset:
    """Turn image_list into a dataset."""
    ft = Features({'label': ClassLabel(names=LABELS),
                   'image': Image()})
    data_list = [{'label': row['gender'],
                  'image': row['img']}
                 for row in image_list]
    ds = Dataset.from_list(data_list, features=ft)
    return ds


def get_processor(checkpoint: str):
    """Return an imageprocessor."""
    return AutoImageProcessor.from_pretrained(checkpoint)


# NOTE Debug with  _transforms(ds[0]['image'])
def set_transforms(processor=None):
    """Define the _transforms pipeline."""
    if processor is None:
        processor = get_processor(CHECKPOINT_VIT_ONLINE)
    global _transforms
    _transforms = Compose([
        RandomResizedCrop((processor.size['height'],
                           processor.size['width'])),
        ToTensor(),
        # Normalize()
    ])


def transform_vit(batch: dict):
    """Transform image batches from a Dataset to torch tensor pixel values."""
    if _transforms is None:
        log.critical("_transforms pipeline undefined, cannot proceed")
        sys.exit(1)
    batch['pixel_values'] = [_transforms(img.convert("RGB")) for img in batch['image']]
    del batch['image']
    return batch


def convert_tensor_to_1d_array(t: Tensor) -> np.array:
    """Convert tensor image to a 1-dimensional numpy array."""
    r = np.asarray(ToPILImage()(t))
    nx, ny, nrgb = r.shape
    new_r = r.reshape(nx * ny * nrgb)
    new_r = new_r / 255
    return new_r


def transform_random_forest(batch: dict):
    """Transform image batches from a Dataset to """
    if _transforms is None:
        log.critical("_transforms pipeline undefined, cannot proceed")
        sys.exit(1)
    # First apply the same transformation as transform_vit()
    batch['pixel_values'] = [_transforms(img.convert("RGB")) for img in batch['image']]
    # Then convert the values to per-batch np arrays:
    batch['pixel_values'] = np.asarray([convert_tensor_to_1d_array(px)
                                        for px in batch['pixel_values']])
    batch['label'] = np.asarray([[label] for label in batch['label']])
    return batch


def get_dataset_with_transform(image_list: list[dict],
                               batch_transformer: Optional[Callable] = None,
                               processor=None) -> Dataset:
    """Turn image_list into a dataset."""
    ds = get_dataset(image_list)
    if processor is None:
        processor = get_processor(CHECKPOINT_VIT_ONLINE)
    if batch_transformer is None:
        batch_transformer = transform_vit
    set_transforms(processor)
    ds.set_transform(batch_transformer)
    return ds


def show_ds_img(ds_dict: dict) -> None:
    """Display the image stored as 'pixel_values' in a DS row."""
    pv = ds_dict['pixel_values']
    img = None
    if type(pv) is Tensor:
        img = ToPILImage()(pv)
    if type(img) is None:
        log.critical("Cannot display image, unknown data type {type(pv)}")
    ImageShow.register(ImageShow.DisplayViewer, 0)
    ImageShow.show(img)


# img = tensor_to_image(tensor)
# plt.imshow(img)
# plt.show()
