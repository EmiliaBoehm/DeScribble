"""Prepare and load the data for the model."""

import PIL
import pandas
import numpy as np
from pathlib import Path
from datasets import Dataset, load_dataset
from datasets.features import Features, Image, ClassLabel
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize, ToPILImage
from transformers import AutoImageProcessor, AutoFeatureExtractor, VisionEncoderDecoderModel, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer
from matplotlib import pyplot as plt
from typing import Union, TypeAlias, Callable, Generator
#import logger
import logging
import sys
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


def transform_ds_batch(batch: dict):
    """Transform image in Dataset row to pixel values."""
    if _transforms is None:
        log.critical("_transforms pipeline undefined, cannot proceed")
        sys.exit(1)
    batch['pixel_values'] = [_transforms(img.convert("RGB")) for img in batch['image']]
    del batch['image']
    return batch
    # return {key: batch[key] for key in batch
    #         if key not in {'pixel_values', 'label'}}


def get_dataset_with_transform(image_list: list[dict], processor=None) -> Dataset:
    """Turn image_list into a dataset."""
    ds = get_dataset(image_list)
    if processor is None:
        processor = get_processor(CHECKPOINT_VIT_ONLINE)
    set_transforms(processor)
    ds.set_transform(transform_ds_batch)
    return ds


def show_ds_img(ds_dict: dict):
    """Display the image stored as 'pixel_values' in a DS row."""
    pv = ds_dict['pixel_values']
    img = ToPILImage(pv)
    return img


# img = tensor_to_image(tensor)
# plt.imshow(img)
# plt.show()
