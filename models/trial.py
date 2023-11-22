
from transformers import AutoImageProcessor, AutoFeatureExtractor, TrOCRProcessor, VisionEncoderDecoderModel, DefaultDataCollator
import requests
from PIL import Image
import torch
from pathlib import Path
from datasets import Dataset, load_dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize, ToPILImage
from matplotlib import pyplot as plt
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

"""Create Dataset: Load images"""
IMAGES = '..\privateDir\example_images\segmented'
images_path = Path(IMAGES)

images_labels = {   '001': 'male',
                    '002': 'male',
                    '003': 'male',
                    '004': 'male',
                    '005': 'male',
                    '006': 'male',
                    '007': 'male',
                    '008': 'male',
                    '020' : 'female',
                    '021' : 'female',
                    '022' : 'female'}

all_images = []
for idx, label in images_labels.items():
    for img_path in (images_path / idx).glob('*.jpeg'):
        all_images.append({ 'label': label, 'path': img_path })

"""Create Dataset: Define IDs for target labels"""
labels = ['male', 'female', 'diverse']
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label]      = i
    id2label[i]     = label

print(label2id, id2label)

# _generator = ({'label': l['label'], 'image': Image.open(l['path'])} for l in all_images) 
#ds = Dataset.from_generator(_generator)
def _generator():
    for l in all_images:
        yield {'label': label2id[l['label']], 
               'image': Image.open(l['path'])}

ds=Dataset.from_generator(_generator)
ds = ds.train_test_split(test_size=0.2)

"""Preprocess Image"""

# Load Processor for ViT

CHECKPOINT="google/vit-base-patch16-224-in21k"
# Alternative: TrOCRProcessor.from_pretrained(CHECKPOINT).image_processor
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Normalize Input Image
# evtl. use processor(img), but this returns an object with different dimensions
def tensor_to_image(t) -> Image:
    transform=ToPILImage()
    return transform(t)

normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
# in other processors, maybe use 'shortest_edge'
size = ( processor.size["height"], processor.size["width"] )
# JV suggests: do not normalize bc these are no food images
#_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
#_transforms = Compose([Resize(size), ToTensor(), normalize])
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

print(size)
def transforms(examples):
    """Transform the image in the Dataset row."""
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

ds = ds.with_transform(transforms)
tensor = ds['train'][0]['pixel_values']

img = tensor_to_image(tensor)
#plt.imshow(img)
#plt.show()