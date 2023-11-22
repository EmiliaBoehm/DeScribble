"""Load the pretrained VIT Image Classification model from huggingface.com and run it with example images."""

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
from transformers import pipeline
import random

#Create Dataset

#Create Dataset: Define ids for target labels
labels = ['male', 'female', 'diverse']
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label]      = i
    id2label[i]     = label

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
    for img_path in (images_path / idx).glob('*.JPG'):
        all_images.append({ 'label': label, 'path': img_path })

# _generator = ({'label': l['label'], 'image': Image.open(l['path'])} for l in all_images) 
#ds = Dataset.from_generator(_generator)
def _generator():
    for l in all_images:
        yield {'label': label2id[l['label']], 
               'image': Image.open(l['path'])}

"""Format Dataset: TrainTestSplit"""
ds=Dataset.from_generator(_generator)
ds = ds.train_test_split(test_size=0.2)

"""Set up model /training"""
data_collator = DefaultDataCollator()
accuracy = evaluate.load("accuracy")

"""Set up model Evaluation"""
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # highest prediction per label index
    return accuracy.compute(predictions=predictions, references=labels)

"""Prepare model"""
model = AutoModelForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="./local_checkpoints/",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

"""Train model"""
trainer.train()