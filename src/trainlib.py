"""Library for dealing with training models."""
import logging
import sys
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoImageProcessor, DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import pipeline
from torchvision.transforms import ToPILImage
from typing import Any
import evaluate
from pathlib import Path
import pandas as pd
import datafeed as feed

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_data_and_split() -> DatasetDict:
    """Return dataset split in 'train', 'test' and 'validation'."""
    ls = feed.get_image_list()
    ds = feed.get_dataset_with_transform(ls, feed.transform_vit)
    return feed.split_dataset_train_test_validate(ds)


def get_model_and_processor(checkpoint) -> tuple[AutoModelForImageClassification, AutoImageProcessor]:
    """Return or laod the model and the processor."""
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(feed.id2label),
        id2label=feed.id2label,
        label2id=feed.label2id,
    )
    return (model, processor)


def train_model(save_dir: Path,
                model,
                processor,
                ds,
                batch_size=32,
                epochs=6) -> tuple[Trainer, Any]:
    """Train model with DataSet ds (split in 'train', 'test', 'validation').
    Return trainer object and train results object."""
    if type(ds) is Dataset:
        log.warning("Dataset is not split")

    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1) # highest prediction per label index
        return accuracy.compute(predictions=predictions, references=labels)

    checkpoints_dir = save_dir / "local_checkpoints"
    training_args = TrainingArguments(
        output_dir=f"{checkpoints_dir}",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
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
    trainer_result = trainer.train()
    return (trainer, trainer_result)


def save_model(save_dir: Path, trainer, train_results) -> None:
    """Save best model from training to SAVE_DIR/model"""
    model_dir = save_dir / "model"
    trainer.save_model(model_dir)                   # Will save the model, so you can reload it using from_pretrained()
    trainer.log_metrics("train", train_results.metrics)
    trainer.log_metrics("test", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics) # save metrics into a .json file
    trainer.save_state()


def get_pred_and_probablities(ds, model, processor) -> tuple[np.array, np.array]:
    """Return all predicitions and probabilities for DS."""
    if type(ds) is DatasetDict:
        log.warning("Cannot create predictions from a sliced dataset.")
        if ds.get('validation'):
            log.warning("Using slice 'validation' now")
            ds = ds['validation']
    classifier = pipeline("image-classification", model=model, image_processor=processor)
    y_pred = []
    y_prob = []
    for row in ds:
        img = ToPILImage()(row['pixel_values'])  # Convert tensor to PIL Image
        pred = classifier(img)  # return an array of dicts {'label', 'score'}
        y_pred.append(feed.label2id[pred[0]['label']]) # add first label as predicted label
        y_prob.append(pred[0]['score']) # add first score as predicted score
    return (np.array(y_pred), np.array(y_prob))


def get_pred_prob_true(ds, model, processor) -> tuple[np.array, np.array, np.array]:
    """Retun all predictions, probabilities and true values for DS."""
    y_pred, y_prob = get_pred_and_probablities(ds, model, processor)
    y_true = np.array([row['label'] for row in ds])
    return (y_pred, y_prob, y_true)


def save_predictions(save_dir: Path, ds, model, processor) -> tuple[np.array, np.array]:
    """Predict on all items in DS and store the result as a CSV.
    Return y_pred and y_true (numpy arrays)."""
    y_pred, y_prob = get_pred_and_probablities(ds, model, processor)
    # TODO Use get_pred_prob_true instead
    y_true = np.array([row['label'] for row in ds])
    df = pd.DataFrame([(true, pred, prob) for (true, pred, prob) in zip(y_true, y_pred, y_prob)], # type: ignore
                      columns=["true", "pred", "prob"])
    csv_file = save_dir / "predictions.csv"
    df.to_csv(f"{csv_file}", index=False, header=True)
    return (y_pred, y_true)


def load_predictions(save_dir: Path) -> tuple[np.array, np.array, np.array]:
    """Return the predictions stored in SAVE_DIR as np.arrays.

    Args:

        save_dir - Directory to look for the .csv file

    Returns:

        y_true, y_pred, y_prob - Numpy arrays with the true values, the predictions and
                                 the probabilities.
    """
    csv_file = Path(save_dir) / "predictions.csv"
    if not csv_file.exists():
        log.critical("CSV file not found: {csv_file}]")
        sys.exit(1)
    df = pd.read_csv(csv_file)
    return tuple([np.array(df[col]) for col in ['true', 'pred', 'prob']]) # type: ignore
