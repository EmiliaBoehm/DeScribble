"""Load the pretrained TrOCR model from huggingface.com and run it with an example image."""

import argparse
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from typing import Any

MODEL_PATH = "../models"


def get_image(imagepath: str) -> Image:
    """Load image from PATH."""
    return Image.open(imagepath).convert("RGB")


def get_processor(huggingface_checkpoint: str, model_subdir: str) -> TrOCRProcessor:
    """Download or import TrOCRProcessor."""
    # TODO use Path (via from pathlib import Path)
    model_dir = MODEL_PATH + "/" + model_subdir
    processor = None
    if os.path.isdir(model_dir):
        processor = TrOCRProcessor.from_pretrained(model_dir)  # local_files_only=
    else:
        processor = TrOCRProcessor.from_pretrained(huggingface_checkpoint)
        processor.save(model_dir)
    return processor


def format_image(image: Image, processor: TrOCRProcessor) -> torch.Tensor:
    """Format image for the TrOCR model."""
    return processor(image, return_tensors='pt').pixel_values


def get_model(huggingface_checkpoint: str, model_subdir: str) -> VisionEncoderDecoderModel:
    """Download or import the TrOCR model."""
    model_dir = MODEL_PATH + "/" + model_subdir
    model = None
    if os.path.isdir(model_dir):
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)  # local_files_only=
    else:
        model = VisionEncoderDecoderModel.from_pretrained(huggingface_checkpoint)
        model.save(model_dir)
    return model


def predict(image_tensor: torch.Tensor,
            model: VisionEncoderDecoderModel,
            processor: TrOCRProcessor) -> Any:
    """Generate a prediction for IMAGE_TENSOR based on MODEL."""
    ids = model.generate(image_tensor)
    text = processor.batch_decode(ids, skip_special_tokens=True)
    return text


def get_ArgumentParser() -> argparse.ArgumentParser:
    """Return an ArgumentParser object for this script."""
    parser = argparse.ArgumentParser(
        description="Scan the image and print the results"
    )
    parser.add_argument("inputfile")
#parser.add_argument("outputfile")
    return parser


def main() -> None:
    """Parse arguments and dispatch."""
    args = get_ArgumentParser().parse_args()
    image = get_image(args.inputfile)
    model = get_model("microsoft/trocr-base-handwritten", "simplemodel")
    processor = get_processor("microsoft/trocr-base-handwritten", "simplemodel")
    tensor = format_image(image, processor)
    print(predict(tensor, model, processor))


if __name__ == "__main__":
    main()
