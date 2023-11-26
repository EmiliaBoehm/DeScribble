"""Train the Huggingface pretrained ViT Image Classifier, save its state and  predictions."""
import logging
import sys
import argparse
from pathlib import Path
import trainlib as lib
from datafeed import CHECKPOINT_VIT_ONLINE

log = logging.getLogger("vit_train")
log.setLevel(logging.INFO)


def main(save_dir: Path, batch_size: int, epochs: int) -> None:
    if save_dir.exists():
        log.critical(f"Directory {save_dir} already exists, training would overwrite an existing checkpoint.")
        log.critical("Delete the directory manually if you want to use it for a new training round.")
        sys.exit(1)

    save_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Creating directory {save_dir} to store trained model")
    log.info("Splitting data...")
    ds = lib.load_data_and_split()

    log.info("Initializing model and processor...")
    model, processor = lib.get_model_and_processor(CHECKPOINT_VIT_ONLINE)

    log.info(f"Training model (batch size {batch_size}, epochs {epochs})")
    trainer, trainer_result = lib.train_model(save_dir, model, processor, ds,
                                              batch_size, epochs)

    log.info("Storing the model...")
    lib.save_model(save_dir, trainer, trainer_result)

    log.info("Storing predictions on the validation data...")
    lib.save_predictions(save_dir, ds['validation'], model, processor)

    # log.info("Storing complete Dataset with all splits....")
    # ds.save_to_disk(save_dir)
    log.info(f"All done! Best model is stored in {save_dir}/model")


def get_ArgumentParser() -> argparse.ArgumentParser:
    """Return an ArgumentParser object for this script."""
    parser = argparse.ArgumentParser(
        description="Load a ViTModel from huggingface, train it and store the results."
    )
    parser.add_argument("save_dir", type=Path,
                        help="Name of the directory to store the trained model in")
    parser.add_argument("batch_size", type=int,
                        help="Batch size (e.g. 16, 32) for training")
    parser.add_argument("epochs", type=int,
                        help="Number of epochs")
    return parser


if __name__ == '__main__':
    args = get_ArgumentParser().parse_args()
    main(args.save_dir, args.batch_size, args.epochs)
