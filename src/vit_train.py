"""Train the Huggingface pretrained ViT Image Classifier, save its state and  predictions."""
from typing_extensions import Annotated
import typer
import logging
import sys
from pathlib import Path
import trainlib as lib     # type: ignore
from datafeed import CHECKPOINT_VIT_ONLINE # type: ignore

logging.basicConfig()
log = logging.getLogger("vit_train")
log.setLevel(logging.INFO)


def main(save_dir: Annotated[Path,
                             typer.Argument(help="Name of the directory to store the trained model in")],
         batch_size: Annotated[int,
                               typer.Argument(help="Batch size (e.g. 16, 32) for training")],
         epochs: Annotated[int,
                           typer.Argument(help="Number of epochs")],
         verbose: Annotated[bool, typer.Option("--verbose", "-v",
                                               help="Be a bit more verbose about what we are doing.")] = True) -> None:
    """Load a pretrained ViT Transformer and train it with the segments defined in params.yaml.
    After training, store the model in SAVE_DIR, which is also used for storing the local checkpoints.
    Also save the training state, metrics and the predictions on the validation data set."""
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

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


if __name__ == '__main__':
    typer.run(main)
