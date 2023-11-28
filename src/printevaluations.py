"""Print some evalutions for a model."""
import logging
import sys
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

import trainlib as lib
import datafeed as feed

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



def get_ArgumentParser() -> argparse.ArgumentParser:
    """Return an ArgumentParser object for this script."""
    parser = argparse.ArgumentParser(
        description="Load the predictions from a csv file and print out some metrics. ."
    )
    parser.add_argument("save_dir", type=Path,
                        help="Name of the directory to find the csv file in")
    return parser


if __name__ == '__main__':
    args = get_ArgumentParser().parse_args()
    y_true, y_pred, y_prob = lib.load_predictions(args.save_dir)
    print(confusion_matrix(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    
