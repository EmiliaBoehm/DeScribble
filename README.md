# DeScribble

Use Machine Learning (ML) on handwritten text to draw inferences about
the gender of the person who wrote it.

---

## Models used

The project uses a [ViT Image
Classifier](https://huggingface.co/transformers/v4.12.5/model_doc/vit.html#vitforimageclassification)
Transformer model for predictions.  It performs better than a standard
Random Forest Image Classifier (baseline model).

## Requirements and Environment

### DVC

This project uses [DVC](https://dvc.org/). It is included as a
requirement in the `requirements.txt` file.

### Python Environment

For installing the virtual environment you can install it manually with the following commands: 

For Linux / Mac OS:

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows, please use the following commands instead:

```
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage 

### Overview

 - Put samples in `images/bw`
 - Run the script `segmenter.py` to cut out words.
 - Create a table `index_raw.csv` with the columns `Geschlecht` and
   `Datei Index`
 - Train the ViT Model with the script `vit_train.py`

### Using Data Versioning with DCC

To download the relevant data using DVC, run the following command (Linux / Mac OS / Windows):

```
dvc pull
```



