# DeScribble

Use Machine Learning (ML) on handwritten text to draw inferences about
the person who wrote it.

---

## Model used

## Requirements and Environment

### DVC

This project uses [DVC](https://dvc.org/). It is included as a
requirement in the `requirements.txt` file.

### Python Environment // UP TO DATE

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

## Usage // TO UPDATE

To download the relevant data using DVC, run the following command (Linux / Mac OS / Windows):

```
dvc pull
```
Further details on DVC can be found in dvc.md file


## Limitations // TO UPDATE

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


