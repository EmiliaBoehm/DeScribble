# Repo for Capstone Project


---

## Set up a Kanban board



### Optional: Add workflows

Workflows can help you keep your kanban board automatically on track. 

Select the project created in the steps above.  

Click on the 3 dots to the far right of the board (...)

Select workflow as the first option. 

Activate the ones you feel necessary to your project

Go back to your project repository (fraud detection))

## Requirements and Environment

Requirements:
- pyenv with Python: 3.11.3

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


