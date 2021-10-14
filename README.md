# data-science-project

This is a work-in-progress. 

At the current commit, you can load data. Specify `dataset_type=torch` for the default PyTorch class
for Omniglot (few-shot, 1623 instances) or `dataset_type=custom` for in-house dataset (100 way, psyphy annotations).

## Usage
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

```
python3 main.py dataset_type=torch task=svm to_numpy=True
```