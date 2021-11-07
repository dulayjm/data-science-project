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

## Usage on the Custom Dataset

The dataset is the `OmniglotReactionTimeDataset` class that appears in some of the files in this project. For this experiment, you will need: 

- `omniglot_realfake` psychophysically annotated data points.
- `sigma_dataset.csv`
- `OmniglotReactionTimeDataset` class

The first subfolders are all of the raw images that are needed for the usage of the class. The `real` subfolder is a subset of 100 classes from the full Omniglot Dataset. THe `fake` folder are DCGAN generated approximations of the each of the same classes from the first folder. The generative images were used as a form of data augmentation to increase intraclass variance exposure to human subjects on the psychophysical experiments in the past. The data loader will load images from both. 

The csv file is simply a reference structure of the data folder to load more easily. Each consists of the two paired images used in a given task, as well as the reaction time on the task and mean accuracy per the real label. 

The first class is the dataset class, subclassed from the Pytorch `Dataset` class. The `__getitem__` function is the most important one. When called, it return a dictionary like: 
```       
sample = {'label1': label1, 'label2': label2, 'image1': image1,
                    'image2': image2, 'rt': rt, 'acc': sigma} 
```
where the labels are the labels of the two respective images, images are torch tensor representations of the imagse, rt is the associated psychophysical reaction time with the images, and sigma is the blurring parameter used for the standard sklearn Gaussian blur. The method also has some commented out parts where you can mess around with blurring one of the images.