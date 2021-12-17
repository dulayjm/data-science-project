# Data Science Project: Classification and Optical Character Recognition on Multi-alphabetic Handwriting Data

**Authors**: Justin Dulay and Stephen Bothwell

## General Usage

First, be sure to create a virtual environment and obtain the appropriate libraries.
For example, perform the following:

```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

After the above, each of the individual files in the classifiers folder can be consulted for its usage.
All are fitted with command line interfaces. 
However, they are not necessarily set up with every possible option able to be changed.
Some experimentation was done on a more manual and incremental basis that preceded the addition of command line interfaces.
As a result, the use of datasets in a given context may need slight edits such that 
they can be probed, examined, and tested.

To use the classifiers as intended, place the data files in the `classifiers` directory.
That is, all of the following should be there:
* `omniglot-py`, which contains all the original Omniglot data.
* `omniglot-realfake`, which has all the in-house data based on the Omniglot dataset. It is a subset of the Omniglot data.
* `sigma_dataset.csv`, which lists information about `omniglot-realfake` with regard to psychophysically-based research. 
It is used in the process of loading the `omniglot-realfake` data.

The five classifier files are as follows:
* `random_supervised.py`, which contains the random baseline.
* `nearest_neighbor.py`, which contains the k-Nearest Neighbor classifier.
* `random_forest.py`, which contains the Random Forest classifier.
* `siamese.py`, which contains a Siamese neural network classifier.
* `prototypical.py`, which contains a prototypical neural network classifier.

Other relevant files include:
* `data/data_preprocessors.py`, which contains functions that aid in getting data from datasets into the right format.
* `data/dataset.py`, which contains the OmniglotReactionTimeDataset class (see below for more details).
* `data/full_omniglot.py`, which contains a class that relates to the complete Omniglot dataset. 
It was created because the PyTorch versions of the Omniglot dataset only allowed for the retrieval of partial sets.
* `helpers/fold_splitter.py`, which contains a function that facilitates in the creation of random folds 
for k-fold cross-validation.
* `helpers/psychloss.py`, which contains functions can be used in neural networks (or elsewhere)
which especially pertain to psychophysically-based approaches.
* `helpers/statistical_functions.py`, which contains various stat-calculating and stat-displaying functions
as utilities for other parts of the project.
* `helpers/stratified_sampler.py`, which contains a class, `StratifiedKFoldSampler`,
that implements stratified sampling for k-fold cross validation.
The class also attempts to facilitate k-fold cross-validation.
* `other`, which is a directory containing code and data that was largely unused or 
not finished enough to be considered as part of the final submission for the project.

## Usage of the Custom Dataset

This section concerns the `OmniglotReactionTimeDataset` class that appears in some of the files in this project. 
For experiments involving it, you will need: 

- `omniglot_realfake`, a directory containing psychophysically-annotated data points.
- `sigma_dataset.csv`, a comma-separated values file which lists information 
about the `omniglot_realfake` data.
- `OmniglotReactionTimeDataset`, a class which allows for the retrieval and usage 
of the `omniglot_realfake` data.

The first subfolders are all the raw images that are needed for the usage of the class. The `real` subfolder is a subset of 100 classes from the full Omniglot Dataset. The `fake` folder are DCGAN-generated approximations of each of the same classes from the first folder. The generative images were used as a form of data augmentation to increase intraclass variance exposure to human subjects on the psychophysical experiments in the past. The data loader will load images from both. 

The csv file is simply a reference structure of the data folder to load more easily. Each consists of the two paired images used in a given task, as well as the reaction time on the task and mean accuracy per the real label. 

The first class is the dataset class, subclassed from the Pytorch `Dataset` class. The `__getitem__` function is the most important one. When called, it returns a dictionary like: 
```       
sample = {'label1': label1, 'label2': label2, 'image1': image1,
                    'image2': image2, 'rt': rt, 'acc': sigma} 
```
where the labels are the labels of the two respective images, images are torch tensor representations of the images, `rt` is the associated psychophysical reaction time with the images, and `sigma` is the blurring parameter used for the standard `sklearn` Gaussian blur.

## Usage on Neural Network Classifiers

- Navigate to the `classifiers` folder
- run `python3 <classifier_name>` with the appropriate command line arguments.
  - NOTE: you can adjust the hyperparameters in the run script
  - NOTE: you can run this on a GPU - either on a local client, the CRC, or Google Colab

#### To test with saved models:

- call `torch.load(<model_name.mod>)` on the specified model