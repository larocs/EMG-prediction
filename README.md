# What is EMG-prediction?
EMG-prediction is a set of ML models used for predicting EMG signals.
You can predict 1-single point or many points in future, using reference EMG data as training dataset.
Change the setttings and selected model on configuration.json file. Models are implemented on build_model.py

## Table of Contents
  * [Installation](#installation)
  * [Usage](#usage)
  * [Results](#results)
  * [Datasets](#datasets)
  * [Paper](#paper)

# Installation
We strongly recommend the usage of Anaconda for managing python environments. 
This set-up was tested under Windows 10, Ubuntu and Raspbian.
```
$ conda create --name emg_prediction python=3.6
$ conda activate emg_prediction
$ git clone https://github.com/larocs/EMG-prediction
$ cd EMG-prediction/
$ pip install -r requirements.txt
$ jupyter notebook
	
```
	
# Usage
You can use the Jupyter Notebook "EMG Signal prediction" to do a step-by-step running of the model. Or simply run the "train.py" to run the training and test of the prediction model.

You can change the way to load dataset and train inside the notebook. 

You can simply run the "predict.py" file with pre-trained models already available inside ./saved_models to generate a batch of predictions based on input data inside ./data folder.

The model is currently configured for receiving as input a 400-point window of floating-point values, and as output the next N-points according to the selected configuration.

You can change the configurations for the prediction model on configuration.json file.

```
python generate.py
python train.py

```

# Datasets
This work used a private EMG dataset from Parkinson's Disease patients.

You can use your own dataset, or use existing available datasets. One recommended dataset is NinaPro.
NinaPro is an open-source dataset that can be downloaded on: http://ninapro.hevs.ch/

# Paper
R. A. Zanini, E. L. Colombini and M. C. F. de Castro, "Parkinson’s Disease EMG Signal Prediction Using Neural Networks," 2019 IEEE International Conference on Systems, Man and Cybernetics (SMC), Bari, Italy, 2019, pp. 2446-2453, doi: 10.1109/SMC.2019.8914553.

```
@inproceedings{DBLP:conf/smc/ZaniniCC19,
  author    = {Rafael Anicet Zanini and
               Esther Luna Colombini and
               Maria Cl{\'{a}}udia Ferrari de Castro},
  title     = {Parkinson's Disease {EMG} Signal Prediction Using Neural Networks},
  booktitle = {2019 {IEEE} International Conference on Systems, Man and Cybernetics,
               {SMC} 2019, Bari, Italy, October 6-9, 2019},
  pages     = {2446--2453},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/SMC.2019.8914553},
  doi       = {10.1109/SMC.2019.8914553},
  timestamp = {Sat, 07 Dec 2019 20:27:21 +0100},
  biburl    = {https://dblp.org/rec/conf/smc/ZaniniCC19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
&nbsp;


# Interested in contributing to EMG-prediction?
Thanks for the interest and please read the [Contributing](https://github.com/larocs/EMG-prediction/blob/master/CONTRIBUTING.md) recommendations.

# Authors
Esther Luna Colombini & Rafael Anicet Zanini