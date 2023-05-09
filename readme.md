# HW1_094295

## Goal
Train classification model on hospital data trying to predict if a patient is about to develop sepsis.

## Description
In this project we trained 3 different models trying to achive best f1 score over the above task. 
The models are
<ul>
  <li> Random Forest  </li>
  <li> Adaboost </li>
  <li> Xgboost </li>
 </ul>
 This Reopsitory contains the following files:
 <ul>
  - data_exploration.py - this holds the code of all data exploration we performed over the data 
  <li> preprocess.py** - preprocessing of the data </li>
  <li> **Model files** = [`Xgboost_model.py`, `random_forest_model.py`, `Adaboost_model.py`] - For each model there is a seperate python file containing all of the model's process -
    training, testing, hyperparameter tuning, and post analysis.<br> In the post analysis part we left only the code we eventually used and not all of our experiments.
  </li>
  <li> **utils.py** - useful functions used in all models </li>
  <li> All of the trained models which achieved best results - `adaboost_final_model.pkl` `rf_final_model.pkl` `xgboost_final_model.json` </li>

 </ul>

## How to run?
#### Prepare envoriment
1. Clone this project
2. conda install the environment.yml file

### Reproduce results
#### predict.py
This will use our best pretrained model - `xgboost_final_model.json` to predict sepsislabel over new data.
Should get as an argument path to the test data
