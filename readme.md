# HW1_094295

## Goal
Train classification model on hospital data trying to predict if a patient is about to develop sepsis.

## Description
In this project we trained 3 different models trying to achive best f1 score over the above task. 
<br>
#### The models are: 
<ul>
  <li> Random Forest  </li>
  <li> Adaboost </li>
  <li> Xgboost </li>
 </ul>
 This Reopsitory contains the following files:
 
<ul>

  * `data_exploration.py` - this holds the code of all data exploration we performed over the train data.To use it you should send it a csv holding all data merged together. To create such csv you can use the `create_data` function in  `preprocess.py`. 
  * `preprocess.py` - preprocessing of the data. 
  * **Model files** = [`Xgboost_model.py`, `random_forest_model.py`, `Adaboost_model.py`] - For each model there is a seperate python file containing all of the model's process -
    training, testing, hyperparameter tuning, and post analysis.<br> In the post analysis part we left only the code we eventually used and not all of our experiments. This code assumes training data is in a directory data/train .
  * `utils.py` - util functions used in all models.
  * All of the trained models which achieved best results - `adaboost_final_model.pkl` , `rf_final_model.pkl`, `xgboost_final_model.json`

 </ul>

## How to run?
#### Prepare envoriment
1. Clone this project
2. conda install the environment.yml file

### Reproduce results
#### predict.py
This will use our best pretrained model - `xgboost_final_model.json` to predict sepsis-label over new data.<br>
Should get as an argument path to the test data
