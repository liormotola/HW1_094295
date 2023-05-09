import re
import pandas as pd
import sys
import pickle
import utils
import preprocess
from xgboost import XGBClassifier

def preprocess_comp(df, keep_cols: list,fill_null_vals:dict):
    """
    preforms preprocessing to the data - dropping columns, filling nulls
    :param df: pandas dataframe holding all data
    :param keep_cols: columns to keep
    :param fill_null_vals: dictionary with the values to use to fill nulls in each column
    :return: processed df
    """

    # dropping irrelevant cols
    drop_cols = list(set(df.columns).difference(set(keep_cols)))
    dropped_df = df.drop(columns=drop_cols)
    dropped_df['age_group'] = pd.cut(dropped_df['Age'], bins=[0, 18, 30, 50, 70, 90, 100], labels=[0, 1, 2, 3, 4, 5])

    # imputing null values
    imputed_df = dropped_df.copy()
    patients_grouped_df = imputed_df.groupby('patient_id')
    # ffil= forward fill = last observation carried forward
    # bfill = backward fill = if there are no previous records so we will take from later records
    imputed_df= patients_grouped_df.apply(lambda row: row.ffill().bfill())

    for index, values in fill_null_vals.items():
        age, gender = index
        imputed_df[(imputed_df.age_group == age) & (imputed_df.Gender == gender)] = imputed_df[
            (imputed_df.age_group == age) & (imputed_df.Gender == gender)].fillna(value=values)
    imputed_df = imputed_df.drop("patient_id", axis=1).reset_index().drop("level_1", axis=1)

    return imputed_df


def run_xgboost(model, test_df, scaler,stat_cols,sepsis_mode):
    """
    runs a pretrained xgboost model on test data and creates a prediction csv
    :param model: path to json file holding data about pre trained xgboost model
    :param test_df: dataframe holding preprocessed test data
    :param scaler: fit scaling object
    :param stat_cols: names of static columns
    :param sepsis_mode: boolean - sepsislabel col is in data or not
    """
    xgb = XGBClassifier()
    xgb.load_model(model)

    test_df = test_df.groupby('patient_id').apply(utils.last_n_rows, n=10).reset_index(drop=True)
    scaling_cols = list(set(test_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))
    aggregated_test = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, test_df)
    scaled_df = aggregated_test.copy()
    scaled_df["ICULOS_scaled"] = scaled_df["ICULOS"]
    scaled_df[scaler.feature_names_in_] = scaler.transform(scaled_df[scaler.feature_names_in_])
    if sepsis_mode:
        X_test = scaled_df.drop(["SepsisLabel","ICULOS"], axis=1)
    else:
        X_test = scaled_df.drop( "ICULOS", axis=1)

    cols= xgb.get_booster().feature_names
    predicted = xgb.predict(X_test[cols])
    scaled_df["prediction"] = predicted
    prediction_df = scaled_df[["patient_id","prediction" ]]
    prediction_df.rename(columns={"patient_id": "id"},inplace=True)
    prediction_df["id_num"] = prediction_df["id"].apply(lambda x: int(re.findall(r"\d+",x)[-1]))
    prediction_df.sort_values(by= "id_num",inplace=True)
    prediction_df.drop("id_num",axis=1,inplace=True)
    prediction_df.to_csv("prediction.csv",index=False)


if __name__ == '__main__':

    test_dir = sys.argv[1]
    test_df = preprocess.create_data(test_dir,"")
    sep_mode = "SepsisLabel" in test_df.columns
    # loading pre trained scaler and null values
    with open('null_vals_imputation.pkl', 'rb') as f:
        null_vals_dict = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    keep_cols = "HR,O2Sat,Temp,SBP,MAP,Resp,BUN,Calcium,Creatinine,Glucose,Magnesium,Hct,Hgb,WBC,Age,Gender,ICULOS,SepsisLabel,patient_id,age_group".split(",")

    stat_cols = ["age_group", "Gender", "ICULOS", "SepsisLabel"]
    if not sep_mode:
        keep_cols.remove("SepsisLabel")
        stat_cols.remove("SepsisLabel")

    #preprocessing
    test_df = preprocess_comp(test_df,keep_cols=keep_cols,fill_null_vals=null_vals_dict)
    #run model
    run_xgboost("xgboost_final_model.json",test_df=test_df,scaler=scaler,stat_cols=stat_cols,sepsis_mode=sep_mode)