import pandas as pd
import os
from tqdm import tqdm
import sys
import pickle


def create_data(dir_path):
    patient_files = os.listdir(dir_path)
    first_patient = os.path.join(dir_path,patient_files[0])
    final_df = create_patient_df(first_patient)
    for patient in tqdm(patient_files[1:]):
        patient_file = os.path.join(dir_path,patient)
        p_df = create_patient_df(patient_file)
        final_df = pd.concat([final_df,p_df])

    return final_df

def create_patient_df(psv_file):
    df = pd.read_csv(psv_file, delimiter="|")
    patient_id = psv_file.split("patient_")[-1].strip(".psv")
    df["patient_id"] = [patient_id] * len(df)
    df["match"] = df.SepsisLabel != df.SepsisLabel.shift()
    if len(df[df["match"]]) == 2:
        shift_index = df[df["match"]].index[-1]
        df = df[df.index <= shift_index]
    elif df["SepsisLabel"][0] ==1:
        df = df[df.index <= 0]


    df.drop(columns=["match"], inplace=True)
    return df


def preprocess_test(df, keep_cols: list,fill_null_vals:dict):

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

def last_n_rows(group, n=10):
    return group.tail(n)

def aggregate_df_reg_mean(stat_cols, dynamic_cols, initial_df):

    grouped_by_patient_df = initial_df.groupby("patient_id")
    aggregated_df_stat = grouped_by_patient_df[stat_cols].max()
    aggregated_df_dynamic = grouped_by_patient_df[dynamic_cols].mean()
    final_aggregated_df = aggregated_df_stat.join(aggregated_df_dynamic, on="patient_id")

    return final_aggregated_df


def run_xgboost(model, test_df, scaler,stat_cols):
    test_df = test_df.groupby('patient_id').apply(last_n_rows, n=10).reset_index(drop=True)
    scaling_cols = list(set(test_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))
    aggregated_test = aggregate_df_reg_mean(stat_cols, scaling_cols, test_df)
    scaled_df = aggregated_test.copy()
    scaled_df[scaling_cols] = scaler.transform(scaled_df[scaling_cols])
    X_test = scaled_df.drop("SepsisLabel", axis=1)
    y_test = scaled_df.SepsisLabel
    predicted = model.predict(X_test)


if __name__ == '__main__':

    test_dir = sys.argv[1]
    test_df = create_data(test_dir)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('backup/null_vals.pkl', 'rb') as f:
        null_vals_dict = pickle.load(f)

    with open('backup/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    keep_cols = "HR,O2Sat,Temp,SBP,MAP,Resp,BUN,Calcium,Creatinine,Glucose,Magnesium,Hct,Hgb,WBC,Age,Gender,ICULOS,SepsisLabel,patient_id,age_group".split(",")

    test_df = preprocess_test(test_df,keep_cols=keep_cols,fill_null_vals=null_vals_dict)
