import pandas as pd
import os
from tqdm import tqdm

def create_patient_df(psv_file):
    """
    creates a pandas df from one psv file, deleting irrelevant rows
    :param psv_file: path of psv file to parse
    :return: dataframe fromt he psv without irrelevant rows
    """
    df = pd.read_csv(psv_file, delimiter="|")
    patient_id = os.path.basename(psv_file).rstrip(".psv")
    df["patient_id"] = [patient_id] * len(df)
    df["match"] = df.SepsisLabel != df.SepsisLabel.shift()
    if len(df[df["match"]]) == 2:
        shift_index = df[df["match"]].index[-1]
        df = df[df.index <= shift_index]
    elif df["SepsisLabel"][0] ==1:
        df = df[df.index <= 0]

    df.drop(columns=["match"], inplace=True)
    return df

def create_data(dir_path,csv_name):
    """
    merging all psv files into one large dataframe and saves to csv file if given
    :param dir_path: path to data directory
    :param csv_name: name of the csv to save. if an empty string will not save to csv file
    :return:  the merged dataframe
    """
    print("reading data\n")
    patient_files = os.listdir(dir_path)
    first_patient = os.path.join(dir_path,patient_files[0])
    final_df = create_patient_df(first_patient)
    for patient in tqdm(patient_files[1:]):
        patient_file = os.path.join(dir_path,patient)
        p_df = create_patient_df(patient_file)
        final_df = pd.concat([final_df,p_df])

    if csv_name:
        final_df.to_csv(csv_name, index=False)
    return final_df

def preprocess_train_data(csv_file, keep_cols: list):
    df = pd.read_csv(csv_file)

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
    df = imputed_df.copy()
    df.drop("patient_id", axis=1,inplace=True)
    grouped_df = df.groupby(["age_group", "Gender"])
    cols = list(set(df.columns).difference({"age_group", "Gender", "SepsisLabel","patient_id"}))
    df[cols] = grouped_df[cols].transform(lambda x: x.fillna(x.mean()))
    df = df.reset_index().drop("level_1",axis=1)

    return df , grouped_df.mean()

def preprocess_test(csv_file, keep_cols: list,fill_null_vals:dict):
    df = pd.read_csv(csv_file)

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
