import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_corr_mat(df):
    correlation= df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    f, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlation, mask=mask, cmap="RdBu", vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("correlation_mat.png")
    plt.show()

def perform_data_exploration(csv_name):
    final_df = pd.read_csv(csv_name)
    # get to know the data
    final_df.describe(include="all").T.to_csv("data_describe.csv")
    create_corr_mat(final_df)
    # count number of patients with sepsis
    num_of_sepsis_patient = len(final_df[final_df["SepsisLabel"] == 1])
    group_by_patient_df = final_df.groupby(by="patient_id").count()
    print(f"Total number of patients with sepsis: {num_of_sepsis_patient}/20000")

    # count total number of null values in each column
    print(final_df.isnull().sum())

    # count number of patients with at least one value which is not null per column
    for col in final_df.columns:
        print(col, final_df[~final_df[col].isna()]["patient_id"].nunique())

    # keep only columns of which at least 17000 (85%) patients have at least one valid value in them
    relevant_cols = [col for col in final_df.columns if final_df[~final_df[col].isna()]["patient_id"].nunique() > 17000]
    print("\n".join(relevant_cols))

    # dataframe of only patients with sepsis
    Sepsis_df = final_df[final_df.SepsisLabel == 1]

    # find gender distribution
    print("Gender distribution:")
    print(final_df.groupby(by="Gender").patient_id.nunique())

    # find gender distribution in sepsis population
    print()
    print("Gender distribution within sepsis patients:")
    print(Sepsis_df.groupby(by="Gender").patient_id.nunique())

    # group by patient the dataframe to perform analysis of static features
    # we take the max because static feartures remain the same, and it will seperate patients with label=1 from
    # patients with label =0
    grouped_df = final_df.groupby("patient_id").max()

    # find Age range:
    print()
    print(f"Min Age total population: {grouped_df.Age.min()}")
    print(f"Max Age total population: {grouped_df.Age.max()}")
    print(f"Min Age sepsis population: {grouped_df[grouped_df.SepsisLabel == 1].Age.min()}")
    print(f"Max Age sepsis population: {grouped_df[grouped_df.SepsisLabel == 1].Age.max()}")

    # Age by gender histogram
    sns.histplot(data=grouped_df, x="Age", hue=grouped_df.Gender, common_norm=False, bins=60, kde=True).set(
        title='Age distribution by gender')
    plt.savefig("Age_by_gender_total.png")
    plt.show()

    sns.histplot(data=grouped_df[grouped_df.SepsisLabel == 1], x="Age",
                 hue=grouped_df[grouped_df.SepsisLabel == 1].Gender, common_norm=False, bins=60, kde=True).set(
        title='Age distribution by gender in sepsis population')
    plt.savefig("Age_by_gender_sepsis.png")
    plt.show()

    # ICULOS Histogram + density
    # Max value == total time in ICU until 6 hours before sepsis was found
    sns.histplot(data=grouped_df, x="ICULOS", hue=grouped_df.SepsisLabel, stat="density", common_norm=False, bins=60,
                 kde=True).set(title='Total time in ICU by label')
    plt.savefig("ICULOS_hist.png")
    plt.show()

    sns.histplot(data=grouped_df, x="HospAdmTime", hue=grouped_df.SepsisLabel, stat="density", common_norm=False,
                 bins=60, kde=True).set(title="HospAdmTime Histogram by Sepsis label")
    plt.savefig("HospAdmTime_hist.png")
    plt.show()

    # exploration of vital signs features
    vital_signs = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]

    plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=.5)
    for i, column in enumerate(vital_signs, 1):
        plt.subplot(4, 2, i)
        sns.histplot(data=final_df, x=column, hue=final_df.SepsisLabel, stat="density", common_norm=False, bins=60,
                     kde=True).set(title=f'{column} distribution by sepsis label')
    plt.savefig("Vital_signs.png")
    plt.show()

    # exploration of lab values features
    lab_values = ['BUN', 'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Calcium']

    plt.figure(figsize=(18, 42))
    plt.subplots_adjust(hspace=.5)
    for i, column in enumerate(lab_values, 1):
        plt.subplot(13, 2, i)
        sns.histplot(data=final_df, x=column, hue=final_df.SepsisLabel, stat="density", bins=60, common_norm=False,
                     kde=True).set(title=f'{column} distribution by sepsis label')

    plt.savefig("lab_values.png")
    plt.show()

if __name__ == '__main__':

    perform_data_exploration("all_data_merged_final.csv")