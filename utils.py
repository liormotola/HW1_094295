from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def weighted_avrage(group):
    n = len(group) + 1
    return group[:-1].sum() / n + 2 * group.tail(1) / n


def aggregate_df_weighted_mean(stat_cols, dynamic_cols, initial_df):
    grouped_by_patient_df = initial_df.groupby("patient_id")
    aggregated_df_stat = grouped_by_patient_df[stat_cols].max()
    aggregated_df_dynamic = grouped_by_patient_df[dynamic_cols].apply(weighted_avrage).reset_index(level=1)
    final_aggregated_df = aggregated_df_stat.join(aggregated_df_dynamic, on="patient_id")

    return final_aggregated_df

def last_n_rows(group, n=10):
    return group.tail(n)


def aggregate_df_reg_mean(stat_cols, dynamic_cols, initial_df):

    grouped_by_patient_df = initial_df.groupby("patient_id")
    aggregated_df_stat = grouped_by_patient_df[stat_cols].max()
    aggregated_df_dynamic = grouped_by_patient_df[dynamic_cols].mean()
    final_aggregated_df = aggregated_df_stat.join(aggregated_df_dynamic, on="patient_id")

    return final_aggregated_df.reset_index()

def model_eval(true_labels, predicted_labels,set_type : str,title:str):
    acc= accuracy_score(true_labels, predicted_labels)
    print(f"{set_type} Accuracy: ", acc)
    precision = precision_score(true_labels, predicted_labels)
    print(f"{set_type} Precision: ", precision)
    recall = recall_score(true_labels, predicted_labels)
    print(f"{set_type} Recall: ", recall)
    f1 = f1_score(true_labels, predicted_labels)
    print(f"{set_type} F1 Score: ", f1)
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))
    print(f"{set_type} RMSE: ", rmse)
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title(f"{title}-{set_type}")
    plt.savefig(f"{title} {set_type}-confusion matrix")
    plt.show()

def ICULOS_analysis(predictions_df):
    levels = [(0, 30), (30, 55), (55, 400)]
    for i in range(len(levels)):
        df1 = predictions_df[(predictions_df.ICULOS >= levels[i][0]) & (predictions_df.ICULOS < levels[i][1])]
        print(f"level {levels[i]}")
        model_eval(df1.SepsisLabel, df1.predicted, f"test_level_{levels[i]}", title="ICULOS")

def create_features_importance_plot(model,title):
    data = {"features": model.feature_names_in_, "importance": model.feature_importances_}
    features_df = pd.DataFrame(data)
    features_df = features_df.sort_values(by="importance", ascending=False)
    features_df = features_df.reset_index()
    fig, ax = plt.subplots()
    ax.barh(features_df.features, features_df.importance)
    ax.set_xlabel("importance")
    for i in range(len(features_df.importance)):
        plt.text(features_df.importance[i] + 0.01, features_df.index[i], str(round(features_df.importance[i], 3)),
                 ha='left', va='center')

    ax.set_title(title)
    ax.invert_yaxis()
    plt.savefig(title)
    plt.show()

def HR_analysis(predicted_df):
    levels = [(-4, -1), (-1, -0.1), (-0.1, 0.1), (0.1, 1), (1, 4)]
    for i in range(len(levels)):
        df1 = predicted_df[(predicted_df.HR >= levels[i][0]) & (predicted_df.HR < levels[i][1])]
        print(f"HR level {levels[i]}")
        print(f1_score(df1.SepsisLabel, df1.predicted))

def Temp_analysis(predicted_df):
    levels = [(-4, -1), (-1, -0.1), (-0.1, 0.1), (0.1, 1), (1, 4)]
    for i in range(len(levels)):
        df1 = predicted_df[(predicted_df.Temp >= levels[i][0]) & (predicted_df.Temp < levels[i][1])]
        print(f"Temp level {levels[i]}")
        print(f1_score(df1.SepsisLabel, df1.predicted))