from preprocess import preprocess_train_data , preprocess_test,create_data
import os
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import f1_score

def rf_reg_mean(train_df, test_df , stat_cols, n=0):
    """
    train+test random forest model with regular mean aggregation
    :param train_df: train data
    :param test_df: test data
    :param stat_cols: static columns - not to be aggregated by mean
    :param n: number of rows to take into account in aggregation
    :return: trained model + test df including predictions
    """

    if n > 0 :
        train_df = train_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)
        test_df = test_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)

    scaling_cols = list(set(train_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))

    aggregated_train = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, train_df)
    aggregated_test = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, test_df)

    scaler = StandardScaler()
    scaled_df = aggregated_train.copy()
    scaled_df["ICULOS_scaled"] = scaled_df.ICULOS
    scaling_cols += ["ICULOS_scaled"]
    scaled_df[scaling_cols] = scaler.fit_transform(scaled_df[scaling_cols])

    scaled_df_test = aggregated_test.copy()
    scaled_df_test["ICULOS_scaled"] = scaled_df_test.ICULOS
    scaled_df_test[scaling_cols] = scaler.transform(scaled_df_test[scaling_cols])

    X_train = scaled_df.drop(["SepsisLabel","ICULOS","patient_id"], axis=1)
    y_train = scaled_df.SepsisLabel
    X_test = scaled_df_test.drop(["SepsisLabel","ICULOS","patient_id"], axis=1)
    y_test = scaled_df_test.SepsisLabel

    rf = RandomForestClassifier(n_estimators=170, verbose=1,class_weight="balanced", max_depth=7)
    rf.fit(X_train, y_train)

    predicted_train = rf.predict(X_train)
    predicted_test = rf.predict(X_test)

    utils.model_eval(y_train, predicted_train, "train",title="random forest")
    print("\n")
    utils.model_eval(y_test, predicted_test, "test",title="random forest")
    scaled_df_test["predicted"] = predicted_test

    return rf , scaled_df_test

def rf_reg_mean_parameter_tuning(train_df , stat_cols, n=0):
    """
    Tuning parameters for random forest model with mean aggregation over n last rows
    :param train_df: train data
    :param stat_cols: static columns - not to be aggregated by mean
    :param n: number of rows to take into account
    :return: prints the best parameters found
    """

    if n > 0 :
        train_df = train_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)

    scaling_cols = list(set(train_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))

    aggregated_train = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, train_df)
    scaler = StandardScaler()

    scaled_df = aggregated_train.copy()
    scaling_cols += ["ICULOS"]
    scaled_df[scaling_cols] = scaler.fit_transform(scaled_df[scaling_cols])
    X_train = scaled_df.drop(["SepsisLabel","patient_id"], axis=1)
    y_train = scaled_df.SepsisLabel

    print("starting searching")
    optimizer = GridSearchCV(RandomForestClassifier(), {
        'n_estimators': [100,150, 170, 200],
        'max_depth': [3, 5, 7],
         'max_features':['sqrt', 'log2'],
        "class_weight":["balanced",None]
    }, scoring="f1", cv=3)

    optimizer.fit(X_train, y_train)
    print('Optimizing complete')
    print(optimizer.best_params_)
    print(optimizer.best_score_)
    df_optimizer = pd.DataFrame(optimizer.cv_results_).dropna()
    df_optimizer = df_optimizer[
        ['param_max_depth', 'param_n_estimators','param_max_features', 'param_class_weight','mean_test_score']]
    df_optimizer.to_csv("params_rf.csv")


def post_analysis(model, predicted_df):
    """
    runs post analysis on the model
    :param model: path to pre trained model
    :param predicted_df: dataframe of all data + predictions

    """
    with open(model, 'rb') as f:
        rf = pickle.load(f)
    # check feature importance
    utils.create_features_importance_plot(model=rf, title='Random Forest model - Features importance')

    # ICULOS analysis
    utils.ICULOS_analysis(predictions_df=predicted_df)

    # check gender subgroups performance:
    for i in range(2):
        df1 = predicted_df[predicted_df.Gender == i]
        print(f"Gender {i}")
        print(f1_score(df1.SepsisLabel, df1.predicted))

    # check Temp subgroups performance:
    utils.Temp_analysis(predicted_df=predicted_df)

def main():
    if not os.path.isfile("all_data_merged_final.csv"):
       create_data("data/train","all_data_merged_final.csv")
    # create_test_data for the first time
    if not os.path.isfile("test_data_merged_final.csv"):
        create_data("data/test","test_data_merged_final.csv")

    keep_cols = "HR,O2Sat,Temp,SBP,MAP,Resp,BUN,Calcium,Creatinine,Glucose," \
                "Magnesium,Hct,Hgb,WBC,Age,Gender,ICULOS,SepsisLabel,patient_id,age_group".split(",")

    initial_df, train_mean_vals_df = preprocess_train_data("all_data_merged_final.csv", keep_cols)
    val_dict = train_mean_vals_df.to_dict(orient='index')

    test_df = preprocess_test("test_data_merged_final.csv", keep_cols, val_dict)
    stat_cols = ["age_group", "Gender", "ICULOS", "SepsisLabel"]
    model, predicted_df = rf_reg_mean(train_df=initial_df, test_df=test_df, stat_cols=stat_cols, n=10)
    predicted_df.to_csv("predicted_test_df_random_forest_new.csv")
    with open('rf_final_model_new.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    # train model
    main()

    # post analysis
    test_df_predicted = pd.read_csv("predicted_test_df_random_forest_new.csv")
    model_name = "rf_final_model_new.pkl"
    post_analysis(model_name, test_df_predicted)