from preprocess import preprocess_train_data , preprocess_test
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score


def adaboost_weighted_mean(train_df, test_df, stat_cols, n=0):
    if n > 0:
        train_df = train_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)
        test_df = test_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)

    scaling_cols = list(set(train_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))

    aggregated_train = utils.aggregate_df_weighted_mean(stat_cols, scaling_cols, train_df)
    aggregated_test = utils.aggregate_df_weighted_mean(stat_cols, scaling_cols, test_df)

    scaler = StandardScaler()
    scaled_df = aggregated_train.copy()
    scaling_cols += ["ICULOS"]
    scaled_df[scaling_cols] = scaler.fit_transform(scaled_df[scaling_cols])
    with open('scaler_adaboost.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    scaled_df_test = aggregated_test.copy()
    scaled_df_test[scaling_cols] = scaler.transform(scaled_df_test[scaling_cols])

    X_train = scaled_df.drop("SepsisLabel", axis=1)
    y_train = scaled_df.SepsisLabel
    X_test = scaled_df_test.drop("SepsisLabel", axis=1)
    y_test = scaled_df_test.SepsisLabel

    adaboost = AdaBoostClassifier(n_estimators=170, random_state=0)
    adaboost.fit(X_train, y_train)

    predicted_train = adaboost.predict(X_train)
    predicted_test = adaboost.predict(X_test)

    utils.model_eval(y_train, predicted_train, "train",title="Adaboost")
    utils.model_eval(y_test, predicted_test, "test",title="Adaboost")
    scaled_df_test["predicted"] = predicted_test

    return adaboost , scaled_df_test


def adaboost_reg_mean(train_df, test_df, stat_cols, n=0):
    if n > 0:
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

    X_train = scaled_df.drop(["SepsisLabel","ICULOS"], axis=1)
    y_train = scaled_df.SepsisLabel
    X_test = scaled_df_test.drop(["SepsisLabel","ICULOS"], axis=1)
    y_test = scaled_df_test.SepsisLabel

    adaboost = AdaBoostClassifier(n_estimators=170, random_state=0)
    adaboost.fit(X_train, y_train)

    predicted_train = adaboost.predict(X_train)
    predicted_test = adaboost.predict(X_test)

    utils.model_eval(y_train, predicted_train, "train",title="Adaboost")
    print("\n")
    utils.model_eval(y_test, predicted_test, "test",title="Adaboost")
    scaled_df_test["predicted"] = predicted_test

    return adaboost, scaled_df_test

def adaboost_reg_mean_parameter_tuning(train_df , stat_cols, n=0):

    if n > 0 :
        train_df = train_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)

    scaling_cols = list(set(train_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))

    aggregated_train = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, train_df)
    scaler = StandardScaler()

    scaled_df = aggregated_train.copy()
    scaled_df[scaling_cols] = scaler.fit_transform(scaled_df[scaling_cols])
    X_train = scaled_df.drop("SepsisLabel", axis=1)
    y_train = scaled_df.SepsisLabel

    print("starting searching")
    optimizer = GridSearchCV(AdaBoostClassifier(), {

        'learning_rate': [0.5,1,2,5],
        'n_estimators': [ 50,100, 150, 170,200],
        'random_state':[0]

    }, scoring="f1", cv=3)

    optimizer.fit(X_train, y_train)
    print('Optimizing complete')
    print(optimizer.best_params_)
    print(optimizer.best_score_)
    df_optimizer = pd.DataFrame(optimizer.cv_results_).dropna()
    df_optimizer = df_optimizer[
        ['param_learning_rate', 'param_n_estimators', 'mean_test_score']]
    df_optimizer.to_csv("params_adaboost.csv")


def main():

    keep_cols = "HR,O2Sat,Temp,SBP,MAP,Resp,BUN,Calcium,Creatinine,Glucose," \
                "Magnesium,Hct,Hgb,WBC,Age,Gender,ICULOS,SepsisLabel,patient_id,age_group".split(",")

    initial_df, train_mean_vals_df = preprocess_train_data("all_data_merged_final.csv", keep_cols)
    val_dict = train_mean_vals_df.to_dict(orient='index')

    test_df = preprocess_test("test_data_merged_final.csv", keep_cols, val_dict)
    stat_cols = ["age_group", "Gender", "ICULOS", "SepsisLabel"]
    model, predicted_df = adaboost_reg_mean(train_df=initial_df, test_df=test_df, stat_cols=stat_cols, n=10)
    predicted_df.to_csv("predicted_test_df_adaboost_new.csv")
    with open('adaboost_final_model_new.pkl', 'wb') as f:
        pickle.dump(model, f)

def post_analysis(model,predicted_df):
    with open(model,"rb") as f:
        adaboost = pickle.load(f)

    #check feature importance
    utils.create_features_importance_plot(model= adaboost,title= 'Ada Boost model - Features importance')

    #ICULOS analysis
    utils.ICULOS_analysis(predictions_df=predicted_df)

    #check gender subgroups performance:
    for i in range(2):
        df1 = predicted_df[predicted_df.Gender == i]
        print(f"Age group {i}")
        print(f1_score(df1.SepsisLabel, df1.predicted))

    # check HR subgroups performance:
    utils.HR_analysis(predicted_df=predicted_df)

if __name__ == '__main__':
    # train model
    # main()

    # post analysis
    test_df_predicted = pd.read_csv("predicted_test_df_adaboost_new.csv")
    model_name = "adaboost_final_model_new.pkl"
    post_analysis(model_name, test_df_predicted)
