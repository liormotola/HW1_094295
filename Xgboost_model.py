from preprocess import preprocess_train_data, preprocess_test
import utils
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier , plot_importance
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt

def xgboost_weighted_mean(train_df, test_df, stat_cols, n=0):
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
    with open('scaler_xgboost.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    scaled_df_test = aggregated_test.copy()
    scaled_df_test[scaling_cols] = scaler.transform(scaled_df_test[scaling_cols])

    X_train = scaled_df.drop("SepsisLabel", axis=1)
    y_train = scaled_df.SepsisLabel
    X_test = scaled_df_test.drop("SepsisLabel", axis=1)
    y_test = scaled_df_test.SepsisLabel

    xgboost = XGBClassifier(n_estimators=150, use_label_encoder=False, scale_pos_weight=12, eval_metric=f1_score,
                            verbosity=1, disable_default_eval_metric=1, enable_categorical=True, tree_method="approx")

    xgboost.fit(X_train, y_train)
    predicted_train = xgboost.predict(X_train)
    predicted_test = xgboost.predict(X_test)

    utils.model_eval(y_train, predicted_train, "train",title="xgboost")
    utils.model_eval(y_test, predicted_test, "test",title="xgboost")
    scaled_df_test["predicted"] = predicted_test

    return xgboost , scaled_df_test


def xgboost_reg_mean(train_df, test_df , stat_cols, n=0):

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

    with open('scaler_new.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    scaled_df_test = aggregated_test.copy()
    scaled_df_test["ICULOS_scaled"] = scaled_df_test.ICULOS
    scaled_df_test[scaling_cols] = scaler.transform(scaled_df_test[scaling_cols])

    X_train = scaled_df.drop(["SepsisLabel","ICULOS"], axis=1)
    y_train = scaled_df.SepsisLabel
    X_test = scaled_df_test.drop(["SepsisLabel","ICULOS"], axis=1)
    y_test = scaled_df_test.SepsisLabel


    xgboost = XGBClassifier(n_estimators=200, scale_pos_weight=8, eval_metric=f1_score,
                            verbosity=1, disable_default_eval_metric=1, enable_categorical=True, tree_method="approx",
                            learning_rate=0.05, max_delta_step=2,max_depth = 6)

    xgboost.fit(X_train, y_train)
    predicted_train = xgboost.predict(X_train)
    predicted_test = xgboost.predict(X_test)

    utils.model_eval(y_train, predicted_train, "train",title="xgboost")
    print()
    utils.model_eval(y_test, predicted_test, "test",title="xgboost")
    scaled_df_test["predicted"] = predicted_test

    return xgboost, scaled_df_test


def xgboost_reg_mean_parameter_tuning(train_df , stat_cols, n=0):
    """
    Tuning parameters for xgboost model with mean aggregation over n last rows
    :param train_df: train data
    :param stat_cols: static columns - not to be aggregated by mean
    :param n: number of rows to take into account
    :return: prints best parameters found
    """

    if n > 0 :
        train_df = train_df.groupby('patient_id').apply(utils.last_n_rows, n=n).reset_index(drop=True)

    scaling_cols = list(set(train_df.columns).difference(set(stat_cols + ["patient_id", "Age"])))
    scaling_cols += ["ICULOS"]
    aggregated_train = utils.aggregate_df_reg_mean(stat_cols, scaling_cols, train_df)
    scaler = StandardScaler()

    scaled_df = aggregated_train.copy()
    scaled_df[scaling_cols] = scaler.fit_transform(scaled_df[scaling_cols])
    X_train = scaled_df.drop("SepsisLabel", axis=1)
    y_train = scaled_df.SepsisLabel

    print("starting searching")
    optimizer = GridSearchCV(XGBClassifier(), {

        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [ 150, 170,200],
        'tree_method': ["approx"],
        "max_delta_step": [1,2],
        "scale_pos_weight" : [8,10,12],
        'max_depth':[3,5,7],
        "subsample":[0.5,0.75,1],
        'eval_metric':[f1_score],
        'enable_categorical' : [True],
    }, scoring="f1", cv=3)

    optimizer.fit(X_train, y_train)
    print('Optimizing complete')
    print(optimizer.best_params_)
    print(optimizer.best_score_)
    df_optimizer = pd.DataFrame(optimizer.cv_results_).dropna()
    df_optimizer = df_optimizer[
        ['param_learning_rate', 'param_n_estimators', 'param_tree_method', 'param_max_delta_step','param_scale_pos_weight',
         'mean_test_score']]
    df_optimizer.to_csv("params_xgboost.csv")

def plot_xgb_importance(model,importance_type):
    """
    plots importance graphs of xgboost object (trained model)
    :param model: trained xgboost object
    :param importance_type: gain/cover/weight
    """
    plot_importance(model, importance_type=importance_type)
    plt.title(f"xgboost importance plot - {importance_type}")
    plt.savefig(f"xgboost importance plot - {importance_type}")
    plt.show()

def post_analysis(model,predicted_df):
    #we left only the code we eventually used
    xgb = XGBClassifier()
    xgb.load_model(model)
    plot_xgb_importance(xgb, importance_type="weight")
    plot_xgb_importance(xgb, importance_type="cover")
    plot_xgb_importance(xgb, importance_type="gain")
    utils.ICULOS_analysis(predicted_df)

    #check gender subgroups performance:
    for i in range(2):
        df1 = predicted_df[predicted_df.Gender == i]
        print(f"Gender {i}")
        print(f1_score(df1.SepsisLabel, df1.predicted))

    #check Age_group subgroups performance:
    for i in range(6):
        df1 = predicted_df[predicted_df.age_group == i]
        print(f"Age group {i}")
        print(f1_score(df1.SepsisLabel, df1.predicted))

def main():

    keep_cols = "HR,O2Sat,Temp,SBP,MAP,Resp,BUN,Calcium,Creatinine,Glucose," \
                "Magnesium,Hct,Hgb,WBC,Age,Gender,ICULOS,SepsisLabel,patient_id,age_group".split(",")

    initial_df, train_mean_vals_df = preprocess_train_data("all_data_merged_final.csv", keep_cols)
    val_dict = train_mean_vals_df.to_dict(orient='index')

    with open('null_vals_new.pkl', 'wb') as f:
        pickle.dump(val_dict, f)

    test_df = preprocess_test("test_data_merged_final.csv", keep_cols, val_dict)
    stat_cols = ["age_group", "Gender", "ICULOS", "SepsisLabel"]
    model, predicted_df = xgboost_reg_mean(train_df=initial_df, test_df=test_df, stat_cols=stat_cols, n=10)
    #saving results to csv for post analysis
    predicted_df.to_csv("predicted_test_df_xgboost_new.csv")
    model.save_model('xgboost_final_model_new.json')

if __name__ == '__main__':
    #train model
    main()

    #post analysis
    test_df_predicted = pd.read_csv("predicted_test_df_xgboost_new.csv")
    model_name = "xgboost_final_model_new.json"
    post_analysis(model_name,test_df_predicted)