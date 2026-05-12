from importlib.resources import path

from utils import options
from utils import folders_and_files

import os
import pandas as pd
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics


# Produces Figure 3, comparing the performance of KNN over various levels of n-significant


def OnHW_ML_read_filtered_data_and_extracted_features(case, dependency, k_fold_number, nsig):
    path_to_models_and_data = os.path.join(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA)
    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"

    train_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_X_filtered.npy"), allow_pickle=True)
    train_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_y_filtered.npy"), allow_pickle=True)
    test_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_X_filtered.npy"), allow_pickle=True)
    test_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_y_filtered.npy"), allow_pickle=True)
    features_filtered_train = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_train.csv"))
    features_filtered_test = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_test.csv"))

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered, features_filtered_train, features_filtered_test

def OnHW_ML_train_and_save(case, dependency, k_fold_number, nsig):
    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number, nsig)

    for k in range(1,50):
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{k}NN")
        folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"
        path_to_model = os.path.join(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA, folder_name)
        model_name, scaler_name = f"model_{k}NN_nsig{nsig}", f"scaler_{k}NN_nsig{nsig}"
        model, scaler = KNeighborsClassifier(n_neighbors=k), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=options.RANDOM_STATE)

        train_X_features_transformed = scaler.fit_transform(train_X_features)
        folders_and_files.save_model(path_to_model, scaler_name, scaler)

        model.fit(train_X_features_transformed, train_y)
        folders_and_files.save_model(path_to_model, model_name, model)


def _train_one_combination(args):
    """Worker: train all k values for one (case, dependency, fold, nsig) combination."""
    case, dependency, k_fold_number, nsig = args
    OnHW_ML_train_and_save(case, dependency, k_fold_number, nsig)
    return f"Train done: {case}_{dependency}_{k_fold_number}_nsig{nsig}"


def OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, nsig, k):
    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"
    path_to_model = os.path.join(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA, folder_name)
    model_name, scaler_name = f"model_{k}NN_nsig{nsig}", f"scaler_{k}NN_nsig{nsig}"
    model = folders_and_files.load_model(path_to_model, model_name)
    scaler = folders_and_files.load_model(path_to_model, scaler_name)

    return model, scaler


def OnHW_ML_evaluate_model(case, dependency, k_fold_number, nsig, k):
    model, scaler = OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, nsig, k)

    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number, nsig)

    if scaler==None:
        test_X_features_transformed = test_X_features.copy()
    else:
        test_X_features_transformed = scaler.transform(test_X_features)

    preds = model.predict(test_X_features_transformed)
    report = classification_report(test_y, preds, zero_division=1, digits=4)
    conf_mat = confusion_matrix(test_y, preds)
    accuracy = accuracy_score(test_y, preds)

    return accuracy, report, conf_mat


def _evaluate_one_combination(args):
    """Worker: evaluate all k values for one (case, dependency, fold, nsig) combination.
    Returns (rows_for_csv, scores_list, k_fold_number, nsig) instead of writing the shared
    CSV directly — the main process collects all rows and writes once to avoid races.
    """
    case, dependency, k_fold_number, nsig = args
    ml_results_path = os.path.join(options.BASE_OUTPUT, options.ML_RESULTS)

    rows = []
    scores_list = []

    for k in range(1, 50):
        print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}")
        accuracy, report, conf_mat = OnHW_ML_evaluate_model(case, dependency, k_fold_number, nsig, k)

        rows.append([case, dependency, k_fold_number, f"{k}NN_nsig{nsig}", accuracy])
        scores_list.append(accuracy)

        # Per-combination files have unique names — safe to write from worker
        report_file = os.path.join(ml_results_path, f"classification_report_{case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}.txt")
        with open(report_file, 'w') as f:
            print(report, file=f)

        conf_mat_file = os.path.join(ml_results_path, f"conf_mat_{case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}.csv")
        pd.DataFrame(conf_mat).to_csv(conf_mat_file, index=False)

    score_list_filepath = os.path.join(ml_results_path, f"kNN_fold{k_fold_number}_nsig{nsig}.txt")
    with open(score_list_filepath, 'wb') as fp:
        pickle.dump(scores_list, fp)

    return rows


def OnHW_ML_train_all(max_workers=None):
    combinations = [
        (case, dependency, k_fold_number, nsig)
        for case in options.OnHW_CASE
        for dependency in options.OnHW_DEPENDENCY
        for k_fold_number in options.OnHW_FOLD
        for nsig in options.NSIG_LIST
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_train_one_combination, args): args for args in combinations}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"ERROR in {futures[future]}: {e}")


def OnHW_ML_evaluate_all(max_workers=None):
    path_to_results = os.path.join(options.BASE_OUTPUT, options.ML_RESULTS, options.ML_RESULTS_VARIOUS_NSIG_CSV)

    combinations = [
        (case, dependency, k_fold_number, nsig)
        for case in options.OnHW_CASE
        for dependency in options.OnHW_DEPENDENCY
        for k_fold_number in options.OnHW_FOLD
        for nsig in options.NSIG_LIST
    ]

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_one_combination, args): args for args in combinations}
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as e:
                print(f"ERROR in {futures[future]}: {e}")

    df_results = pd.DataFrame(all_rows, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
    df_results.to_csv(path_to_results, index=False)


def make_some_folders():
    folders_and_files.make_folder_at('.', options.BASE_OUTPUT)

    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.ML_RESULTS)
    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.DL_RESULTS)
    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.DL_RESULTS_OPT)
    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.OPTUNA)

    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA)
    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.DL_MODELS_AND_DATA)

    for case in options.OnHW_CASE:
        for dependency in options.OnHW_DEPENDENCY:
            for k_fold_number in options.OnHW_FOLD:
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                path_to_models_and_data = os.path.join(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA)
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                path_to_models_and_data = os.path.join(options.BASE_OUTPUT, options.DL_MODELS_AND_DATA)
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)


if __name__ == "__main__":
    make_some_folders()

    '''Run OnHW_ML_extract_features_various_nsig.py prior to extract features for all levels of n_significant, specified in options.NSIG_LIST
    CHANGE XY TO ML TO RUN ML ALGORITHMS
    CHANGE XY TO MetricLearn TO RUN ML ALGORITHMS WITH METRIC LEARNING'''

    OnHW_ML_train_all()
    OnHW_ML_evaluate_all()
