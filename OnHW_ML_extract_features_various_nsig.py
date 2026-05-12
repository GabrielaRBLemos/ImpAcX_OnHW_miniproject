from importlib.resources import path

from utils import options
from utils import folders_and_files

import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC




def load_OnHW_data(case, dependency, k_fold_number):
    path_to_folder = f"onhw2_{case}_{dependency}_{k_fold_number}"
    train_X = np.load(os.path.join(options.PATH_TO_PREPEND, path_to_folder, "X_train.npy"), allow_pickle=True)
    test_X = np.load(os.path.join(options.PATH_TO_PREPEND, path_to_folder, "X_test.npy"), allow_pickle=True)
    train_y = np.load(os.path.join(options.PATH_TO_PREPEND, path_to_folder, "y_train.npy"), allow_pickle=True)
    test_y = np.load(os.path.join(options.PATH_TO_PREPEND, path_to_folder, "y_test.npy"), allow_pickle=True)
    return train_X, train_y, test_X, test_y

def filter_train_test(X, y, case):
    lower_bound, upper_bound = max(options.HARD_LOWER_BOUND, options.MEAN[case] - options.CUT_OFF_COEFF*options.STD[case]), min(options.HARD_UPPER_BOUND, options.MEAN[case] + options.CUT_OFF_COEFF*options.STD[case])
    mask = [(len(data) >= lower_bound) & (len(data) <= upper_bound) for data in X]
    return X[mask], y[mask]

def X_npy_to_df(X):
    feat_cols = [f'f_{i}' for i in range(13)]
    chunks = []
    for id, data in enumerate(X):
        data = np.asarray(data, dtype=np.float64)
        chunk = pd.DataFrame(data, columns=feat_cols)
        chunk['id'] = id
        chunk['time'] = np.arange(len(data))
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df[['id', 'time'] + feat_cols]

def _extract_and_select_for_fold(args):
    """Worker: extract features once per (case, dep, fold), then select for each nsig.
    Runs extract_features only once regardless of how many nsig values are requested,
    avoiding redundant computation and reducing peak memory usage.
    """
    case, dependency, k_fold_number, nsig_list = args
    path_to_models_and_data = os.path.join(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA)

    # Checkpoint: skip entirely if all nsig outputs already exist
    all_done = all(
        os.path.exists(os.path.join(path_to_models_and_data,
                                    f"{case}_{dependency}_{k_fold_number}_nsig{nsig}",
                                    'features_filtered_train.csv'))
        for nsig in nsig_list
    )
    if all_done:
        return f"Skipped (all nsig done): {case}_{dependency}_{k_fold_number}"

    print(f"[INFO] Loading and filtering: {case}_{dependency}_{k_fold_number}")
    train_X, train_y, test_X, test_y = load_OnHW_data(case, dependency, k_fold_number)
    train_X_filtered, train_y_filtered = filter_train_test(train_X, train_y, case)
    test_X_filtered, test_y_filtered = filter_train_test(test_X, test_y, case)

    df_train_X = X_npy_to_df(train_X_filtered)

    # n_jobs=1 to avoid nested multiprocessing inside worker
    print(f"[INFO] Extracting features: {case}_{dependency}_{k_fold_number}")
    extracted_features = extract_features(df_train_X, column_id='id', column_sort="time", n_jobs=1)
    impute(extracted_features)

    df_test_X = X_npy_to_df(test_X_filtered)

    for nsig in nsig_list:
        folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"
        out_folder = os.path.join(path_to_models_and_data, folder_name)

        if os.path.exists(os.path.join(out_folder, 'features_filtered_train.csv')):
            print(f"[INFO] Skipping (already done): {folder_name}")
            continue

        folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

        features_filtered_train = select_features(extracted_features, train_y_filtered,
                                                   multiclass=True, n_significant=nsig)

        features_filtered_test = extract_features(
            df_test_X, column_id='id', column_sort="time",
            kind_to_fc_parameters=settings.from_columns(features_filtered_train.columns),
            n_jobs=1)
        features_filtered_test = features_filtered_test[features_filtered_train.columns]
        impute(features_filtered_test)

        np.save(os.path.join(out_folder, 'train_X_filtered.npy'), train_X_filtered)
        np.save(os.path.join(out_folder, 'train_y_filtered.npy'), train_y_filtered)
        np.save(os.path.join(out_folder, 'test_X_filtered.npy'), test_X_filtered)
        np.save(os.path.join(out_folder, 'test_y_filtered.npy'), test_y_filtered)
        features_filtered_train.to_csv(os.path.join(out_folder, 'features_filtered_train.csv'), index=False)
        features_filtered_test.to_csv(os.path.join(out_folder, 'features_filtered_test.csv'), index=False)
        print(f"[INFO] Done: {folder_name}")

    return f"Done: {case}_{dependency}_{k_fold_number}"


'''For each train and test OnHW data, the data is filtered and
statistical features are extracted using tsfresh. Corresponding outputs are stored under the 'output' folder.
The code assumes that the 'ouput' folder already exists.
The OnHW dataset (https://stabilodigital.com/onhw-dataset/) should be unzipped and placed under the 'dataset' folder so it looks like
dataset/
└── IMWUT_OnHW-chars_dataset_2021-06-30
    └── OnHW-chars_2021-06-30
        ├── onhw2_both_dep_0
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        ├── onhw2_both_dep_1
        ├── ...
        ├── ...
        ├── onhw2_upper_indep_4
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        └── readme.txt
'''
def OnHW_ML_filter_and_extract(max_workers=1):
    """Extract tsfresh features for all (case, dependency, fold) × nsig combinations.

    Groups by fold so extract_features runs once per fold (not once per nsig),
    reducing total work by len(NSIG_LIST)×. Defaults to max_workers=1 to avoid
    OOM on memory-constrained environments (e.g. Colab free tier).
    Increase max_workers only if RAM allows multiple simultaneous extractions.
    """
    folders_and_files.make_folder_at(options.BASE_OUTPUT, options.ML_MODELS_AND_DATA)

    # One task per (case, dependency, fold) — handles all nsig values internally
    fold_combinations = [
        (case, dependency, k_fold_number, options.NSIG_LIST)
        for case in options.OnHW_CASE
        for dependency in options.OnHW_DEPENDENCY
        for k_fold_number in options.OnHW_FOLD
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_and_select_for_fold, args): args for args in fold_combinations}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                args = futures[future]
                print(f"ERROR in {args[:3]}: {e}")


if __name__ == "__main__":
    # os.chdir('/Volumes/T7') # For accessing feature data (Steven)

    '''For Figure 3 and 4, used all 5 folds of lowercase writer-independent data'''
    # folders_and_files.make_folder_at('.', options.BASE_OUTPUT)
    OnHW_ML_filter_and_extract()
