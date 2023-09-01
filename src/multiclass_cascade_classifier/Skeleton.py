"""
Module Skeleton

Skeleton functions

Handles CSV files

@author: ThomasAujoux
"""

### Imports ###

import Variables as var
from Scripts import check_split, check_train, check_classifiers_train
from Scripts import load_data, prepare_data, save_data, save_classifiers
from Scripts import prepro
from Scripts import select_hyperparameters, save_hyperparameters
from Scripts import split_train_test, train_data

from LogJournal import LogJournal


### Split ###

def split(csv_in, csv_out_train="train_test/train_split.csv", csv_out_test="train_test/test_split.csv", test_size=0.20):
    """
    Splits data from csv_in into a training set and a test set with size of test set being test_size%
    Saves the new sets at train_out and test_out.

    Parameters
    ----------
    csv_in : String
        File where raw data are stored.
    test_size : Int, optional
        Size of the test set. The default is 80%.
    csv_out_train : String, optional
        File where the training set will be stored. The default is "train_split.csv".
    csv_out_test : String, optional
        File where the test set will be stored. The default is "test_split.csv".

    Returns
    -------
    None.

    """
    
    ## Checking variables
    csv_in, csv_out_train, csv_out_test, test_size = check_split(csv_in, csv_out_train, csv_out_test, test_size)
    
    ## Loading data
    df_produit = load_data(csv_in, index_column=None, columns=var.columns, logjournal=None)
    df_produit = prepro(df_produit, logjournal=None)
    X_train, X_test = split_train_test(df_produit, test_size)
    print(X_train.columns, X_test.columns)
    print(X_train.head(), X_test.head())
    # Saving data sets
    save_data(csv_out_train, X_train)
    save_data(csv_out_test, X_test)

# # # ################## Tests ####################
# csv_in = "./data2/merged_final.csv"

# split(csv_in)

# # # ################## Tests ####################


### Modele ###

def train(csv_train_in, models_folder, hyper_sector_file=None, hyper_family_per_sector_file=None, force=True, n_jobs=var.n_jobs, log_folder=None):
    """
    Trains classifiers on data train set (csv_train_in) and saves trained classifier into models_folder.

    Parameters
    ----------
    csv_train_in : String
        Path to data train set.
    models_folder : String
        Path to models folder (where the joblib files will be saved).
    hyper_sector_file : String, optional
        Path to yaml file where the hyperparameters for the sector classifier are stored. The default is None.
    hyper_family_per_sector_file : String, optional
        Path to yaml file where the hyperparameters for the family classifier are stored. The default is None.

    Returns
    -------
    None.

    """

    ## Checking variables
    csv_train_in, models_folder, hyper_sector_file, hyper_family_per_sector_file, log_folder = check_train(csv_train_in, models_folder, hyper_sector_file, hyper_family_per_sector_file, log_folder)

    # Log Journal
    log_journal = LogJournal(log_folder, "log") if log_folder else None

    if log_journal:
        log_journal.write_texts([
            "%s: %s" % (v_i, str(v_j)) for v_i, v_j in [
                ("csv_train_in", csv_train_in),
                ("models_folder", models_folder),
                ("hyper_sector_file", hyper_sector_file),
                ("hyper_family_per_sector_file", hyper_family_per_sector_file),
                ("log_folder", log_folder),
                ("n_jobs", n_jobs),
                ("force", force),
                ("var.hyperParamsGrid", var.hyperParamsGrid),
            ]
        ])

    ## Loading data
    df_train = load_data(csv_train_in, index_column=None, columns=var.columns, logjournal=log_folder)
    training_size = df_train.shape[0]
    
    ## Preparing data
    y_train = df_train[var.columns_label]
    sectors_diff = check_classifiers_train(y_train, hyper_family_per_sector_file, force)
    X_train = prepare_data(df_train, log_journal)
    
    ## Select hyperparameters
    clf_sector, clfs_family = select_hyperparameters(X_train, y_train, hyper_sector_file, hyper_family_per_sector_file, sectors_diff, n_jobs, log_journal)
    save_hyperparameters(models_folder, clf_sector, clfs_family, training_size, log_journal)
    
    ## Train Data
    clf_sector_trained, clfs_family_trained = train_data(X_train, y_train, clf_sector, clfs_family, log_journal)
    save_classifiers(models_folder, clf_sector_trained, clfs_family_trained, log_journal)

    if log_journal:
        log_journal.close()



# # ################## Tests ####################
csv_train_in = "./train_test/train_split.csv"
models_folder = "./model"
hyper_sector_file = "./hyper/hyper_sector.yaml"
hyper_family_per_sector_file = "./hyper/hyper_family.yaml"
hyper_sector_file = None
hyper_family_per_sector_file = None
train(csv_train_in, models_folder, hyper_sector_file, hyper_family_per_sector_file, force=True, n_jobs=var.n_jobs, log_folder=None)
# # ################## Tests ####################