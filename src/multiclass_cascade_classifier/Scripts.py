"""
Module Scripts

Scripts functions

Handles pd.DataFrame

@author: ThomasAujoux
"""


### Imports ###

from sklearn.model_selection import train_test_split

import time
import sys

import Variables as var

from VariablesChecker import check_csv, check_yaml#, check_folder, create_folder, check_yaml_sector, check_yaml_families
from VariablesChecker import check_test_size#, checks_nFamilies, check_trained_classifiers, check_classifiers_train_sectors, check_classifiers_test_diff
from DataHelper import get_dataframe

from DataFrameNormalizer import DataFrameNormalizer
from DataVectorizer import DataVectorizer

from HyperSelector import select_hyperparameters_sector, select_hyperparameters_family
from DataTrainer import train_sector_classifier, train_families_per_sector_classifier





### Check functions ###

def check_split(csv_in, csv_out_train, csv_out_test, test_size):
    """
    Checks if the arguments given by the user for the split are valid.

    Parameters
    ----------
    csv_in : String
        Path to the file that contains the data set.
    csv_out_train : String
        Path to the file that contains the train set.
    csv_out_test : String
        Path to the file that contains the test set.
    test_size : Float
        Size of test set.

    Returns
    -------
    csv_in : String
        Updated path to the file that contains the data set.
    csv_out_train : String
        Updated path to the file that contains the train set.
    csv_out_test : String
        Updated path to the file that contains the test set.
    test_size : Float
        Updated size of test set.

    """
    print("Initialization...")
    csv_in = check_csv(csv_in, True)
    csv_out_train = check_csv(csv_out_train, False)
    csv_out_test = check_csv(csv_out_test, False)
    test_size = check_test_size(test_size)
    
    return csv_in, csv_out_train, csv_out_test, test_size


### data ###

def load_data(csv_in, index_column=None, columns=None, logjournal=None):
    """
    Load raw data

    Parameters
    ----------
    csv_in : String
        Path to data file (csv).
    index_column : List<String>, optional
        Index columns of DataFrame. The default is var.column_index.
    columns : List<String>, optional
        Columns of DataFrame. The default is var.columns.

    Returns
    -------
    df_produit : pd.DataFrame
        Données.

    """

    if logjournal:
        logjournal.write_text("Loading data.")
    
    df_produit = get_dataframe(csv_in)
    
    if index_column:
        if any(item in df_produit.columns.tolist() for item in index_column):
            df_produit.set_index(index_column, inplace=True)
        else:
            if len(index_column) == 1:
                df_produit[index_column] = df_produit.index
            else:
                index = df_produit.index.tolist()
                index_values = [[i for j in index_column] for i in index]
                df_produit[index_column] = index_values
            df_produit.set_index(index_column, inplace=True)
    
    if columns:
        df_produit = df_produit[columns]
        
    return df_produit


def save_data(csv_out, df_produit, logjournal=None):
    """
    Saves the DataFrame into csv_out.

    Parameters
    ----------
    csv_out : String
        Path to data file (csv) where the data will be saved.
    df_produit : pd.DataFrame
        Data to save.

    Returns
    -------
    None.

    """

    if logjournal:
        logjournal.write_text("Saving data.")
    
    df_produit.to_csv(csv_out, index=True, sep=';')


def prepare_data(df_data, logjournal):
    """
    Prepares data.
    Pretreatment and vectorization

    Parameters
    ----------
    df_data : pd.DataFrame
        Data set.

    Returns
    -------
    X_vect : pd.DataFrame
        Pretreated and vectorized data set.

    """
    
    # Pre-treatment
    print("Pretreatment...")
    if logjournal:
        logjournal.write_text("Pretreatment.")
    start_time = time.time()
    df_normalizer=DataFrameNormalizer(lowercase=var.lowercase, removestopwords=var.removestopwords, removedigit=var.removedigit, getstemmer=var.getstemmer, getlemmatisation=var.getlemmatisation, columns_text=var.columns_text, columns_binary=var.columns_bin, columns_frozen=var.columns_frozen)
    X_train = df_normalizer.fit_transform(df_data)
    pretreatment_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(pretreatment_time)
    
    ## Vectorization
    print("Vectorization...")
    if logjournal:
        logjournal.write_text("Vectorization.")
    start_time = time.time()
    df_vectorizer = DataVectorizer(columns_text=var.columns_text, columns_binary=var.columns_bin)
    X_vect = df_vectorizer.fit_transform(X_train)
    vectorization_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(vectorization_time)
    
    return X_vect



# # ################## Tests ####################
# csv_in = "./data2/merged_final.csv"
# df_data = load_data(csv_in=csv_in, index_column=None, columns=var.columns, logjournal=None)
# print(df_data)
# new = prepare_data(df_data, logjournal=None)
# print(new)

# # ################## Tests ####################


### Split ###

def split_train_test(df_produit, test_size):
    """
    Splits data from csv_in into a training set and a test set with size of test set being predict_size%

    Parameters
    ----------
    df_produit : pandas.DataFrame
        Data.
    predict_size : Float
        Size of test set.

    Returns
    -------
    X_train : pd.DataFrame
        Train set.
    X_test : pd.DataFrame
        Test set.

    """
    
    X = df_produit[var.columns_X_id]
    y = df_produit[var.id_famille]
    
    ## Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    ## Retrieving indexes
    # Train
    index_train = X_train.index.to_list()
    # Test
    index_test = X_test.index.to_list()
    
    ## Concatenating train and test
    # Train
    X_train[var.id_famille] = y_train
    # Test
    X_test[var.id_famille] = y_test
    
    # Retrieving other columns
    columns_rest = []
    for column in var.columns_all:
        if column not in X_train.columns.values.ravel():
            columns_rest.append(column)
        
    X_train[columns_rest] = df_produit.loc[index_train][columns_rest]
    X_test[columns_rest] = df_produit.loc[index_test][columns_rest]
    
    return X_train, X_test


# # ################## Tests ####################
# new2 = split_train_test(df_data, 0.8)
# print(new2[0], 2222222)
# print(new2[1], 3333333)
# # ################## Tests ####################


### Modele ###

def select_hyperparameters(X, y, hyper_sector_file=None, hyper_family_per_sector_file=None, sectors_diff=None, n_jobs=var.n_jobs, logjournal=None):
    """
    Selects hyperparameters (or load them if yaml files are filled).

    Parameters
    ----------
    X : pd.DataFrame
        Data train set.
    y : pd.DataFrame
        Labels of data trained set.
    hyper_sector_file : String, optional
        Path to yaml file that contains the sector classifier's hyperparameters. The default is None.
    hyper_family_per_sector_file : String, optional
        Path to yaml file that contains the family classifiers' hyperparameters. The default is None.
    sectors_diff : List<String>, optional
        List of sectors that don't have their hyperparameters filled in the yaml file. The default is None.
    n_jobs : Integer, optional
        Number of jobs created during cross-validation (hyperparameters selection). The default is var.n_jobs.

    Returns
    -------
    clf_sector : Classifier
        Initialized sector classifier.
    clfs_family : Dict<Classifier>
        Initialiazed family classifiers.

    """
    
    print("Hyperparameters selection...")
    if logjournal:
        logjournal.write_text("Hyperparameters selection.")
    
    ## Sector
    print("Sectors...")
    if logjournal:
        logjournal.write_text("\tHyperparameters selection: dectors")
    start_time = time.time()
    clf_sector = select_hyperparameters_sector(X, y, hyper_sector_file, n_jobs, logjournal)
    sector_selection_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(sector_selection_time)

    ## Families
    print("Families...")
    if logjournal:
        logjournal.write_text("\tHyperparameters selection: families")
    start_time = time.time()
    clfs_family = select_hyperparameters_family(X, y, hyper_family_per_sector_file, sectors_diff, n_jobs, logjournal)
    family_selection_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(family_selection_time)
    
    return clf_sector, clfs_family




def train_data(X, y, clf_sector, clfs_family, logjournal=None):
    """
    Trains classifiers on train set.

    Parameters
    ----------
    X : pd.DataFrame
        Data train set.
    y : pd.DataFrame
        Labels of data trained set.
    clf_sector : Classifier
        Initialized sector classifier.
    clfs_family : Dict<Classifier>
        Initialiazed family classifiers.

    Returns
    -------
    clf_sector_trained : Classifier
        Trained sector classifier.
    clfs_family_trained : Dict<Classifier>
        Trained family classifiers.

    """
    
    print("Training...")
    if logjournal:
        logjournal.write_text("Training")
    
    ## Sector
    print("Sectors...")
    if logjournal:
        logjournal.write_text("\tTraining: sectors")
    start_time = time.time()
    clf_sector_trained = train_sector_classifier(X, y, clf_sector, logjournal)
    sector_training_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(sector_training_time)
    
    ## Families
    print("Families...")
    if logjournal:
        logjournal.write_text("\tTraining: families")
    start_time = time.time()
    clfs_family_trained = train_families_per_sector_classifier(X, y, clfs_family, logjournal)
    family_training_time = var.time_ % (divmod(time.time() - start_time, 60))
    print(family_training_time)
    
    return clf_sector_trained, clfs_family_trained


# # ################## Tests ####################
# new2 = split_train_test(df_data, 0.2)
# print(new2[0], 2222222)
# print(new2[1], 3333333)
# # ################## Tests ####################
