"""
Module Skeleton

Skeleton functions

Handles CSV files

@author: ThomasAujoux
"""

### Imports ###

import Variables as var
from Scripts import check_split
from Scripts import load_data, save_data
from Scripts import split_train_test


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
    
    X_train, X_test = split_train_test(df_produit, test_size)
    print(X_train.columns, X_test.columns)
    # Saving data sets
    save_data(csv_out_train, X_train)
    save_data(csv_out_test, X_test)

# # ################## Tests ####################
csv_in = "./data2/merged_final.csv"

split(csv_in)

# # ################## Tests ####################