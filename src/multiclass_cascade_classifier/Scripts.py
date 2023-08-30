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

from DataHelper import get_dataframe
from DataFrameNormalizer import DataFrameNormalizer
from DataVectorizer import DataVectorizer



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



# ################## Tests ####################
csv_in = "./data/merged_final.csv"
var.columns
df_data = load_data(csv_in=csv_in, index_column=None, columns=var.columns, logjournal=None)

# ################## Tests ####################