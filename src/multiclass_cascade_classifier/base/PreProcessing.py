"""
Module PreProcessing

DataFrame PreProcessing class

@author: ThomasAujoux
"""



import pandas as pd

import base.variables.Variables as var



def remove_colon(list
                 ):
    n = len(list)
    i = 1
    colonfree = list[0]
    while i < n and list[i] != ":":
        colonfree = colonfree + " " + list[i]
        i = i + 1
    return colonfree

def remove_punctuation(text
                       ):
    list_punctuation = '!"#$%&\'()+,-./;:<=>?@[\\]^_`{|}~1234567890'
    punctuationfree="".join([i for i in text if i not in list_punctuation])
    return punctuationfree #storing the puntuation free text

def CleanColumns(X,
               columns_text=[]
                ):
    for column in var.columns_ingredient_pre:
        print("We are processing column:", column)
        X[column] = X[column].str.split().map(lambda x:remove_colon(x))
        X[column] = X[column].str.replace(r"\s\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\s\(.*\)\s", " ", regex=True)
        X[column] = X[column].str.replace(r"\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\(.*", "", regex=True)
        X[column] = X[column].apply(lambda x : x.replace('_',' '))
        X[column] = X[column].apply(lambda x : x.replace('*',' '))
        X[column] = X[column].apply(lambda x: x.rstrip())
        X[column] = X[column].apply(lambda x: x.lstrip())

        X[column]= X[column].apply(lambda x:remove_punctuation(x))

    X = X.groupby(var.columns_group_pre)[var.columns_ingredient_pre[0]].agg(lambda col: ' '.join(col)).reset_index(name=var.columns_ingredient_pre[0])

    for column in columns_text:
        print("We are processing column:", column)
        X[column] = X[column].str.split().map(lambda x:remove_colon(x))
        X[column] = X[column].str.replace(r"\s\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\s\(.*\)\s", " ", regex=True)
        X[column] = X[column].str.replace(r"\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\(.*", "", regex=True)
        X[column] = X[column].apply(lambda x : x.replace('_',' '))
        X[column] = X[column].apply(lambda x : x.replace('*',' '))
        X[column] = X[column].apply(lambda x: x.rstrip())
        X[column] = X[column].apply(lambda x: x.lstrip())

        X[column]= X[column].apply(lambda x:remove_punctuation(x))

    return X



from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 columns_text=[]
                ):

        self.columns_text=columns_text

    def transform(self, X, **transform_params):
        # Nettoyage du texte
        X=X.copy() # pour conserver le fichier d'origine
        return CleanColumns(X,
               columns_text=self.columns_text,
               )

    def fit(self, X, y=None, **fit_params):
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {
            'columns_text':self.columns_text
        }    
    
    def set_params (self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self    
    
