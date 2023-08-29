"""
Module DataFrameNormalizer

DataFrame normalization class

@author: Thomas Aujoux
"""




import pandas as pd
import re
import string

# from sklearn import preprocessing


import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words_list = list(set(stopwords.words('french')) )
inrae_dictio = ["", "a", "mg", "cm", "g", "gl", "ml", "k", "el", "ed", "gr" "k" "mi" "st" "the" , "kg", "dl", "l", "cl", "about", "ad", "al", "and", "in", "it", "too"]
for word in inrae_dictio:
    stop_words_list.append(word)



# import oqlassif.Variables as var


csv_in = "C:/Users/Thomas Aujoux/Documents/GitHub/package/src/multiclass_cascade_classifier/data/merged_final.csv"
X = get_dataframe(csv_in)


columns_text_pre = ["Denomination_de_vente", "Nom", "Ingredient"]
columns_binary_pre = ["Code_produit", "Secteur", "Famille", "Denomination_de_vente", "Nom", "Conservation"]
columns_ingredient_pre = "Ingredient"
X = CleanColumns(X, columns_text_pre, columns_binary_pre, columns_ingredient_pre)


columns_text = ["Denomination_de_vente", "Nom", "Conservation"]
columns_binary=["Code_produit", "Secteur", "Famille"]
columns_frozen=[]
CleanDataFrame(X, True,True,True,True,True,columns_text,columns_binary,columns_frozen)



def remove_colon(list):
    n = len(list)
    i = 1
    colonfree = list[0]
    while i < n and list[i] != ":":
        colonfree = colonfree + " " + list[i]
        i = i + 1
    return colonfree


def remove_punctuation(text):
    list_punctuation = '!"#$%&\'()+,-./;:<=>?@[\\]^_`{|}~1234567890'
    punctuationfree="".join([i for i in text if i not in list_punctuation])
    return punctuationfree #storing the puntuation free text


def CleanColumns(X,
               columns_text_pre = [],
               columns_binary_pre = [],
               columns_ingredient_pre = "Ingredient"
                ):
    for column in columns_text_pre:
        X[column] = X[column].str.split().map(lambda x:remove_colon(x))
        X[column] = X[column].str.replace(r"\s\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\s\(.*\)\s", " ", regex=True)
        X[column] = X[column].str.replace(r"\(.*\)", "", regex=True)
        X[column] = X[column].str.replace(r"\(.*", "", regex=True)
        X[column] = X[column].apply(lambda x: x.rstrip())
        X[column] = X[column].apply(lambda x: x.lstrip())
        X[column] = X[column].apply(lambda x : x.replace('_',' '))
        X[column] = X[column].apply(lambda x : x.replace('*',' '))
        X[column]= X[column].apply(lambda x:remove_punctuation(x))

    X = X.groupby(columns_binary_pre)[columns_ingredient_pre].agg(lambda col: ' '.join(col)).reset_index(name=columns_ingredient_pre)
    return X

    

def CleanDataFrame(X, 
               lowercase=False,
               removestopwords=False,
               removedigit=False,
               getstemmer=False,
               getlemmatisation=False,
               columns_text=[],
               columns_binary=[],
               columns_frozen=[]
              ):
    """
    Cleans, normalizes the dataframe

    Parameters
    ----------
    X : pd.DataFrame
        Base de données.
    lowercase : String, optional
        Lettres capitales -> minuscules. The default is False.
    removestopwords : Boolean, optional
        Enlever les stop words. The default is False.
    removedigit : Boolean, optional
        Enlever les chiffres. The default is False.
    getstemmer : Boolean, optional
        Racinisation. The default is False.
    getlemmatisation : Boolean, optional
        Lemmatisation. The default is False.
    columns_text : List<String>, optional
        Variables de textes (à prétraiter). The default is [].
    columns_binary : List<String>, optional
        Variables binaires (à ne pas prétraiter). The default is [].
    columns_frozen : List<String>, optional
        Variables avec prétraitement particulier si elles apparaissent dans les variables de texte. The default is [].

    Returns
    -------
    pd.DataFrame
        Données prétraitées.

    """
    
    data_out = []
    
    # labels_binary_list = []
    # for column_binary in columns_binary:
    #     labels_binary_list.append([column_binary, X[column_binary].unique().tolist()])

    #Parcourir chaque produit
    for index, row in X.iterrows():
        new_row = []
        
        #Parcourir chaque variable de chaque produit
        for text in row[columns_text]:
            
            # suppression de tous les caractères uniques
            if removedigit:
                text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
                text = re.sub(r'(?:^| )\w(?:$| )', ' ', text)
            # substitution des espaces multiples par un seul espace
            text = re.sub(r'\s+', ' ', text, flags=re.I)

            # decoupage en mots
            tokens = text.split(sep = ' ')
            if lowercase:
                  tokens = [token.lower() for token in tokens]

            # suppression ponctuation
            table = str.maketrans('', '', string.punctuation)
            words = [token.translate(table) for token in tokens]

            # suppression des tokens non alphabetique ou numerique
            words = [word for word in words if word.isalnum()]

            # suppression des tokens numerique
            if removedigit:
                words = [word for word in words if not word.isdigit()]

            # suppression des stopwords
            if removestopwords:
                words = [word for word in words if not word in stop_words_list]

            # lemmatisation
            if getlemmatisation:
                lemmatizer=WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word)for word in words]

            # racinisation
            if getstemmer:
                ps = PorterStemmer()
                words=[ps.stem(word) for word in words]

            new_row.append(' '.join(words))
         
        label_row = row[columns_binary].to_list()
        #label_row = [words for words in row[columns_binary]]
        data_out.append(new_row + label_row)
        
    return pd.DataFrame(data_out, columns=columns_text + columns_binary, index=X.index)



from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 removestopwords=False,
                 lowercase=False,
                 removedigit=False,
                 getstemmer=False,
                 getlemmatisation=False,
                 columns_text=[],
                 columns_binary=[],
                 columns_frozen=[]
                ):
        
        self.lowercase=lowercase
        self.getstemmer=getstemmer
        self.removestopwords=removestopwords
        self.getlemmatisation=getlemmatisation
        self.removedigit=removedigit
        self.columns_text=columns_text
        self.columns_binary = columns_binary
        self.columns_frozen=columns_frozen

    def transform(self, X, **transform_params):
        # Nettoyage du texte
        X=X.copy() # pour conserver le fichier d'origine
        return CleanDataFrame(X,lowercase=self.lowercase,
                            getstemmer=self.getstemmer,
                            removestopwords=self.removestopwords,
                            getlemmatisation=self.getlemmatisation,
                            removedigit=self.removedigit,
                            columns_text=self.columns_text,
                            columns_binary=self.columns_binary,
                            columns_frozen=self.columns_frozen)

    def fit(self, X, y=None, **fit_params):
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {
            'lowercase':self.lowercase,
            'getstemmer':self.getstemmer,
            'removestopwords':self.removestopwords,
            'getlemmatisation':self.getlemmatisation,
            'removedigit':self.removedigit,
            'columns_text':self.columns_text,
            'columns_binary':self.columns_binary,
            'columns_frozen':self.columns_frozen
        }    
    
    def set_params (self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self    
