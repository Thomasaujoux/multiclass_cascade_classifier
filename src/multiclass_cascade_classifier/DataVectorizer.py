"""
Module DataVectorizer

Data vectorization class

@author: ThomasAujoux
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import Variables as var



# # ################## Tests ####################
# csv_in = "C:/Users/Thomas Aujoux/Documents/GitHub/package/src/multiclass_cascade_classifier/data/merged_final.csv"
# X = get_dataframe(csv_in)


# columns_text = ["Nom", "Denomination_de_vente", "Ingredient"]
# columns_binary=["Conservation"]
# columns_frozen=[]
# columns_ingredient_pre = "Ingredient"
# X = CleanColumns(X, columns_text,
#                  columns_binary_pre = "Nom",
#                  columns_ingredient_pre = "Ingredient")


# columns_text = ["Nom", "Denomination_de_vente", "Ingredient"]
# columns_binary=["Conservation"]
# columns_frozen=[]
# X = CleanDataFrame(X, True,True,True,True,True,columns_text,columns_binary,columns_frozen)

# columns_text=["Denomination_de_vente", "Nom", "Ingredient"]
# columns_binary=["Conservation"]
# id_mode_conservation = "Conservation"
# binary_features = {
#     id_mode_conservation: ["frais", "ambiant", "surgele"],
# }


# X.to_csv("C:/Users/Thomas Aujoux/Documents/GitHub/package/src/multiclass_cascade_classifier/data/merged_final2.csv")
# # ################## Tests ####################



class DataVectorizer():
    def __init__(self,
                 columns_text=[],
                 columns_binary=[]
                 ):
        
        self.columns_text=columns_text
        self.columns_binary=columns_binary
        
        self.TfidfVectorizer_text = TfidfVectorizer(binary=False, norm='l2',
            use_idf=True, smooth_idf=True,
            min_df=1, max_features=500, 
            ngram_range=(1, 2))
        
        self.TfidfVectorizer_binary = { }
        
        for column in self.columns_binary:
            self.TfidfVectorizer_binary[column] = TfidfVectorizer(
                binary=True, norm=None,
                use_idf=False, smooth_idf=False,
                min_df=1, max_features=None, ngram_range=(1, 1))
        
    def fit_transform(self, X):
        
        X["new"] = ""
        for columns in self.columns_text:
            print(X[columns])
            X["new"] = X["new"] + " " + str(X[columns])
            X.drop([columns], axis=1)
        print(X, "changements X")
        data_text_vect = self.TfidfVectorizer_text.fit_transform(X["new"]).toarray().tolist()

        #print(data_text_vect, "changmeent le reste")
        data_bin_vect = { }
        for column_bin in self.columns_binary:
            data_bin_vect[column_bin] = self.TfidfVectorizer_binary[column_bin].fit_transform(X[column_bin]).toarray().tolist()
        
        X_vect_text = pd.DataFrame(data_text_vect, columns=self.TfidfVectorizer_text.get_feature_names_out())
        X_vect_text.set_index(X.index, inplace=True)
        
        X_vect_binary = { }
        for column_bin in self.columns_binary:
            X_vect_bin = pd.DataFrame(data_bin_vect[column_bin], columns=self.TfidfVectorizer_binary[column_bin].get_feature_names_out())
            X_vect_bin.set_index(X.index, inplace=True)
            X_vect_binary[column_bin] = X_vect_bin

        # Checking that all binary features are vectorized
        for column_bin in var.binary_features:
            if column_bin in self.columns_binary:
                x_features = X_vect_binary[column_bin].copy(deep=True)
                for feature in var.binary_features[column_bin]:
                    if feature not in x_features.columns.to_list():
                        x_features[feature] = 0
            X_vect_binary[column_bin] = x_features[var.binary_features[column_bin]]
            
        data_vect = []
        for index, row in X_vect_text.iterrows():
            X_row = X_vect_text.loc[index].values.tolist()
            for column_bin in self.columns_binary:
                X_row += X_vect_binary[column_bin].loc[index].values.tolist()
            data_vect.append(X_row)
                
        X_columns = X_vect_text.columns.to_list()
        for column_bin in self.columns_binary:
            X_columns += X_vect_binary[column_bin].columns.to_list()
        X_vect = pd.DataFrame(data_vect, columns=X_columns, index=X.index)
        
        print('--- Base ---')
        print('Nombre de produits : ', str(X_vect.shape[0]))
        print('Nombre de mots : ', str(X_vect.shape[1]))
        print('--- ---- ---')
        
        return X_vect
    
    def get_params(self, deep=True):
        return {
            "columns_text": self.columns_text,
            "columns_binary": self.columns_binary,
            "features": self.features, # Where does it come from ??????
        }    
    
    def set_params (self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self  
    

