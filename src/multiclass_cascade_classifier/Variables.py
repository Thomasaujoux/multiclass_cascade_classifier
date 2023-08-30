"""
Module Variables

Variables

@author: ThomasAujoux
"""

### Champs

# Input

id_produit = "Code_produit"
id_famille = "Famille"
predicted_famille = "predicted_famille"
id_secteur = "Secteur"
predicted_secteur = "predicted_secteur"
#id_marque = "id_marque"
libel = "Nom"
denomination_vente = "Denomination_de_vente"
#libel_precision = "precision_produit"
id_mode_conservation = "Conservation"
ingredient = "Ingredient"



# Champs (combination)

columns = [id_produit, id_famille, id_secteur, libel, denomination_vente, id_mode_conservation, ingredient]
column_index = [id_produit]
columns_all= [id_famille, id_secteur, libel, denomination_vente, id_mode_conservation, ingredient]
column_X_all = [libel, denomination_vente, id_mode_conservation, ingredient]
columns_X = [libel, denomination_vente, id_mode_conservation, ingredient]

# Binary features value

binary_features = {
    id_mode_conservation: ["frai", "ambiant", "surgele"],
}

columns_label = [id_secteur, id_famille]
columns_label_all  = [id_secteur, id_famille, predicted_secteur, predicted_famille]
columns_text = [libel, denomination_vente, ingredient]
columns_bin = [id_mode_conservation]
#columns_frozen = [id_marque]


### Prétraitement

lowercase = True
removestopwords = True
removedigit = True
getstemmer = True
getlemmatisation = True



### Classifier

probability = "probability"
probabilityValue = True

hyper_sector_yaml = "hyper-sector.yaml"
hyper_families_yaml = "hyper-family.yaml"

confusion_matrix = "confusion_matrix"
confusion_matrix_sector = confusion_matrix + "_secteur.xlsx"
confusion_matrix_family = confusion_matrix + "_famille.xlsx"

classification_report = "classification_report"
classification_report_sector = classification_report + "_secteur.xlsx"
classification_report_family = classification_report + "_famille.xlsx"

classifierType = "type"
classifierHyperParams = "hyperparameters"
classifierCVMean = "score"
secteur = "secteur"
famille = "famille"

SVM = "SVM"
RF = "RandomForest"

hyperParamsGrid = {
    SVM: {
    "kernel": ["linear", "rbf", "sigmoid"],
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [10, 1, 0.1, 0.01, 0.001, 0.0001],
    "probability": [True,],
    },
    RF: {
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    },
}

cv = 5
n_jobs = 10

training_date = "training_date"
training_size = "training_size"

time_= "--- %02d minutes et %02d secondes ---"



# Number of binary features

nb_bin_features = 0
for bin_column in binary_features.keys():
    nb_bin_features += len(binary_features[bin_column])