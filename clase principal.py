import pandas as pd
from sklearn import model_selection
from sklearn import tree

titanic_dataset = 'docs/titanic.csv'
breast_cancer_dataset = 'docs/breast_cancer_dataset.csv'

titanic = pd.read_csv(titanic_dataset, header=None, names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas

atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo
N_EXP = 1
CV = 3
print(objetivo_titanic)

def metodo_evaluacion_robusta(titanic_dataset, atributos_titanic, N_EXP, CV):
    for i in range(N_EXP):
        scores = model_selection.cross_val_score(estimator=tree, X=atributos_titanic, cv=CV, scoring='balanced_accuracy')
        print(scores)

print(metodo_evaluacion_robusta)
