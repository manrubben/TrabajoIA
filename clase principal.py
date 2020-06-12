import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

titanic_dataset = 'docs/titanic.csv'
breast_cancer_dataset = 'docs/breast_cancer_dataset.csv'

titanic = pd.read_csv(titanic_dataset, header=None, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas

atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo
# N_EXP = 1
# CV = 3

def metodo_evaluacion_robusta(dataset, atributos, N_EXP, CV):

    iterator = dataset.itertuples() # creamos el iterador de tuplas
    next(iterator) # nos saltamos la primer fila
    for row in iterator: # recorremos cada fila del csv
        for e in row: # recorremos cada elemento de la fila
            float(e) # pasamos el elemento a float
    
    codificador_atributos = preprocessing.OrdinalEncoder()
    codificador_atributos.fit(atributos)

    clasif_arbol_decision = tree.DecisionTreeClassifier()
    clasif_arbol_decision.fit(X=dataset, y=atributos)

    for i in range(N_EXP):
        print('Iteración: '+i)
        scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos, cv=CV, scoring='balanced_accuracy')
        print(scores)

metodo_evaluacion_robusta(titanic, atributos_titanic, 1, 3)
