import pandas as pd
from sklearn import preprocessing
import numpy as np

titanic_dataset = 'docs/titanic.csv'


titanic = pd.read_csv(titanic_dataset, header=None, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas

atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo

codificador_atributos = preprocessing.OrdinalEncoder() # Codificador adecuado para los atributos
codificador_atributos.fit(atributos_titanic) 

variables_predictoras = ['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
                            'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married']

v = 'Pclass'
columna = titanic[v]
print(columna)
#for v in variables_predictoras:
#    print(v)

#itertitanic = titanic.itertuples()
#next(itertitanic)
#for row in itertitanic:
#    for e in row:
 #       print(e)



#atributos_entrenamiento, atributos_prueba, objetivo_entrenamiento, objetivo_prueba = model_selection.train_test_split(
#    atributos_codificados, objetivo_codificado,  # Conjuntos de datos a dividir, usando los mismos índices para ambos
#    random_state=13465,  # Valor de la semilla aleatoria, para que el muestreo sea reproducible, a pesar de ser aleatorio
#    test_size=0.2,  # Tamaño del conjunto de prueba
#    stratify=objetivo_codificado)  # Estratificamos respecto a la distribución de valores en la variable objetivo

   # clasif_arbol_decision = tree.DecisionTreeClassifier()
   # clasif_arbol_decision.fit(X=atributos_entrenamiento, y=objetivo_entrenamiento)

   # for i in range(N_EXP):
    #    print('Iteración: '+i)
    #    scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos, cv=CV, scoring='balanced_accuracy')
    #    print(scores)


#for i in range(10):
#    print(i)