import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

titanic_dataset = 'docs/titanic2.csv'
breast_cancer_dataset = 'docs/breast_cancer_dataset.csv'

titanic = pd.read_csv(titanic_dataset, header=None, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas

atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo
print(atributos_titanic.shape)
print(objetivo_titanic.shape)
# N_EXP = 1
# CV = 3

def metodo_evaluacion_robusta(dataset, atributos, objetivo, N_EXP, CV):

    codificador_atributos = preprocessing.OrdinalEncoder() # Codificador adecuado para los atributos
    codificador_atributos.fit(atributos) 
    atributos_codificados = codificador_atributos.transform(atributos)
    #print('Atributos codificados: ',atributos_codificados)
    print('Tamaño x: ',atributos_codificados.shape)

    codificador_objetivo = preprocessing.LabelEncoder() # Codificador adecuado para el objetivo
    objetivo_codificado = codificador_objetivo.fit_transform(objetivo)
    print('Tamaño y: ',objetivo_codificado.shape)
    
    #print(codificador_objetivo.classes_)  # Clases detectadas por el codificador para la variable objetivo
    #print('Objetivo codificado: ',objetivo_codificado)
    #print(codificador_objetivo.inverse_transform([0, 1])) #Ordena alfabéticamente

    #print(dataset.shape[0])  # Cantidad total de ejemplos
    #print(pd.Series(objetivo).value_counts(normalize=True))  # Frecuencia total de cada clase de aceptabilidad

    # Dividimos en conjuntos de entrenamiento y prueba los atributos y el objetivo codificado
    atributos_entrenamiento, atributos_prueba, objetivo_entrenamiento, objetivo_prueba = model_selection.train_test_split(
    atributos_codificados, objetivo_codificado,  # Conjuntos de datos a dividir, usando los mismos índices para ambos
    random_state=12345,  # Valor de la semilla aleatoria, para que el muestreo sea reproducible, a pesar de ser aleatorio
    test_size=.20  # Tamaño del conjunto de prueba
    )  

    clasif_arbol_decision = tree.DecisionTreeClassifier() # creamos el clasificador
    clasif_arbol_decision.fit(X=atributos_entrenamiento, y=objetivo_entrenamiento)

    for i in range(N_EXP):
        print('Iteración: ',i)
        scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos_prueba, y=objetivo_prueba, cv=CV, scoring='balanced_accuracy')
        print('Score: ',scores)
        promedio = scores.mean()
        print('Promedio: ',scores.mean())
    
    return promedio


def algoritmo_sfs(dataset, D):
    solucion_actual = []

    k=0
    variables_predictoras = dataset.columns.tolist()
    print(variables_predictoras)
    nombre_objetivo = variables_predictoras.pop(len(variables_predictoras)-1)
    objetivo = dataset[nombre_objetivo]
    while k < D:
        variables_sin_añadir = variables_predictoras#list(set(variables_predictoras) - set(solucion_actual))
        i=1
        lista_scores = []
        for v in variables_sin_añadir:
            solucion_actual.append(dataset[v])
            solucion_temporal = solucion_actual
            solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
            #print(solucion_temporal)
            score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 10)
            lista_scores.append(score)
            i=i+1
        print('Lista de scores: ',lista_scores)
        mejor_solucion_temporal = np.amax(lista_scores)
        solucion_actual.append(variables_sin_añadir[lista_scores.index(mejor_solucion_temporal)])
        variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_solucion_temporal)])
        print('Solucion actual: ',solucion_actual)
        k=k+1


        



    

#metodo_evaluacion_robusta(titanic, atributos_titanic, objetivo_titanic, 1, 10)
algoritmo_sfs(titanic, 5)
