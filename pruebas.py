import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

#titanic_dataset = 'docs/titanic.csv'
#breast_cancer_dataset = 'docs/breast_cancer_dataset.csv'

#titanic = pd.read_csv(titanic_dataset) # se lee el csv y se indican el nombre de las columnas
#columnas = titanic.columns.tolist()
#print(len(columnas))

#atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
#objetivo_titanic = titanic['Survived'] # selección de la columna objetivo
#print(objetivo_titanic)


def get_dataset_df(ruta, header):
    r = str(ruta)
    if header == False:
        dataset = pd.read_csv(r, header=None)
    if header == True:
        dataset = pd.read_csv(r)
    
    return dataset

def get_atributos(dataset):
    columnas = dataset.columns.tolist()
    atributos = dataset.iloc[:, 0:len(columnas)-1]

    return atributos
    



def metodo_evaluacion_robusta(dataset, atributos, N_EXP, CV):

    columnas = dataset.columns.tolist()
    print(columnas)
    nombre_objetivo = columnas.pop(len(columnas)-1)
    objetivo = dataset[nombre_objetivo]
    #print(objetivo)

    codificador_atributos = preprocessing.OrdinalEncoder() # Codificador adecuado para los atributos
    codificador_atributos.fit(atributos) 
    atributos_codificados = codificador_atributos.transform(atributos)

    codificador_objetivo = preprocessing.LabelEncoder() # Codificador adecuado para el objetivo
    objetivo_codificado = codificador_objetivo.fit_transform(objetivo)

    # Dividimos en conjuntos de entrenamiento y prueba los atributos y el objetivo codificado
    atributos_entrenamiento, atributos_prueba, objetivo_entrenamiento, objetivo_prueba = model_selection.train_test_split(
    atributos_codificados, objetivo_codificado,  # Conjuntos de datos a dividir, usando los mismos índices para ambos
    random_state=12345,  # Valor de la semilla aleatoria, para que el muestreo sea reproducible, a pesar de ser aleatorio
    test_size=.20  # Tamaño del conjunto de prueba
    )  

    clasif_arbol_decision = tree.DecisionTreeClassifier() # creamos el clasificador
    clasif_arbol_decision.fit(X=atributos_entrenamiento, y=objetivo_entrenamiento)

    lista_promedios = []

    for i in range(N_EXP):
        #print('Iteración: ',i)
        scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos_prueba, y=objetivo_prueba, cv=CV, scoring='balanced_accuracy')
        #print('Score: ',scores)
        promedio = scores.mean()
        #print('Promedio: ',scores.mean())
        lista_promedios.append(promedio)
    
    media = sum(lista_promedios)/len(lista_promedios)
    print(media)
    return media

def algoritmo_sfs(dataset, D):
    solucion_actual = []
    solucion = []
    k=0
    variables_predictoras = dataset.columns.tolist()
    
    nombre_objetivo = variables_predictoras.pop(len(variables_predictoras)-1)
    objetivo = dataset[nombre_objetivo]
    variables_sin_añadir = variables_predictoras
    i=1
    while k < D:
        print('Variables sin añadir: ', variables_sin_añadir)
        lista_scores = []
        lista_sol = []
        for v in variables_sin_añadir:
            lista_sol = solucion_actual
            lista_sol.append(dataset[v])
            #solucion_actual.append(dataset[v])
            #print('Solucion actual: ', solucion_actual)
            solucion_temporal = lista_sol
            solucion_temporal = np.reshape(np.ravel(lista_sol), (len(objetivo),i+k))
            #print(solucion_temporal)
            score = metodo_evaluacion_robusta(dataset, solucion_temporal, 1, 3)
            lista_scores.append(score)
            i=i+1
        print('Lista de scores: ',lista_scores)
        mejor_promedio = np.amax(lista_scores)
        mejor_solucion_temporal = variables_sin_añadir[lista_scores.index(mejor_promedio)]
        print('Mejor solucion temporal: ', mejor_solucion_temporal)
        solucion_actual.append(dataset[mejor_solucion_temporal])
        solucion.append(mejor_solucion_temporal)
        print('Solucion: ', solucion)
        #solucion_actual.append(variables_sin_añadir[lista_scores.index(mejor_solucion_temporal)])
        variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_promedio)])
        #print('Solucion actual: ',solucion_actual)
        k=k+1

dataset = get_dataset_df('docs/titanic.csv', True)
atributos = get_atributos(dataset)
#algoritmo_sfs(titanic, 5)
metodo_evaluacion_robusta(dataset, atributos, 10, 10)