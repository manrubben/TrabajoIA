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
    #print('Tamaño x: ',atributos_codificados.shape)

    codificador_objetivo = preprocessing.LabelEncoder() # Codificador adecuado para el objetivo
    objetivo_codificado = codificador_objetivo.fit_transform(objetivo)
    #print('Tamaño y: ',objetivo_codificado.shape)
    
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
        #print('Iteración: ',i)
        scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos_prueba, y=objetivo_prueba, cv=CV, scoring='balanced_accuracy')
        #print('Score: ',scores)
        promedio = scores.mean()
        #print('Promedio: ',scores.mean())
    
    return promedio


def algoritmo_sfs(dataset, D):
    solucion_actual = []
    solucion = []
    k=0
    variables_predictoras = dataset.columns.tolist()
    
    nombre_objetivo = variables_predictoras.pop(len(variables_predictoras)-1)
    objetivo = dataset[nombre_objetivo]
    variables_sin_añadir = variables_predictoras
    while k < D:
        #variables_sin_añadir = list(set(variables_predictoras) - set(solucion_actual))
        print('Variables sin añadir: ', variables_sin_añadir)
        i=1
        lista_scores = []
        for v in variables_sin_añadir:
            solucion_actual.append(dataset[v])
            #print('Solucion actual: ', solucion_actual)
            solucion_temporal = solucion_actual
            solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
            #print(solucion_temporal)
            score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
            lista_scores.append(score)
            i=i+1
        print('Lista de scores: ',lista_scores)
        mejor_promedio = np.amax(lista_scores)
        mejor_solucion_temporal = variables_sin_añadir[lista_scores.index(mejor_promedio)]
        print('Mejor solucion temporal: ', mejor_solucion_temporal)
        solucion_actual = []
        solucion.append(mejor_solucion_temporal)
        print('Solucion: ', solucion)
        #solucion_actual.append(variables_sin_añadir[lista_scores.index(mejor_solucion_temporal)])
        variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_promedio)])
        #print('Solucion actual: ',solucion_actual)
        k=k+1
    
    return solucion


def algoritmo_sffs(dataset):
    solucion_actual = []
    añadidos = []
    eliminados = []
    k=0

    variables_predictoras = dataset.columns.tolist()
    #print('Variables predictoras iniciales: ', variables_predictoras)
    nombre_objetivo = variables_predictoras.pop(len(variables_predictoras)-1)
    #print('Variables predictoras sin objetivo: ', variables_predictoras)
    variables_sin_añadir = dataset.columns.tolist()
    nombre_objetivo2 = variables_sin_añadir.pop(len(variables_sin_añadir)-1)
    #print('variables sin añadir: ', variables_sin_añadir)
    variables_sin_eliminar = dataset.columns.tolist()
    objetivo = dataset[nombre_objetivo]
    solucion = []
    while not (len(añadidos)==len(variables_predictoras) and k==10): #La condicion de parada es que añadidos contenga todas las variables predictoras
        i=len(solucion_actual)+1
        lista_scores = []
        print('Añadidos: ', añadidos)
        print('Variables sin añadir: ', variables_sin_añadir)
        for v in variables_sin_añadir:
            solucion_actual.append(dataset[v])
            solucion_temporal = solucion_actual
            solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
            score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
            lista_scores.append(score)
            i=i+1
        mejor_promedio = np.amax(lista_scores)
        print('Mejor promedio: ', mejor_promedio)
        mejor_solucion_temporal = variables_sin_añadir[lista_scores.index(mejor_promedio)]
        print('Mejor solucion temporal: ', mejor_solucion_temporal)
        #solucion_actual = []
        solucion.append(mejor_solucion_temporal)
        añadidos.append(mejor_solucion_temporal)
        #print('variables predictoras 2: ', variables_predictoras)
        variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_promedio)])
        #print('variables predictoras 3: ', variables_predictoras)
        i=i-1
        
        for v in variables_predictoras: #hay que recorrer todas las variables que no esten en eliminados
            print('Variables predictoras: ', variables_predictoras)
            i=i-1
            print('Variable: ', v)
            #print('indice: ', variables_sin_eliminar.index(v))
            #print('Solucion actual: ', solucion_actual)
            solucion_actual.remove(solucion_actual[variables_sin_eliminar.index(v)])
            variables_sin_eliminar.remove(v)
            solucion_temporal = solucion_actual
            #print('Solucion actual: ', solucion_actual)
            if len(solucion_actual) > 0:
                solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
                score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
                lista_scores.append(score)
            
        mejor_promedio2 = np.amax(lista_scores)
        mejor_solucion_temporal2 = variables_sin_añadir[lista_scores.index(mejor_promedio)]
        if mejor_promedio2 > mejor_promedio:
            print('======')
            solucion_actual.append(dataset[mejor_solucion_temporal2])
            eliminados.append(mejor_solucion_temporal)
            mejor_solucion_temporal = mejor_solucion_temporal2
            k=0
        else:
            solucion_actual.append(dataset[mejor_solucion_temporal])

        k=k+1
        print('Mejor solucion temporal 2: ', mejor_solucion_temporal)

        



    

#metodo_evaluacion_robusta(titanic, atributos_titanic, objetivo_titanic, 1, 10)
#algoritmo_sfs(titanic, 5)
algoritmo_sffs(titanic)
