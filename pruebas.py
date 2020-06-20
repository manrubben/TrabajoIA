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

lista = []
lista1 = ['a', 'b', 'c']
lista2 = ['d', 'e', 'f']
lista.append(lista1)
lista.append(lista2)
print(lista)




def metodo_evaluacion_robusta(dataset, atributos, objetivo, N_EXP, CV):

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

    for i in range(N_EXP):
        #print('Iteración: ',i)
        scores = model_selection.cross_val_score(estimator=clasif_arbol_decision, X=atributos_prueba, y=objetivo_prueba, cv=CV, scoring='balanced_accuracy')
        #print('Score: ',scores)
        promedio = scores.mean()
        #print('Promedio: ',scores.mean())
    
    return promedio

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
    nombre_objetivo3 = variables_sin_eliminar.pop(len(variables_sin_eliminar)-1)
    
    objetivo = dataset[nombre_objetivo]
    solucion = []
    while not (len(añadidos)==len(variables_predictoras) and k==10): #La condicion de parada es que añadidos contenga todas las variables predictoras
        lista_auxiliar = dataset.columns.tolist()
        nombre_objetivo_auxiliar = lista_auxiliar.pop(len(variables_sin_eliminar)-1)
        i=len(solucion_actual)+1
        lista_scores = []
        
        
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
        
        solucion.append(mejor_solucion_temporal)
        añadidos.append(mejor_solucion_temporal)
        
        variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_promedio)])
        
        i=i-1
        
        for v in variables_sin_eliminar:
            i=i-1
            print('index: ', lista_auxiliar.index(v))
            solucion_actual.remove(solucion_actual[lista_auxiliar.index(v)])
            lista_auxiliar.remove(v)
            solucion_temporal = solucion_actual

            if len(solucion_actual) > 0:
                solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
                score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
                lista_scores.append(score)

        mejor_promedio2 = np.amax(lista_scores)
        if mejor_promedio2 > mejor_promedio:
            solucion_actual.append(dataset[mejor_solucion_temporal2])
            eliminados.append(mejor_solucion_temporal)
            mejor_solucion_temporal2 = variables_sin_eliminar[lista_scores.index(mejor_promedio2)]
            variables_sin_eliminar.remove(mejor_solucion_temporal)
            mejor_solucion_temporal = mejor_solucion_temporal2
            k=0
        else:
            solucion_actual.append(dataset[mejor_solucion_temporal])

        



    

#metodo_evaluacion_robusta(titanic, atributos_titanic, objetivo_titanic, 1, 10)
#algoritmo_sfs(titanic, 5)
#algoritmo_sffs(titanic)