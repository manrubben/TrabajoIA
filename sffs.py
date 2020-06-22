import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

def get_dataset_df(ruta, header):
    '''
    El csv con los datos debe tener en la ultima columna el atributo objetivo. El parámetro header debe ser True si el csv
    contiene header, false en caso contrario. Para que la ruta funcione correctamente, meter el csv en la carpeta docs y pasar
    la ruta de la siguiente manera: "docs/titanic.csv"
    Este metodo devuelve los datos del csv como dataframe.
    '''
    r = str(ruta)
    if header == False:
        dataset = pd.read_csv(r, header=None)
    if header == True:
        dataset = pd.read_csv(r)
    
    return dataset

def metodo_evaluacion_robusta(dataset, atributos, N_EXP, CV):

    columnas = dataset.columns.tolist()
    nombre_objetivo = columnas.pop(len(columnas)-1)
    objetivo = dataset[nombre_objetivo]

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
    #print(media)
    return media

def algoritmo_sffs(dataset):
    solucion_actual = []
    añadidos = []
    eliminados = []
    k=0

    variables_sin_añadir = dataset.columns.tolist()
    variables_sin_eliminar = dataset.columns.tolist()
    variables = dataset.columns.tolist()
    nombre_objetivo = variables_sin_añadir.pop(len(variables_sin_añadir)-1)
    nombre_objetivo2 = variables_sin_eliminar.pop(len(variables_sin_eliminar)-1)
    nombre_objetivo3 = variables.pop(len(variables)-1)
    objetivo = dataset[nombre_objetivo]
    solucion = []
    i = 1
    j=0

    while not (len(añadidos) == len(variables) and k==10):
        print('iteracion: ', k)
        
        lista_scores2 = []
        lista_sol2 = []
        if not len(añadidos) == len(variables):
            print('=====')
            lista_scores = []
            lista_sol = []
            for v in variables_sin_añadir:
                lista_sol = list(solucion_actual)
                lista_sol.append(dataset[v])
                solucion_temporal = list(lista_sol)
                solucion_temporal = np.reshape(np.ravel(lista_sol), (len(objetivo),1+j))         
                score = metodo_evaluacion_robusta(dataset, solucion_temporal, 1, 3)
                lista_scores.append(score)
                i=i+1
            mejor_promedio = np.amax(lista_scores)
            mejor_solucion_temporal = variables_sin_añadir[lista_scores.index(mejor_promedio)]      
            solucion_actual.append(dataset[mejor_solucion_temporal])
            añadidos.append(mejor_solucion_temporal)
            solucion.append(mejor_solucion_temporal)
            variables_sin_añadir.remove(variables_sin_añadir[lista_scores.index(mejor_promedio)])
        else:
            i=i-1

        if not len(solucion) == 1:

            for v in solucion:
                i=i-1
                if not variables_sin_eliminar.index(v) == 0:
                    old_index = variables_sin_eliminar.index(v)
                    variables_sin_eliminar.insert(0, variables_sin_eliminar.pop(old_index)) 
                lista_sol2 = list(solucion_actual)
                lista_sol2.remove(lista_sol2[variables_sin_eliminar.index(v)])
                solucion_temporal = list(lista_sol2)
                solucion_temporal = np.reshape(np.ravel(lista_sol2), (len(objetivo), j))
                score = metodo_evaluacion_robusta(dataset, solucion_temporal, 1, 3)
                lista_scores2.append(score)
            mejor_promedio2 = np.amax(lista_scores2)
            mejor_solucion_temporal2 = variables_sin_eliminar[lista_scores2.index(mejor_promedio2)]
            if mejor_promedio2 > mejor_promedio:
                if not solucion.index(mejor_solucion_temporal2) == 0:
                    old_index = solucion.index(mejor_solucion_temporal2)
                    solucion.insert(0, solucion.pop(old_index))
                    solucion_actual.insert(0, solucion_actual.pop(old_index))
                solucion_actual.remove(solucion_actual[solucion.index(mejor_solucion_temporal2)])
                variables_sin_eliminar.remove(mejor_solucion_temporal2)
                #solucion_actual.append(dataset[mejor_solucion_temporal2])
                #variables_sin_añadir.remove(mejor_solucion_temporal2)
                eliminados.append(mejor_solucion_temporal2)
                #añadidos.append(mejor_solucion_temporal2)
                solucion.remove(mejor_solucion_temporal2)
                #solucion.append(mejor_solucion_temporal2)
                if len(añadidos) == len(variables):
                    mejor_promedio = mejor_promedio2
                i=i-1
                j=j-1
                k=0
        if not len(añadidos) == len(variables):
            j=j+1
        k=k+1
        print('solucion: ', solucion)
        print('Añadidos: ', añadidos)
        print('Eliminados: ', eliminados)




        


dataset = get_dataset_df('docs/titanic.csv', True)
algoritmo_sffs(dataset)
#print(dataset['Pclass'])
#print(dataset.columns.get_loc('Pclass'))


