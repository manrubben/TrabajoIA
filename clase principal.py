import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

titanic_dataset = 'docs/titanic.csv'
breast_cancer_dataset = 'docs/breastCancerDataset.csv'

titanic = pd.read_csv(titanic_dataset, header=0, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 
'Age_band', 'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y 
                                                                                            #se indican el nombre de las columnas

cancer = pd.read_csv(breast_cancer_dataset, header=0, delimiter=',', names=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 
'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 
'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 
'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension', 'diagnosis'])

atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos del dataset titanic
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo del dataset titanic

atributos_cancer = cancer.loc[:, 'mean radius':'worst fractal dimension']
objetivo_cancer = cancer['diagnosis']
#print(len(objetivo_cancer))


#print(atributos_titanic.shape)
#print(objetivo_titanic)
# N_EXP = 1
# CV = 3

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
            score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
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


def algoritmo_sffs(dataset):
    solucion_actual = []
    añadidos = []
    eliminados = []
    k=0
    lista_auxiliar_añadidos = []
    variables_predictoras = dataset.columns.tolist()
    #print('Variables predictoras iniciales: ', variables_predictoras)
    nombre_objetivo = variables_predictoras.pop(len(variables_predictoras)-1)
    #print('Variables predictoras sin objetivo: ', variables_predictoras)
    variables_sin_añadir = dataset.columns.tolist()
    nombre_objetivo2 = variables_sin_añadir.pop(len(variables_sin_añadir)-1)
    #print('variables sin añadir: ', variables_sin_añadir)
    
    variables = dataset.columns.tolist()
    nombre_objetivo3 = variables.pop(len(variables)-1)
    objetivo = dataset[nombre_objetivo]
    solucion = []
    while not (len(añadidos)==len(variables) and k==10): #La condicion de parada es que añadidos contenga todas las variables predictoras
        print('iteracion: ', k)
        i=len(solucion_actual)+1
        lista_scores = []
        variables_sin_eliminar = dataset.columns.tolist()
        nombre_objetivo3 = variables_sin_eliminar.pop(len(variables_sin_eliminar)-1)
        print('Añadidos: ', añadidos)
        print('Eliminados: ', eliminados)
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
        variables_sin_añadir.remove(mejor_solucion_temporal)
        i=i-1
        
        for v in variables_predictoras: #hay que recorrer todas las variables que no esten en eliminados
            i=i-1
            print('index: ', variables_sin_eliminar.index(v))
            if not variables_sin_eliminar.index(v) == 0:
                old_index = variables_sin_eliminar.index(v)
                variables_sin_eliminar.insert(0, variables_sin_eliminar.pop(old_index))
            solucion_actual.remove(solucion_actual[variables_sin_eliminar.index(v)])
            variables_sin_eliminar.remove(v)
            solucion_temporal = solucion_actual
            if len(solucion_actual) > 0:
                solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,i))
                score = metodo_evaluacion_robusta(dataset, solucion_temporal, objetivo, 1, 3)
                lista_scores.append(score)
            
        mejor_promedio2 = np.amax(lista_scores)
        mejor_solucion_temporal2 = variables_sin_añadir[lista_scores.index(mejor_promedio)]
        if mejor_promedio2 > mejor_promedio:
            print('======')
            #solucion_actual.append(dataset[mejor_solucion_temporal2])
            lista_auxiliar_añadidos.append(mejor_solucion_temporal2)
            variables_sin_añadir.remove(mejor_solucion_temporal2)
            eliminados.append(mejor_solucion_temporal)
            variables_predictoras.remove(mejor_solucion_temporal)
            #variables_sin_añadir.remove(mejor_solucion_temporal)
            mejor_solucion_temporal = mejor_solucion_temporal2
            k=0
        else:
            lista_auxiliar_añadidos.append(mejor_solucion_temporal)
            #solucion_actual.append(dataset[mejor_solucion_temporal])

        for la in lista_auxiliar_añadidos:
            solucion_actual.append(dataset[la])
            

        k=k+1
        print('Mejor solucion temporal 2: ', mejor_solucion_temporal)

        



    

#metodo_evaluacion_robusta(cancer, atributos_cancer, objetivo_cancer, 10, 10)
#algoritmo_sfs(cancer, 5)
#algoritmo_sffs(titanic)
