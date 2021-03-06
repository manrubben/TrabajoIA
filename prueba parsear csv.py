import pandas as pd
from sklearn import preprocessing
import numpy as np

titanic_dataset = 'docs/titanic.csv'


titanic = pd.read_csv(titanic_dataset, header=None, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas
h = titanic.columns.tolist()
print(h)
h.pop(len(h)-1)
print(h)
atributos_titanic = titanic.loc[:, 'Pclass':'Is_Married'] # selección de las columnas de atributos
objetivo_titanic = titanic['Survived'] # selección de la columna objetivo

codificador_atributos = preprocessing.OrdinalEncoder() # Codificador adecuado para los atributos
codificador_atributos.fit(atributos_titanic) 




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

        print("\nTabla respuesta:\n")
        for j in range (0,D):
            frame_data={'solution':solucion, 'score':mejor_promedio}

    df=pd.DataFrame(frame_data)

    return df.sort_values(by=['score'],ascending = False)

algoritmo_sfs(titanic, 5)