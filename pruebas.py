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



a = []
v = 'Pclass'
h = titanic[v]
print(h)
print(h.shape)

a.append(h)
print(a)


codificador_atributos = preprocessing.OrdinalEncoder() # Codificador adecuado para los atributos
codificador_atributos.fit(a) 
atributos_codificados = codificador_atributos.transform(a)
atributos_codificados.reshape(891,1)
  
array = h 
print("Original array : \n", array) 
  
# shape array with 2 rows and 4 columns 
#array2 = np.arange(array).reshape(891, 1) 
#print("\narray reshaped with 2 rows and 4 columns : \n", array2) 

print(np.reshape(np.ravel(h), (891,1)))

#solucion_temporal = np.reshape(np.ravel(solucion_actual), (891,1)) No borrar


#data = [4,5]
#data.append([5,6])
#print(data)

