import pandas as pd

titanic_dataset = 'docs/titanic.csv'


titanic = pd.read_csv(titanic_dataset, header=None, delimiter=',', names=['Pclass', 'sex', 'age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 
'Family_size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived']) # se lee el csv y se indican el nombre de las columnas

for row in titanic.itertuples():
    print(row)