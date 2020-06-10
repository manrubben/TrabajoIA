import pandas as pd

titanic_dataset = 'docs/titanic.csv'
breast_cancer_dataset = 'docs/breast_cancer_dataset.csv'

def xls_reader(file):
    read_file = pd.read_csv(file)
    return read_file