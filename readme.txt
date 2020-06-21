# Seleccion de características en modelos predictivos

## Versiones utilizadas:

python3
pandas 1.0.4
numpy 1.18.5

## Pasar los datos del csv a un DataFrame

dataset = get_dataset_df('docs/titanic.csv', True)

### Los parámetros que recibe son:

1. Ruta del csv a usar. En caso de que se quieran usar nuevos datasets, se recomienda meterlos en la carpeta docs
2. Boolean que indica si el csv contiene header o no. Si contiene header el boolean sera True, en caso contrario False

## Obtener los atributos del dataframe

atributos = get_atributos(dataset)

### Los parámetros que recibe son: 

1. DataFrame a usar

## Método de evaluación robusta: Calcular rendimiento

score = metodo_evaluacion_robusta(titanic, atributos_titanic, 10, 10)

### Los parámetros que recibe son:

1. DataFrame a usar
2. El subconjunto que se va a evaluar.
3. Número de repeticiones del experimento de validación cruzada.
4. Número de folds a considerar en la validación cruzada.

## Algoritmo Sequential Forward Selection (SFS)

algoritmo_sfs(cancer, 5)

### Los parámetros que recibe son:

1. DataFrame a usar
2. Número máximo de variables a probar.

## Algoritmo Sequential Floating Forward Selection (SFFS)

algoritmo_sffs(titanic)

### Los parámetros que recibe son:

1. DataFrame a usar

