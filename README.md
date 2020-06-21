# Seleccion de características en modelos predictivos

## Versiones utilizadas:

python3
pandas 1.0.4
numpy 1.18.5

## Método de evaluación robusta: Calcular rendimiento

score = metodo_evaluacion_robusta(titanic, atributos_titanic, objetivo_titanic, 10, 10)

### Los parámetros que recibe son:

1. Ruta del dataset a usar. En el caso de que se quieran usar nuevos datasets, se recomienda meterlos en la carpeta "docs".
2. El subconjunto que se va a evaluar.
3. La columna del dataset que contiene la variable respuesta.
4. Número de repeticiones del experimento de validación cruzada.
5. Número de folds a considerar en la validación cruzada.

## Algoritmo Sequential Forward Selection (SFS)

algoritmo_sfs(cancer, 5)

### Los parámetros que recibe son:

1. Ruta del dataset a usar. En el caso de que se quieran usar nuevos datasets, se recomienda meterlos en la carpeta "docs".
2. Número máximo de variables a probar.

## Algoritmo Sequential Floating Forward Selection (SFFS)

algoritmo_sffs(titanic)

### Los parámetros que recibe son:

1. Ruta del dataset a usar. En el caso de que se quieran usar nuevos datasets, se recomienda meterlos en la carpeta "docs".
