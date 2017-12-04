# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Octubre/2017
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
        http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
import sklearn
from math import floor

accidentes = pd.read_csv('accidentes_2013.csv')

#seleccionar accidentes de tipo 'colisión de vehículos'
#subset = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]

#seleccionar solo accidentes entre las 6 y las 12 de la mañana
subset = accidentes.loc[(accidentes['HORA']>=6) & (accidentes['HORA']<=12)]

#seleccionar accidentes no mortales
#subset = accidentes.loc[accidentes['TOT_MUERTOS']==0]

#seleccionar variables de interés para clustering
#usadas = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]

X_normal = preprocessing.normalize(X, norm='l2')

print('----- Ejecutando k-Means',end='')
k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
t = time.time()
cluster_predict = k_means.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette consume mucha RAM, se selecciona una muestra del 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.1*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

#y se añade como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)

print("---------- Preparando el scatter matrix...")
import seaborn as sns
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
sns_plot.savefig("kmeans.png")
print("")
#'''