# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Diciembre/2017
Contenido:
    Ejemplo de clustering jerárquico en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering jerárquico en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://seaborn.pydata.org/generated/seaborn.clustermap.html
'''

import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import preprocessing

accidentes = pd.read_csv('accidentes_2013.csv')

subset = accidentes

#seleccionar accidentes de tipo 'colisión de vehículos'
subset = subset[subset['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]

#seleccionar variables de interés para clustering
usadas = ['TOT_VICTIMAS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]

#Para sacar el dendrograma en el jerárquico, no puedo tener muchos elementos.
#Hago un muestreo aleatorio para quedarme solo con 1000, aunque lo ideal es elegir un caso de estudio que ya dé un tamaño pequeño
X = X.sample(1000, random_state=123456)

#En clustering hay que normalizar para las métricas de distancia
X_normal = preprocessing.normalize(X, norm='l2')

#Vamos a usar este jerárquico y nos quedamos con 100 clusters, es decir, cien ramificaciones del dendrograma
ward = cluster.AgglomerativeClustering(n_clusters=100, linkage='ward')
name, algorithm = ('Ward', ward)

cluster_predict = {}
k = {}

print(name,end='')
t = time.time()
cluster_predict[name] = algorithm.fit_predict(X_normal) 
tiempo = time.time() - t
k[name] = len(set(cluster_predict[name]))
print(": k: {:3.0f}, ".format(k[name]),end='')
print("{:6.2f} segundos".format(tiempo))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
#y se añade como columna a X
X_cluster = pd.concat([X, clusters], axis=1)

#Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
min_size = 3
X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
k_filtrado = len(set(X_filtrado['cluster']))
print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k['Ward'],k_filtrado,min_size,len(X),len(X_filtrado)))
X_filtrado = X_filtrado.drop('cluster', 1)

#Normalizo el conjunto filtrado
X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')

#Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
from scipy.cluster import hierarchy
linkage_array = hierarchy.ward(X_filtrado_normal)
plt.figure(1)
plt.clf()
hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn

#Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
import seaborn as sns
X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)