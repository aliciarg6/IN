# Alberto Armijo Ruiz.
# 4º GII.

"""
Algoritmos utilizados en el estudio.
- KMeans.
- AgglomerativeClustering.
- MeanShift.
- Birch.
- SpectralClustering
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.cluster import KMeans, AgglomerativeClustering,estimate_bandwidth
from sklearn.cluster import Birch,SpectralClustering,MeanShift
from sklearn import metrics, preprocessing
from math import floor
import seaborn as sns
import time
from scipy.cluster.hierarchy import dendrogram,linkage
import numpy as np

sys.setrecursionlimit(1500)

t_global_start = time.time()

def makeScatterPlot(data,outputName=None,displayOutput=True):
    sns.set()
    variables = list(data)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25},
                            diag_kind="hist")  # en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)

    if outputName != None:
        outputName += ".png"
        print(outputName)
        sns_plot.savefig(outputName)
    if displayOutput:
        plt.show()

def calculateMetrics(clusterPredict,data):
    metric_CH = metrics.calinski_harabaz_score(data, clusterPredict)
    metric_SC = metrics.silhouette_score(data, clusterPredict, metric='euclidean', sample_size=floor(0.1 * len(data)),
                                        random_state=123456)

    return [metric_CH,metric_SC]

def createDataFrame(dataframe,prediction):
    cluster = pd.DataFrame(prediction,index=dataframe.index,columns=['cluster'])
    clusterX = pd.concat([dataframe,cluster],axis=1)

    return clusterX

def createPrediction(dataframe,data,model):
    time_start = time.time()
    cluster_predict = model.fit_predict(data)
    time_finish = time.time() - time_start

    X_dataFrame = createDataFrame(dataframe,cluster_predict)

    return [calculateMetrics(cluster_predict,data),X_dataFrame,time_finish,cluster_predict]

def makeDendogram(data,predict,outputName=None, displayOutput=True):
    vars = list(data)
    vars.remove("cluster")
    dataFrame = data[vars]
    Z = linkage(dataFrame, 'ward')
    plt.title("Dendograma")
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='lastp', p=12, show_contracted=True, labels=predict)

    if outputName != None:
        outputName += "_dendogram.png"
        plt.savefig(outputName)
    if displayOutput:
        plt.show()

def makeHeatMap(data,clusters,outputName=None, displayOutput=True):
    sns.set()
    vars = list(data)
    vars.remove('cluster')
    table = data[vars].T
    sum_row = {row:np.sum(table[row]) for row in table}
    sum_df = pd.DataFrame(sum_row, index=["Suma"])
    sum_df = sum_df.T
    table = createDataFrame(dataframe=sum_df,prediction=clusters)
    table = pd.pivot_table(table,index=table.index,columns="cluster",values="Suma",fill_value=0)
    print(table)
    heatmap = sns.heatmap(table.T,xticklabels="auto", yticklabels="auto")
    plt.yticks(rotation=0)

    # Guardamos el archivo con el nombre especificado + terminación en heatmap.
    if outputName != None:
        outputName += "_heatmap.png"
        heatmap.savefig(outputName)
    if displayOutput:
        plt.show()

def makeClusterMap(data,clusters,outputName=None,displayOutput=True):
    sns.set()
    vars = list(data)
    vars.remove('cluster')
    table = data[vars].T
    sum_row = {row: np.sum(table[row]) for row in table}
    sum_df = pd.DataFrame(sum_row, index=["Suma"])
    sum_df = sum_df.T
    table = createDataFrame(dataframe=sum_df, prediction=clusters)
    table = pd.pivot_table(table, index=table.index, columns="cluster", values="Suma", fill_value=0)
    print(table)
    clustermap = sns.clustermap(table, xticklabels="auto", yticklabels="auto")
    plt.yticks(rotation=0)

    if(outputName != None):
        outputName += "clustermap.png"
        clustermap.savefig(outputName)

    if displayOutput:
        plt.show()

accidentes = pd.read_csv('accidentes_2013.csv')
subset = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]
n = 500

X  = X.sample(n,random_state=123456)
X_normal = preprocessing.normalize(X, norm='l2')

k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
aglo=AgglomerativeClustering(n_clusters=40,linkage="ward")
bandwidth = estimate_bandwidth(X_normal, quantile=0.2, random_state=123456)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
birch=Birch(n_clusters=6,threshold=0.1)
spectral=SpectralClustering(n_clusters=4,eigen_solver="arpack")


clustering_algorithms = (
    ("K-medias",k_means),
    ("AC",aglo),
    ("Mean Shift",meanshift),
    ("Birch",birch),
    ("SpectralC",spectral)
)

met, clusterFrame, timeAlg, cluster_predict = createPrediction(dataframe=X, data=X_normal, model=k_means)
# makeScatterPlot(data=clusterFrame,displayOutput=True)
# makeDendogram(data=clusterFrame,predict=cluster_predict, outputName="agglomerativeclustering")
makeHeatMap(data=clusterFrame, clusters=cluster_predict)
makeClusterMap(data=clusterFrame, clusters=cluster_predict)

# Creamos los datos de salida, y mostramos las gráficas si queremos.

# outputData = dict()
# for algorithm_name,algorithm in clustering_algorithms:
#     results = dict()
#     met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
#     n_clusters=len(set(cluster_predict))
#     #makeScatterPlot(data=clusterFrame, outputName=algorithm_name, displayOutput=True)
#
#     results['n_clusters']=n_clusters
#     results['sc_metric']=met[0]
#     results['hc_metric']=met[1]
#     results['time']=timeAlg
#
#     outputData[algorithm_name] = results
#
# print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
#     'Name', 'N clusters', 'SC metric', 'HC metric', 'Time (s)'))
#
# for name, res in outputData.items():
#     print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
#         name, res['n_clusters'], res['sc_metric'], res['hc_metric'],
#         res['time']))
#
# print('\nTotal time = {0:.2f}'.format(time.time() - t_global_start))

