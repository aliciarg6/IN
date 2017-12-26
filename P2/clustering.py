'''
@author: Alberto Armijo Ruiz.
@description:
    Práctica 2 Inteligencia de Negocio.
    Algoritmos de clustering.
    4º GII
'''

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
from scipy.cluster.hierarchy import dendrogram,ward
import numpy as np

sys.setrecursionlimit(1500)

def makeScatterPlot(data,outputName=None,displayOutput=True):
    sns.set()
    variables = list(data)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25},
                            diag_kind="hist")  # en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)


    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += ".png"
        print(outputName)
        plt.savefig(outputName)
        plt.clf()


def transformToLatexTable(data,fileName,header=[]):
    f = open(fileName,'w')

    for col in header[0:len(header)-1]:
        f.write(col+'&')

    f.write(header[-1])
    f.write('\\\\ \n')

    for name, res in data.items():
        f.write(name+'&')
        keys = list(res.keys())
        for item in keys[:len(keys)-1]:
            if (type(res[item]) == int):
                f.write('{:10}&'.format(res[item]))
            else:
                f.write('{:.2f}&'.format(res[item]))


        if(type(res[keys[-1]])==int):
            f.write('{:10}&'.format(res[keys[-1]]))
        else:
            f.write('{:.2f}'.format(res[keys[-1]]))


        f.write("\\\\ \n")

    f.close()


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

def calculateMeanDictionary(cluster,cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.mean(cluster[vars],axis=0))

def calculateDeviationDictionary(cluster, cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.std(cluster[vars],axis=0))

def createMeanClusterDF(dataFrame, clusterCol = 'cluster'):
    n_clusters = list(set(dataFrame[clusterCol]))

    my_mean_df = pd.DataFrame()
    my_deviation_df = pd.DataFrame()

    for cluster_n in n_clusters:
        my_cluster = dataFrame[dataFrame[clusterCol] == cluster_n]
        meanDic = calculateMeanDictionary(cluster=my_cluster,cluster_col = clusterCol)
        deviationDic = calculateDeviationDictionary(cluster=my_cluster, cluster_col = clusterCol)
        stdDF = pd.DataFrame(deviationDic, index=[str(cluster_n)])
        auxDF = pd.DataFrame(meanDic,index=[str(cluster_n)])
        my_mean_df = pd.concat([my_mean_df,auxDF])
        my_deviation_df = pd.concat([my_deviation_df,stdDF])

    return [my_mean_df, my_deviation_df]

def findIndex(iterable, value):
    index = -1

    for val in iterable:
        if(value == val):
            return index

    return index

def createNormalizedDF(dataFrame):
    vars = list(dataFrame)
    if(findIndex(vars,'cluster') != -1):
        vars.remove('cluster')

    norm = preprocessing.normalize(dataFrame,norm='l2')
    df = pd.DataFrame(norm,columns=vars, index=dataFrame.index)

    return df

def createLatexDataFrame(data):
    my_index = list(dict(data.items()).keys())
    my_data = list(data.values())
    my_cols = list(my_data[0].keys())
    latexDF = pd.DataFrame()

    for row in range(len(my_index)):
        aux = pd.DataFrame(data=my_data[row],index=[my_index[row]],columns=my_cols)
        latexDF = pd.concat([latexDF,aux])

    return latexDF

def makeHeatmap(data,displayOutput=True,outputName=None):
    meanDF, stdDF = createMeanClusterDF(dataFrame=data)
    meanDF = createNormalizedDF(dataFrame=meanDF)
    anotations = True
    hm = sns.heatmap(data=meanDF, linewidths=.1, cmap='Blues_r', annot=anotations, xticklabels='auto')
    plt.xticks(rotation=0)

    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += '.png'
        print(outputName)
        plt.savefig(outputName)
        plt.clf()


accidentes = pd.read_csv('accidentes_2013.csv')

#-----------------------------------------------------------------------------------------------------------------------
# CASO DE USO 1.
#-----------------------------------------------------------------------------------------------------------------------
t_global_start = time.time()
print("Caso de uso: accidentes dónde hay colisón entre vehículos")

subset = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]
subset = accidentes[~accidentes['ISLA'].str.contains("NO_ES_ISLA")]
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]
print(X.shape[0])
X_normal = preprocessing.normalize(X, norm='l2')

k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
aglo=AgglomerativeClustering(n_clusters=38,linkage="ward")
bandwidth = estimate_bandwidth(X_normal, quantile=0.2, random_state=123456)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
birch=Birch(n_clusters=6,threshold=0.1)
spectral=SpectralClustering(n_clusters=4,eigen_solver="arpack")


clustering_algorithms = (
    ("K-medias",k_means),
    ("AC",aglo),
    ("MeanShift",meanshift),
    ("Birch",birch),
    ("SpectralC",spectral)
)


# Creamos los datos de salida, y mostramos las gráficas si queremos.
outputData = dict()
min_size = 5
for algorithm_name,algorithm in clustering_algorithms:
    results = dict()
    met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
    n_clusters=len(set(cluster_predict))

    if( n_clusters > 15 ):
        X_filtrado = clusterFrame[clusterFrame.groupby('cluster').cluster.transform(len) > min_size]
    else:
        X_filtrado = clusterFrame

    makeScatterPlot(data=X_filtrado,outputName="./imagenes/scatterMatrix_caso1_" +algorithm_name,
                    displayOutput=False)

    makeHeatmap(data=X_filtrado,outputName="./imagenes/heatmap_caso1_"+algorithm_name,
                displayOutput=False)

    results['N Clusters']=n_clusters
    results['HC metric']=met[0]
    results['SC metric']=met[1]
    results['Time']=timeAlg

    outputData[algorithm_name] = results

latexCaso1 = createLatexDataFrame(data=outputData)

f = open('caso1.txt','w')
f.write(latexCaso1.to_latex())
f.close()

print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
    'Name', 'N clusters', 'HC metric', 'SC metric', 'Time(s)'))

for name, res in outputData.items():
    print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
        name, res['N Clusters'], res['HC metric'], res['SC metric'],
        res['Time']))

print('\nTotal time = {0:.2f}'.format(time.time() - t_global_start))

#-----------------------------------------------------------------------------------------------------------------------
# CASO DE USO 2.
#-----------------------------------------------------------------------------------------------------------------------
t_global_start = time.time()
print("Caso de uso: accidentes en Cataluña cada 30 días.")
subset = accidentes[accidentes['COMUNIDAD_AUTONOMA'].str.contains("Cataluña")]
usadas = ['TOT_MUERTOS30D','TOT_HERIDOS_GRAVES30D','TOT_HERIDOS_LEVES30D', 'TOT_VEHICULOS_IMPLICADOS']

X = subset[usadas]
n = 15000
print(X.shape[0])
X  = X.sample(n,random_state=123456)
X_normal = preprocessing.normalize(X, norm='l2')
min_size = 30

k_means = KMeans(init='k-means++', n_clusters=6, n_init=5)
aglo=AgglomerativeClustering(n_clusters=76,linkage="ward")
bandwidth = estimate_bandwidth(X_normal, quantile=0.3, random_state=123456)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
birch=Birch(n_clusters=12,threshold=0.1)
spectral=SpectralClustering(n_clusters=6,eigen_solver="arpack")

clustering_algorithms = (
    ("K-medias",k_means),
    ("AC",aglo),
    ("MeanShift",meanshift),
    ("Birch",birch),
    ("SpectralC",spectral)
)


# Creamos los datos de salida, y mostramos las gráficas si queremos.
outputData = dict()
for algorithm_name,algorithm in clustering_algorithms:
    results = dict()
    met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
    n_clusters=len(set(cluster_predict))

    if (n_clusters > 15):
        X_filtrado = clusterFrame[clusterFrame.groupby('cluster').cluster.transform(len) > min_size]
    else:
        X_filtrado = clusterFrame

    makeScatterPlot(data=clusterFrame, outputName="./imagenes/scatterMatrix_caso2_" + algorithm_name,
                    displayOutput=False)

    makeHeatmap(data=X_filtrado, outputName="./imagenes/heatmap_caso2_" + algorithm_name,
                displayOutput=False)

    results['N Clusters']=n_clusters
    results['HC metric']=met[0]
    results['SC metric']=met[1]
    results['Time']=timeAlg

    outputData[algorithm_name] = results

latexCaso1 = createLatexDataFrame(data=outputData)
f = open('caso2.txt','w')
f.write(latexCaso1.to_latex())
f.close()

print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
    'Name', 'N clusters', 'HC metric', 'SC metric', 'Time (s)'))

for name, res in outputData.items():
    print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
        name, res['N Clusters'], res['HC metric'], res['SC metric'],
        res['Time']))

print('\nTotal time = {0:.2f}'.format(time.time() - t_global_start))

#-----------------------------------------------------------------------------------------------------------------------
# CASO DE USO 3
#-----------------------------------------------------------------------------------------------------------------------
t_global_start = time.time()
print("Tercer caso de uso: accidentes en Bizkaia en los que llueve.")
subset = accidentes[accidentes['PROVINCIA'] == 'Bizkaia']
subset = accidentes[accidentes['FACTORES_ATMOSFERICOS'].str.contains('LLUVIA')]
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]
print(X.shape[0])
X_normal = preprocessing.normalize(X, norm='l2')

min_size=5
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5)
aglo=AgglomerativeClustering(n_clusters=26,linkage="ward")
bandwidth = estimate_bandwidth(X_normal, quantile=0.4, random_state=123456)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
birch=Birch(n_clusters=5,threshold=0.1)
spectral=SpectralClustering(n_clusters=4,eigen_solver="arpack")

clustering_algorithms = (
    ("K-medias",k_means),
    ("AC",aglo),
    ("MeanShift",meanshift),
    ("Birch",birch),
    ("SpectralC",spectral)
)


# Creamos los datos de salida, y mostramos las gráficas si queremos.
outputData = dict()
for algorithm_name,algorithm in clustering_algorithms:
    results = dict()
    met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
    n_clusters=len(set(cluster_predict))

    if (n_clusters > 15):
        X_filtrado = clusterFrame[clusterFrame.groupby('cluster').cluster.transform(len) > min_size]
    else:
        X_filtrado = clusterFrame

    makeScatterPlot(data=clusterFrame, outputName="./imagenes/scatterMatrix_caso3_" + algorithm_name,
                    displayOutput=False)

    makeHeatmap(data=X_filtrado, outputName="./imagenes/heatmap_caso3_" + algorithm_name,
                displayOutput=False)

    if algorithm_name == 'AC':
        X_filtrado_normal = preprocessing.normalize(X_filtrado,norm='l2')
        linkage_array = ward(X_filtrado_normal)

        dendrogram(linkage_array,leaf_rotation=90., leaf_font_size=5.)
        plt.show()
        #plt.clf()

    results['N Clusters']=n_clusters
    results['HC metric']=met[0]
    results['SC metric']=met[1]
    results['Time']=timeAlg

    outputData[algorithm_name] = results

latexCaso1 = createLatexDataFrame(data=outputData)
f = open('caso3.txt','w')
f.write(latexCaso1.to_latex())
f.close()


print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
    'Name', 'N clusters', 'HC metric', 'SC metric', 'Time (s)'))

for name, res in outputData.items():
    print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
        name, res['N Clusters'], res['HC metric'], res['SC metric'],
        res['Time']))

print('\nTotal time = {0:.2f}'.format(time.time() - t_global_start))