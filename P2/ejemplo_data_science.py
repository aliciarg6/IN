# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Octubre/2017
Contenido:
    Primeros ejemplos de uso de librerías Python para Ciencia de Datos
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Para aprender Python:
    https://github.com/jakevdp/WhirlwindTourOfPython
Documentación de Python para consulta:
    https://docs.python.org/3/
Scikit-learn:
    https://www.oreilly.com/ideas/intro-to-scikit-learn
    http://nbviewer.jupyter.org/github/jakevdp/sklearn_tutorial/blob/master/notebooks/Index.ipynb
    http://scikit-learn.org/stable/tutorial/basic/tutorial.html
NumPy:
    https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
Pandas:
    http://pandas.pydata.org/pandas-docs/stable/10min.html
Visualización:
    https://blog.modeanalytics.com/python-data-visualization-libraries/
    https://dsaber.com/2016/10/02/a-dramatic-tour-through-pythons-data-visualization-landscape-including-ggplot-and-altair/
    http://matplotlib.org/
    https://seaborn.pydata.org/
'''

import matplotlib.pyplot as plt
import pandas as pd

#True si cada variable categórica se convierte en varias binarias (tantas como categorías),
#False si solo se convierte la categórica a numérica (ordinal)
binarizar = False

'''
devuelve un DataFrame, los valores perdidos notados como '?' se convierten a NaN,
si no, se consideraría '?' como una categoría más
'''
if not binarizar:
    adult_orig = pd.read_csv('adult.csv')
else:
    adult_orig = pd.read_csv('adult.csv',na_values="?")

print("------ Lista de características y tipos (object=categórica)")
print(adult_orig.dtypes,"\n")

print("------ Distribución de datos en la característica 'workclass'")
print(adult_orig['workclass'].value_counts(),"\n")

print("------ Y en la clase")
print(adult_orig['class'].value_counts(),"\n")

#'''
# gráfico de barras horizontales con la proporción de cada clase
plt.figure(1)
plt.clf()
import seaborn as sns
ax = sns.countplot(y="class", data=adult_orig, color="c");
ncount = adult_orig.shape[0]
for p in ax.patches:
    val_x=p.get_bbox().get_points()[:,0]
    val_y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.0f} ({:.1f}%)'.format(val_x[1], 100.*val_x[1]/ncount), (val_x.mean(), (val_y-0.4)), ha='center', va='center')
#'''
    
#'''
print("------ Preparando el scatter matrix...")
plt.figure(2)
plt.clf()
# para scatter matrix, se convierten las variables categóricas a numéricas
adult_int = adult_orig
char_cols = adult_int.dtypes.pipe(lambda x: x[x == 'object']).index #lista de columnas con var. categóticas (las de tipo 'object')
for c in char_cols:
    adult_int[c] = pd.factorize(adult_int[c])[0]
lista_vars = list(adult_int)
lista_vars.remove('class') #excluimos la columna 'class' del plot
#se genera el scatter matrix
sns.set()
sns_plot = sns.pairplot(adult_int, vars=lista_vars, hue="class", diag_kind="kde") #en hue indicamos que la columna 'class' define los colores
sns_plot.savefig("adult_scatter_plot.png")
print("")
#'''

'''
si el dataset contiene variables categóricas con cadenas, es necesario convertirlas a numéricas antes de usar 'fit', y para
no hacerlas ordinales, mejor convertirlas a variables binarias con get_dummies
Otras alternativas para convertir las variables categóricas es usar LabelEncoder, One-Hot-Encoding o LabelBinarizer en la matriz numpy (ver más abajo)
Para saber más: http://pbpython.com/categorical-encoding.html
'''
# devuelve una lista de las características categóricas excluyendo la columna 'class' que contiene la clase
lista_categoricas = [x for x in adult_orig.columns if (adult_orig[x].dtype == object and adult_orig[x].name != 'class')]
if not binarizar:
    adult = adult_orig
else:
    # reemplaza las cateogóricas por binarias
    adult = pd.get_dummies(adult_orig, columns=lista_categoricas)

# coloco la columna que contiene la clase como última columna por convención
clase = adult['class']
adult.drop(labels=['class'], axis=1,inplace = True)
adult.insert(len(adult.columns), 'class', clase)

# separamos el DataFrame en dos arrays numpy, uno con las características (X) y otro (y) con la clase
# si la última columna contiene la clase, se puede separar así
X = adult.values[:,0:len(adult.columns)-1]
y = adult.values[:,len(adult.columns)-1]

'''
# también se puede separar indicando los nombres de las columnas
columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]
X = adult[list(columns)].values
y = adult["class"].values
#'''

'''
Si las variables categóticas tienen muchas categorías, se generarán muchas variables y algunos algoritmos (por ejemplo, SVM) serán
extremadamente lentos. Se puede optar por solo convertirlas a variables numéricas (ordinales) sin binarizar. Esto se haría si no se ha
ejecutado pd.get_dummies() previamente. No funciona si hay valores perdidos notados como NaN
'''
if not binarizar:
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    for i in range(0,X.shape[1]):
        if isinstance(X[0,i],str):
            X[:,i] = le.fit_transform(X[:,i])

'''
# validación cruzada, pero sin control de semilla ni particionado estratificado
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree, features, target, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#'''

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
import numpy

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    y_prob_all = []

    for train, test in cv.split(X, y):
        modelo = modelo.fit(X[train],y[train])
        y_pred = modelo.predict(X[test])
        y_prob = modelo.predict_proba(X[test])[:,1] #la segunda columna es la clase positiva '>50K' en adult
        y_test_bin = le.fit_transform(y[test]) #se convierte a binario para AUC: '>50K' -> 1 (clase positiva) y '<=50K' -> 0 en adult
        print("Accuracy: {:6.2f}%, F1-score: {:.4f}, G-mean: {:.4f}, AUC: {:.4f}".format(accuracy_score(y[test],y_pred)*100 , f1_score(y[test],y_pred,average='macro'), geometric_mean_score(y[test],y_pred,average='macro'), roc_auc_score(y_test_bin,y_prob)))
        y_test_all = numpy.concatenate([y_test_all,y_test_bin])
        y_prob_all = numpy.concatenate([y_prob_all,y_prob])

    print("")

    return modelo, y_test_all, y_prob_all
#------------------------------------------------------------------------


#------------------------------------------------------------------------
'''
Dibuja la curva ROC
'''
from sklearn.metrics import roc_curve, auc

def curva_ROC(figura_id,new,y_test,y_prob,nombre):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figura_id)

    if new:
        plt.clf()

    plt.plot(fpr, tpr, lw=2, label=nombre+' (%0.4f)' % roc_auc) #color='darkorange',
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.legend(loc="lower right")

    if new:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')

    plt.show()

    return roc_auc
#------------------------------------------------------------------------

#'''
print("------ Árbol de decisión...")
from sklearn import tree
arbol = tree.DecisionTreeClassifier(random_state=0, max_depth=10) #podemos limitar a profundidad 5 para generar un árbol legible aunque pierda algo de precisión

arbol, y_test_arbol, y_prob_arbol = validacion_cruzada(arbol,X,y,skf)
curva_ROC(3,True,y_test_arbol,y_prob_arbol,'Árbol')

'''
Para visualizar el árbol generado, se puede usar graphviz, que debe ser previamente instalado
Por ejemplo, desde Anaconda Navigator: Environments / Seleccionar "Not installed" / Buscar "graphviz" / Marcar + "Apply"
Incluir el directorio "...Anaconda3\pkgs\graphviz-2.38.0-4\Library\bin\graphviz" en las variables de entorno PATH y GRAPHVIZ_DOT (variable nueva que debe crearse)
'''
print("------ Generando una visualización del árbol en 'adult.pdf'...")
import graphviz
feat = list(adult)
feat.remove('class')
dot_data = tree.export_graphviz(arbol, out_file=None, filled=True, feature_names=feat, class_names=['menos_50K', 'mas_50K'], rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("adult") #genera un fichero adult.pdf con el árbol
#'''

#'''
print("------ XGB...")
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators = 200)

clf, y_test_clf, y_prob_clf = validacion_cruzada(clf,X,y,skf)
curva_ROC(3,False,y_test_clf,y_prob_clf,'XGB')

'''
Visualizar las características más importantes según la frecuencia con que se usan en los árboles de XGB (sobre el último modelo de la CV)
'''
plt.figure(4)
plt.clf()
features = list(adult)
mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
ts.index = ts.reset_index()['index'].map(mapFeat)
ax2=ts.sort_values()[-20:].plot(kind="barh", figsize = (8,8), title=("20 características más importantes"), color='orange')
ax2.set_xlabel("importancia")
ax2.set_ylabel("característica")
#'''


#Otros algoritmos y paquetes:
    
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)

from sklearn import svm
svc = svm.SVC(kernel='linear', C=1)

#Para XGB hay que instalar el paquete py-xgboost o similar
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators = 200)

'''
Para instalar xgboost desde Anaconda Navigator: Environments / Seleccionar "Not installed" / Buscar "xgboost" / Marcar "py-xgboost" + "Apply"
También puedes usar 'conda install py-xgboost' (buscar con 'conda search xgb')

Desbalanceo (necesario para g-mean) https://github.com/scikit-learn-contrib/imbalanced-learn:
conda install -c glemaitre imbalanced-learn

Para codificar variables categóricas (https://github.com/scikit-learn-contrib/categorical-encoding):
pip install category_encoders
'''