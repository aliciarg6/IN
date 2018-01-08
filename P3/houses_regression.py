
# coding: utf-8

# # Problema de regresión sobre precio de viviendas.

# In[533]:


# Liberías utilizadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV, Lasso
import scipy.stats as stats


# In[534]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[535]:


print(train.shape, test.shape)


# In[536]:


# Vamos a comprobar que variables están perdidas.
train.columns[train.isnull().any()]


# In[537]:


# Tala con los atributos que tienen valores perdidos, y su porcentaje.
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))


# De esos valores, tendremos que borras aquellas variables que tengan muchos valores perdidos,  por ejemplo, todas aquellas que tienen más de un 10%.
# Antes de eso, estudiaremos la variable que queremos predecir, 'SalePrice' y la relación de otras variables con esta.

# In[538]:


sns.distplot(train['SalePrice'])
plt.show()


# Como se puede ver, la distribución del la variable no es igual a una distribución normal, esto lo podemos arreglar utilizando una transformación logarítmica sobre los datos. A la hora de calcular las predicciones, tendremos que volver a transformar los datos.

# In[539]:


train['SalePrice'].skew()


# In[540]:


# Crearemos la variable que contendrá los datos.
y_train = train['SalePrice']
y_train = np.log(y_train)
print(y_train.skew())
sns.distplot(y_train)
plt.show()


# Ahora, estudiaremos el resto de las variables que hay dentro de nuestro conjunto de entrenamiento, así podremos ver cuáles son las más significativas.

# In[541]:


quantitative_data = train.select_dtypes(include=[np.number])
qualitative_data = train.select_dtypes(exclude=[np.number])
quantitative_data = quantitative_data.drop('Id',axis=1)
print(quantitative_data.shape, qualitative_data.shape)


# In[542]:


# Comprobaremos la relación que hay entre SalePrice y el resto de las variables.
f, ax = plt.subplots(figsize=(12, 9))
corrmat = train.corr()
sns.heatmap(corrmat)


# In[543]:


corrmat['SalePrice'].sort_values(ascending=False)


# In[544]:


# Las variables que más correlación tienen con el precio de venta de una
# vivienda son.
imp_vars = list(corrmat['SalePrice']
                .sort_values(ascending=False)[:10].index)
print(imp_vars)


# In[545]:


comp_var = train['SalePrice']
for var in imp_vars:
    sns.jointplot(x=train[var],y=comp_var)
    plt.show()


# Como se puede ver en las gráficas, existen algunos outliers, como por ejemplo los de de GrLivArea. También se puede ver que la variable YearBuilt no sigue una distribución normal, por lo que deberíamos de transformarla. También deberíamos transformar otras variables para que se asemeje más a una distribución normal.
# Ahora, estudiaremos las variables categóricas.

# In[546]:


qualitative_data.describe()


# In[547]:


qualitative = list(qualitative_data)


# Para ver que variables son más importantes dentro de las cualitativas, utilizaremos el test anova para calcular la disprersión de cada variable.
#

# In[548]:


def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)



# In[549]:


cat_imp = a
cat_imp


# ## Procesamiento de datos.
# Primero quitaremos los outliers, después transformaremos las variables categóricas, imputaremos valores perdidos y quitaremos inconsistencias.

# In[550]:


# Lo primero será quitar los outliers de GrLivArea.
train.drop(train[train['GrLivArea']>4000].index, inplace=True)
train.shape


# In[551]:


train_labels = train.pop('SalePrice')
y_train = np.log(train_labels)


# In[552]:


# Lo siguiente que haremos será crear un único conjunto de datos.
# Así podremos preprocesar de forma más sencilla los datos.


features = pd.concat([train, test], keys=['train', 'test'])
features.head()


# In[553]:


#importing function
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def factorize(data, var, fill_na = None):
      if fill_na is not None:
            data[var].fillna(fill_na, inplace=True)
      le.fit(data[var])
      data[var] = le.transform(data[var])
      return data

missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total']>0]
missing_data = list(missing_data.index)
missing_data


# In[554]:


# Primero imputaremos las variables que son numéricas.
features["MasVnrArea"].fillna(0,inplace=True)
features["BsmtFinSF1"].fillna(0,inplace=True)
features["BsmtFinSF2"].fillna(0,inplace=True)
features["TotalBsmtSF"].fillna(0,inplace=True)
features["GarageArea"].fillna(0,inplace=True)
features["BsmtFullBath"].fillna(0,inplace=True)
features["BsmtHalfBath"].fillna(0,inplace=True)
features["GarageCars"].fillna(0,inplace=True)
features["GarageYrBlt"].fillna(0,inplace=True)
features["PoolArea"].fillna(0,inplace=True)
features["BsmtUnfSF"].fillna(0,inplace=True)
features["LotFrontage"].fillna(0,inplace=True)


# In[555]:


# Ahora veremos cuales son los datos de las variables categóricas del tipo
# calidad.
features['GarageQual'].unique()


# In[556]:


# Lo que haremos será transformar estos datos, nan será 0, Po será 1, etc..
# Haremos lo mismo con otros tipos de variables parecidas.
diccionario = {np.nan:0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}

missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total']>0]
missing_data = list(missing_data.index)

name = [name for name in missing_data if "QC" in name
        or "Qual" in name or "Cond" in name
        or "Qu" in name]

for n in name:
    features[n] = features[n].map(diccionario).astype(int)


# In[557]:


missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total']>0]
missing_data = list(missing_data.index)

# Varaibles del tipo Bsmt.
name = [name for name in missing_data if "Bsmt" in name]
print(name)

# Algunas de estas ya las hemos imputado.
name = ["BsmtFinType2","BsmtFinType1","BsmtExposure"]
for n in name:
    features[n] = features[n].fillna(features[n].mode()[0])


# In[558]:


missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total']>0]
missing_data = list(missing_data.index)
missing_data


# In[559]:


features = factorize(features, "MSZoning", "RL")
features = factorize(features, "Exterior1st", "Other")
features = factorize(features, "Exterior2nd", "Other")
features = factorize(features, "MasVnrType", "None")
features = factorize(features, "SaleType", "Oth")


# In[560]:


missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total']>0]
missing_data = list(missing_data.index)
missing_data


# In[561]:


features["GarageFinish"] = features["GarageFinish"].map(
    {np.nan:0,"Unf":1,"RFn":2,"Fin":3}).astype(int)
features["Functional"] = features["Functional"].fillna(
    features["Functional"].mode()[0])
features = factorize(features,"GarageType","Other")
features["Utilities"] = features["Utilities"].map({np.nan:0,
                                        "NoSeWa":1,
                                        "AllPub":2})
features["Electrical"] = features["Electrical"].fillna(
                    features["Electrical"].mode()[0])
features.drop(["MiscFeature","Alley","Fence"],axis=1,inplace=True)

# Por último, cambiamos todas las variables categóricas
# que nos queden en dummies.
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)


# In[562]:


# Comprobamos que no quedan datos Nan.
features.isnull().sum().max()


# In[563]:


# Separamos los conjuntos de datos.
x_train = features.loc['train'].drop('Id',axis=1)
x_test = features.loc['test'].drop('Id',axis=1)


# In[564]:


# Calculamos la varianza de cada una de las columnas (numéricas)
numeric_features = [f for f in x_train.columns if x_train[f].dtype != object]

# Transformamos las variables con mucha varianza con el logaritmo.
from scipy.stats import skew
skewed = x_train[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index

x_train[skewed] = np.log(x_train[skewed])
x_test[skewed] = np.log(x_test[skewed])


# In[568]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train[numeric_features])

scaled = scaler.transform(x_train[numeric_features])
for i, col in enumerate(numeric_features):
    x_train[col] = scaled[:, i]

scaled = scaler.transform(x_test[numeric_features])
for i, col in enumerate(numeric_features):
    x_test[col] = scaled[:, i]
