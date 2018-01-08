"""
Autor:
Alberto Armijo Ruiz.
Descripción:
Práctica 3 Inteligencia de Negocio, Universidad de Granada.
Competeción en Kaggle sobre precios de casas (Regresión).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV, Lasso


def get_score(prediction, lables):
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


"""
Primero haremos un estudio de la relación de las variables con el precio de venta, para ello.
"""
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))


# Tratamiento de missing data en train.
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
test = test.drop(test.loc[test['Electrical'].isnull()].index)
print(train.isnull().sum().max())

# Tratamiento extra para test.
missing_data = test.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data],axis=1,keys=['Total'])
missing_data = missing_data[missing_data['Total'] > 0]
print(list(missing_data.index) )
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
# test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
# test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test.drop(['BsmtFullBath','BsmtHalfBath'],axis=1,inplace=True)
train.drop(['BsmtFullBath','BsmtHalfBath'],axis=1,inplace=True)
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(0)
test['GarageArea'] = test['GarageArea'].fillna(0)
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna('NoBSMT')
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])



# Outliers, solamente de las variables que estamos utilizando.
outliers = train.sort_values(by = 'GrLivArea', ascending = False)[:2]
print(outliers['GrLivArea'])

train = train.drop(train[train['Id']==1298].index)
train = train.drop(train[train['Id']==523].index)

# Transformaremos los datos de SalePrice para que sigan una distribución normal.
train['SalePrice'] = np.log(train['SalePrice'])

# También haremos esto para todas las variables que estamos estudiando que no tienen una distribución normal.
train['GrLivArea'] = np.log(train['GrLivArea'])


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

# Transformar aquellos datos que tenemos como categoricos a variables dummies.
x_train = pd.get_dummies(train)
y_train = x_train.pop('SalePrice')


x_test = test.copy()
x_test['GrLivArea'] = np.log(x_test['GrLivArea'])
vars =['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
x_test['HasBsmt'] = pd.Series(len(x_test['TotalBsmtSF']), index=x_test.index)
x_test['HasBsmt'] = 0
x_test.loc[x_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(x_test['TotalBsmtSF'])
x_test = pd.get_dummies(x_test)
col_test = list(x_test)
col_train = list(x_train)

used_vars = set(
    vars for vars in col_test if vars in col_train
)


x_train = x_train[list(used_vars)]
x_test = x_test[list(used_vars)]

numeric_features = [f for f in x_train.columns if x_train[f].dtype != object]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train[numeric_features])

scaled = scaler.transform(x_train[numeric_features])
for i, col in enumerate(numeric_features):
    x_train[col] = scaled[:, i]

scaled = scaler.transform(x_test[numeric_features])
for i, col in enumerate(numeric_features):
    x_test[col] = scaled[:, i]


# creamos un modelo.
xgbReg = xgb.XGBRegressor( colsample_bytree=0.2, gamma=0.0, learning_rate=0.05, max_depth=6, min_child_weight=1.5,
                           n_estimators=7200, reg_alpha=0.9, reg_lambda=0.6, subsample=0.2, seed=42, silent=1)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Entrenamos el modelo.
xgbReg.fit(x_train,y_train)

model_lasso = Lasso(alpha=0.00099, max_iter=50000)
model_lasso.fit(x_train,y_train)

train_labels = np.exp(y_train)

# Calculamos el error.
print("xgb score:")
get_score(prediction=xgbReg.predict(x_train) ,lables=y_train)
print("lasso score:")
get_score(prediction=model_lasso.predict(x_train), lables=y_train)

y_pred = ( model_lasso.predict(x_train)*0.9 + xgbReg.predict(x_train) )/1.9
y_pred = np.exp(y_pred)
print(y_pred)

output = pd.concat([pd.Series(train_labels),pd.Series(y_pred)],axis=1,keys=['real','pred'])

print("model in train score:")
print(rmse(y_true=train_labels,y_pred=y_pred))

y_pred_xgb = xgbReg.predict(x_test)
y_pred_lasso = model_lasso.predict(x_test)
#
y_pred = (y_pred_xgb + y_pred_lasso * 0.9) / 1.9
y_pred = np.exp(y_pred_lasso)

test2 = pd.read_csv('output.csv')
y_test = test2['SalePrice']

get_score(prediction=y_pred,lables=y_test)
#
pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred}).to_csv('submission_29-12-3.csv', index =False)