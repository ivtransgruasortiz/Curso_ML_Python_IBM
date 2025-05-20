#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score


# ## Ejercicio 1
# Imprime por pantalla tu nombre, apellido y email

# In[2]:


# Resuelve en esta celda el ejercicio 1
nombre = "Iván" 
apellido = "Ortiz Martín"
email = "ivtransgruasortiz@gmail.com"
print(f"Nombre: {nombre} \nApellidos: {apellido} \nemail: {email}")


# ## Ejercicio 2

# ### Ejercicio 2.1
# Muestra los primeros registros del siguiente conjunto de datos

# In[3]:


# Cargar el dataset "titanic" de seaborn
df = sns.load_dataset("titanic")

# Resolver aqui el ejercicio 2.1
df.head()


# ### Ejercicio 2.2
# Muestra una descripción general del conjunto de datos cargado en el ejercicio 2.1

# In[4]:


# Resuelve aqui el ejercicio 2.2
df.describe()


# ### Ejercicio 2.3
# Elimina los datos de la columna "deck" y sustituye los datos nulos para la columno "age" con el valor medio. Luego muestra el dataframe:

# In[5]:


# Resuelve aqui el ejercicio 2.3
df.drop('deck', axis=1, inplace=True)
df['age'] = df['age'].fillna(df['age'].mean())
df


# #### Ejercicio 2.4
# Crea un gráfico de tipo histograma que represente la distribución de las edades de los pasajeros. Usa **matplotlib** o **seaborn** como herramientas de visualización

# In[6]:


# Resuelve aqui el ejercicio 2.4
sns.displot(df['age'], bins=8)  # Utilizo bins=8 porque muestra unos rangos de edad fácilmente interpretables


# #### Ejercicio 2.5
# Crea un gráfico circular que represente la distribución por género de los pasajeros. Usa **matplotlib** o **seaborn** como herramientas de visualización

# In[7]:


# Resuelve aqui el ejercicio 2.5
dict_sex = df.groupby("sex")["sex"].count().to_dict()
plt.pie(x=dict_sex.values(), labels=dict_sex.keys())
plt.show()


# ## Ejercicio 3

# #### Ejercicio 3.1
# Separa el dataset en dos variables, 'x' e 'y'.
#  - La variable 'x' debe incluir las columnas 'depth', 'table', 'price', 'x', 'y' y 'z'.
#  - La variable 'y' debe incluir la columna 'carat'.

# In[8]:


# Carga el dataset "diamonds" de seaborn
diamantes = sns.load_dataset("diamonds")
diamantes.head()

# Resuelve aqui el ejercicio 3.1
x = diamantes[['depth', 'table', 'price', 'x', 'y', 'z']]
y = diamantes[['carat']]


# #### Ejercicio 3.2
# Separa el dataset provisto en conjunto de entrenamiento y de test, dando un 40% de los datos al conjunto de test

# In[9]:


# Resuelve aqui el ejercicio 3.2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)


# #### Ejercicio 3.3
# Crea un modelo de regresión lineal y entrenalo con los conjuntos de datos separados en el ejercicio anterior. La variable objetivo corresponde a la columna 'carat'

# In[10]:


# Resuelve aqui el ejercicio 3.3
lrm = LinearRegression()
lrm.fit(X_train, y_train)


# #### Ejercicio 3.4
# Utiliza el modelo entrenado en el apartado anterior para predecir los valores de la columna 'carat'

# In[11]:


# Resuelve aqui el ejercicio 3.3
y_pred = lrm.predict(X_test)
print(y_pred)

# Descomentar lo siguiente si se quiere crear una nueva columna con las predicciones del modelo sobre la columna "carat"
#diamantes["carat_pred"] = lrm.predict(x)
#diamantes
#r2_score(y_test, y_pred)  # Descomentar para evaluar el modelo


# ### Ejercicio 4

# #### Ejercicio 4.1
# Muestra un resumen estadístico de los datos numéricos del dataset provisto

# In[12]:


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ALTERNATIVA CON OTRO DATASET 
# Cargar el dataset "titanic" de seaborn
# df = sns.load_dataset("titanic")

# Resuelve aqui el ejercicio 4.1
X.describe()


# #### Ejercicio 4.2
# Separa el dataset provisto en conjunto de entrenamiento y de test, dando un 40% de los datos al conjunto de test

# In[13]:


# Resuelve aqui el ejercicio 4.2
scaler = StandardScaler() # Escalamos los datos porque hay mucha diferencia en la desviación estándar entre ellos
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.4, random_state=42)


# #### Ejercicio 4.3
# Crea un modelo de regresión logística y entrenalo con los conjuntos de datos separados en el ejercicio anterior. 

# In[14]:


# Resuelve aqui el ejercicio 4.3
logm = LogisticRegression(max_iter=4000)
logm.fit(X_train, y_train)


# #### Ejercicio 4.4
# Utiliza el modelo entrenado en el apartado anterior para predecir los valores de la variable objetivo. 

# In[15]:


# Resuelve aqui el ejercicio 4.4
y_pred = logm.predict(X_test)
print(y_pred)

# Descomentar lo siguiente para evaluar el modelo
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))


# ### Ejercicio 5

# #### Ejercicio 5.1
# Muestra los primeros registros del siguiente conjunto de datos

# In[16]:


df = sns.load_dataset("penguins")

# Resuelve aqui el ejercicio 5.1
df.head()


# #### Ejercicio 5.2
# Separa el dataset provisto en conjunto de entrenamiento y de test, dando un 40% de los datos al conjunto de test. La variable objetivo corresponde a la columna "species"

# In[17]:


# Resuelve aquí el ejericio 5.2

df['bill_length_mm'] = df['bill_length_mm'].fillna(df['bill_length_mm'].mean())
df['bill_depth_mm'] = df['bill_depth_mm'].fillna(df['bill_depth_mm'].mean())
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean())
df['body_mass_g'] = df['body_mass_g'].fillna(df['body_mass_g'].mean())

# Convertimos variables categóricas nominales de string a categóricas numéricas con labelencoder
df['island_enc'] = LabelEncoder().fit_transform(df['island'])
df['sex_enc'] = LabelEncoder().fit_transform(df['sex'])

X = df.drop(['species', 'island', 'sex'], axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# #### Ejercicio 5.3
# Elige un modelo de clasificación entre los siguientes: 
#  - Decision tree (Árbol de decisión)
#  - Random forest
#  - SVM (Support Vector Machine)
#  
# Entrenalo con los conjuntos de datos separados en el ejercicio anterior. 

# In[18]:


# Resuelve aquí el ejericio 5.3
rfmodel = RandomForestClassifier(n_estimators=100, random_state=42)
rfmodel.fit(X_train, y_train)


# #### Ejercicio 5.4
# Utiliza el modelo entrenado en el apartado anterior para predecir los valores de la variable objetivo. 

# In[19]:


# Resuelve aquí el ejericio 5.4
y_pred = rfmodel.predict(X_test)
print(y_pred)

# Descomentar lo siguiente para ver la evaluación del modelo
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))


# ### Ejercicio 6

# #### Ejercicio 6.1
# Muestra un resumen estadístico de los datos numéricos del dataset provisto

# In[20]:


url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
df = pd.read_csv(url, header=None)
df.columns = [
   "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]
#df = df.drop("Sex", axis=1)

# Resuelve aqui el ejercicio 6.1
df.describe()


# #### Ejercicio 6.2
# Separa el conjunto de datos en variables 'x' e 'y', donde la 'y' corresponde a la variable objetivo 'Sex'.

# In[21]:


# Resuelve aqui el ejercicio 6.2
columns = df.columns
columns = columns.drop("Sex")
scaler = StandardScaler()
x = df[columns]
x_sc = scaler.fit_transform(x)  # Escalado de datos
y = df["Sex"].values.ravel()  # aplana los datos de la varible a predecir


# #### Ejercicio 6.3
# Separa el dataset provisto en conjunto de entrenamiento y de test, dando un 40% de los datos al conjunto de test. 

# In[22]:


# Resuelve aquí el ejercicio 6.3
X_train, X_test, y_train, y_test = train_test_split(x_sc, y, test_size=0.4, random_state=42)


# #### Ejercicio 6.4
# Crea un modelo de regresión logística y entrenalo con los conjuntos de datos separados en el ejercicio anterior. Utiliza el valor 3 para el parámetro 'n_neighbors'

# In[23]:


# Resuelve aquí el ejercicio 6.4
print(df["Sex"].unique())
# Vamos a utilizar un algoritmo de clasificación KNN, mejor que uno logístico,
# ya que la variable objetivo puede tomar 3 valores, no es una clasificación binaria

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)


# #### Ejercicio 6.5
# Muestra la matriz de confusión resultante para la salida del ejercicio anterior

# In[24]:


# Resuelve aquí el ejercicio 6.5
y_pred = knn_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

