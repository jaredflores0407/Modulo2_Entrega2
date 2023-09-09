#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names")


# In[2]:


#pip install pydotplus


# In[3]:


df= pd.read_csv('loan_approval_dataset.csv') #Cargamos el DataFrame


# In[4]:


len(df) #Determinar el número de filas


# In[5]:


df.head(15) #Ver la estructura de los datos


# In[6]:


df.drop(columns= 'loan_id', inplace=True) #Eliminar la columna loan_id, que no aporta información importante


# In[7]:


df.columns #Ver la escritura de las columnas


# In[8]:


df[' loan_status'].unique() #Ver los valores de las columnas categóricas


# In[9]:


# Reemplazar los valores categóricos por valores enteros 0 y 1 respectivamente
df[' loan_status']= df[' loan_status'].replace([' Approved', ' Rejected'],[1,0])
df[' education']= df[' education'].replace([' Graduate',' Not Graduate'],[1,0])
df[' self_employed']= df[' self_employed'].replace([' No',' Yes'],[1,0])


# In[10]:


#Observar cuantas filas tiene un credito aprovado y no aprovado
my_tab = pd.crosstab(index=df[" loan_status"], columns="count") 
my_tab


# In[ ]:





# In[11]:


# Seleccionar las columnas de entrada y la columna objetivo
X= df.drop(columns= ' loan_status')
y= df[' loan_status']


# In[12]:


# Separación del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # Se tomará el 20% como test


# In[13]:


# Creal el objeto Árbol de Decisiones
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10) #Se eligió una profundidad de 10 y utilizará entropy como función de pérdida

# Entrenamiento del Árbol de Decisiones 
clf = clf.fit(X_train,y_train)

#Predicciones con el dataset de prueba
y_pred = clf.predict(X_test)


#Métricas para la evaluación del modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='micro')
recall = metrics.recall_score(y_test, y_pred, average='micro')
f1 = metrics.f1_score(y_test, y_pred, average='micro')

# Imprimimos las métricas
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Reporte de Clasificación
cr = metrics.classification_report(y_test, y_pred)
print("Classification report:")
print(cr)

# Matriz de confusión
cm = metrics.confusion_matrix(y_test, y_pred)
#Display de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



# In[14]:


import random

# Crea una lista de 10 números aleatorios enteros entre 0 y 100
random_numbers = []
for i in range(10):
    random_numbers.append(random.randint(0, 4269))

#print(random_numbers)


# In[ ]:



    


# In[15]:


type(df.loc[1967])


# In[16]:


cont=1
for fila in random_numbers:
    fil= df.loc[fila].copy()
    fil.drop(columns=' loan_status', inplace=True)
    array = fil[:-1].to_numpy()
    array = array.reshape(1, -1)
    predictions = clf.predict(array)
    print(f'Set {cont}, predicción: {predictions}')
    cont= cont+1
    print()
    print()


# In[ ]:





# In[ ]:




