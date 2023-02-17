######################################
# Importando as libraries
######################################
 
import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
 
######################################
# Importa o dataset
######################################
 
dataset = pd.read_csv("bcdados03.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
######################################
# Separar dados em Treino e Teste
######################################
 
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.9, random_state = None)
 
######################################
# Dimensionamento de recurso
######################################
sc_X_train = StandardScaler()
sc_X_test = StandardScaler()
 
X_train = sc_X_train.fit_transform(X_train)
X_test = sc_X_test.fit_transform(X_test)
 
######################################
# Treinando o modelo
######################################
 
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
 
######################################
# Dados da cidade a ser classificada
######################################
 
print(classifier.predict([[0.800,11.569,48.130,100,100,48.890]]))

######################################
# Previsao
######################################
 
y_pred = classifier.predict(X_test)
y_result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)


######################################
# Matrix de confusao
######################################
 
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))