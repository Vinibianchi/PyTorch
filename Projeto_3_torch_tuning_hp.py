# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 08:09:53 2022

@author: Vinicius
"""

import pandas as pd
import numpy as np

import skorch
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set sementes
np.random.seed(123)
torch.manual_seed(123)


# Base de dados
previsores = pd.read_csv('Datasets/entradas_breast.csv')
classe = pd.read_csv('Datasets/saidas_breast.csv')

previsores = np.array(previsores, dtype = 'float32')
classe = np.array(classe, dtype = 'float32').squeeze(1)

# Classe da rede

class Classificador_torch(nn.Module):
    
    def __init__(self, activation, neurons, initializer):
        super().__init__()
        
        self.dense0 = nn.Linear(30, neurons)
        initializer(self.dense0.weight)
        self.activation0 = activation
        self.dense1 = nn.Linear(neurons, neurons)
        initializer(self.dense1.weight)
        self.activation1 = activation
        self.dense2 = nn.Linear(neurons, 1)
        initializer(self.dense2.weight)
        
    def forward(self, X):
        
        X = self.dense0(X)
        X = self.activation0(X)
        X = self.dense1(X)
        X = self.activation1(X)
        X = self.dense2(X)
        
        return X




# Skorh
classisicador_sklearn = skorch.NeuralNetBinaryClassifier(Classificador_torch,
                                 lr = 0.001,
                                 optimizer__weight_decay = 0.0001,
                                 train_split=False
                                 )


# Tuning
#'''Parâmetros que serão atualizados:
#    batch_size,
#    max_epochs,
#    optimizer,
#    criterion,
#    module__activation,
#    module __neurons,
#    module__initializer'''
    
    
params = {'batch_size':[10,20,30],
           'max_epochs':[50,100,200],
           'optimizer':[torch.optim.Adam, torch.optim.SGD],
           'criterion':[torch.nn.BCEWithLogitsLoss],
           'module__activation':[F.relu, F.tanh],
           'module__neurons':[8,16,32],
           'module__initializer':[torch.nn.init.uniform_,torch.nn.init.normal_]} 






grid_search  = GridSearchCV(classisicador_sklearn,
                            param_grid=params,
                            scoring='accuracy',
                            cv = 10)

grid_search = grid_search.fit(previsores, classe)


melhores_params = grid_search.best_params_
print(grid_search.best_params_)



