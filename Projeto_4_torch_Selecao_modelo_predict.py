# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:32:15 2022

@author: Vinicius
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torch


# Set sementes
np.random.seed(123)
torch.manual_seed(123)


# Base de dados
previsores = pd.read_csv('Datasets/entradas_breast.csv')
classe = pd.read_csv('Datasets/saidas_breast.csv')

# Convertendo para torch

previsores = torch.tensor(np.array(previsores), dtype=torch.float)
classe = torch.tensor(np.array(classe['0']),dtype=torch.long)

data_train = torch.utils.data.TensorDataset(previsores, classe)
train_loader= torch.utils.data.DataLoader(data_train, batch_size=10,shuffle=True)


# Contrucão da Rede


class ClassificadosTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense0 =  torch.nn.Linear(30,16)
        torch.nn.init.uniform_(self.dense0.weight)
        self.activation0 = torch.nn.ReLU()
        self.dropout0 = torch.nn.Dropout(p=0.2)
        
        self.dense1 = torch.nn.Linear(16, 16)
        torch.nn.init.uniform_(self.dense1.weight)
        self.activation1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        
        self.dense2 =torch.nn.Linear(16,1)
        torch.nn.init.uniform_(self.dense2.weight) 
        self.activation2 = torch.nn.ReLU()
        
        self.output = torch.nn.Sigmoid()
        
        
    def forward(self,X):
        X = self.dense0(X)
        X = self.activation0(X)
        X = self.dropout0(X)
        X = self.dense1(X)
        X = self.activation1(X)
        X = self.dropout1(X)
        X = self.dense2(X)
        X = self.activation2(X)
        X = self.output(X)
        
        return X
        
        
        
classificador= ClassificadosTorch()
        
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(classificador.parameters(), lr = 0.001, 
                             weight_decay = 0.0001)
        
for epoch in range(200):
    running_loss = 0.
    
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()        

        outputs = classificador(inputs)
        loss = criterion(outputs, labels)
        loss.backward()    
        optimizer.step()

        running_loss += loss.item()

    print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))