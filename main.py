from util.preprocess import DataPreprocessor
from models.Network import Network
import numpy as np
import sys
import torch.nn as nn



dataprocessor = DataPreprocessor()

#load the preprocessed data
X , y  = dataprocessor.generate('data/tictak.csv')



def loss( y_pred , y_true):
        m = np.array(y_true).shape[0] 
        y_true = np.argmax(y_true, axis=1)

        p = np.exp(y_pred) / np.sum(np.exp(y_pred) , axis = 1 , keepdims=True) # the softmax probability
        
        log_likelihood = -np.log(p[range(m) , y_true]) 
        loss = np.sum(log_likelihood) / m 
        return loss
    
def grad_loss( y_pred , y_true):
        m = np.array(y_true).shape[0]

        p = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        grad = p.copy()

        # Loop over each sample in the batch to update gradients
        for i in range(m):
                grad[i, y_true[i]] -= 1

        grad /= m

        return grad


net = Network(9,100,9)
epochs = 100
learning_rate = 0.001
batch_size = 25


# train the network
for epoch in range (epochs):

    #shuffle the train set indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,len(X) , batch_size):
        batch_indices = [indices[i:i+batch_size]][0]
        X_batch = []
        y_batch = []
        # get the current batch data and labels
       
        for batchindex in batch_indices:
          
                X_batch.append(X[batchindex])  
                y_batch.append(y[batchindex])  

        # forward pass
        y_pred = net.forward(X_batch)
        
        #backward pass
        da = grad_loss(y_pred , y_batch) # the gradient of the loss function with respect ot the predicted labels
        net.backward(da) # the gradient of the network with respect to the input

        net.update(learning_rate) # parameter update

    #evalute the network on the train sets
    y_pred = net.forward(X)
    train_loss = loss(y_pred , y)
    print(f"Epoch {epoch+1} : Train Loss = {train_loss}")

X = [[0,0,0,0,1,0,0,0,0]]
y_pred = net.predict(X)

print(y_pred)
