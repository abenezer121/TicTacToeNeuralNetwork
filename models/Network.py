import numpy as np
class Layer:
    def __init__(self,input,output):
        self.W = np.random.randn(input,output)* 0.01
        self.b = np.zeros(output)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self,x):
        z = np.dot(x , self.W)
        a = np.maximum(0,z)

        #save for backpropagation
        self.x = x
        self.a = a
        self.z = z
        return a
    def backward(self,da):
        dz = da * (self.z > 0) #compute the gradient of the relu function
       
        dx = np.dot(dz,self.W.T)
        dW = np.dot(np.array(self.x).T , dz)
        db = np.sum(dz , axis=0)

         #update the gradients
        self.dW = dW
        self.db = db
        return dx



class Network:
    def __init__(self,input_size,hidden_size,output_size):
      
        self.fc1 = Layer(input_size , hidden_size)
        self.fc2 = Layer(hidden_size , hidden_size)
        self.fc3 = Layer(hidden_size , output_size)
    def predict(self, x):
        # Forward pass to make predictions
        predictions = self.forward(x)

        # Apply softmax activation to get probabilities
        probabilities = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        
        return probabilities
    
        return predictions
    
    def forward(self,x):
        a1 = self.fc1.forward(x)
        a2 = self.fc2.forward(a1)
        a3 = self.fc3.forward(a2)
        return a3
    
    def backward(self,da):
        da2 = self.fc3.backward(da)
        da1 = self.fc2.backward(da2) 
        dx = self.fc1.backward(da1) 

        #this prevent division by zero
        #as funny as it seems maybe gradient exploading happening here
         # Clip gradients
        clip_value = 1.0  
        for layer in [self.fc1, self.fc2, self.fc3]:
                layer.dW = np.clip(layer.dW, -clip_value, clip_value)
                layer.db = np.clip(layer.db, -clip_value, clip_value)

        return dx
    
    def update(self, lr):
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db

        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db

        self.fc3.W -= lr * self.fc3.dW
        self.fc3.b -= lr * self.fc3.db
    