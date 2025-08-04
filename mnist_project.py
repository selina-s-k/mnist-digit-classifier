#Training a model using MNIST dataset 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

#Load data
data = pd.read_csv("C:/Users/katum/Documents/SPRING2025/STATISTICAL LEARNING/Midsem_Project/train.csv")

#Split into training and validation sets
X = data.drop("label", axis=1).values
y = (data["label"] == 5).astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.8, random_state=42)

#Normalize
X_train = X_train / 255.0
X_val = X_val /255.0

#Add bias term 
X_train = np.insert(X_train, 0, 1, axis=1)
X_val = np.insert(X_val, 0, 1, axis=1)

#define functions
def sigmoid(u):
	return 1 / (1 + np.exp(-u))

def bce(p, q):
	return - p * np.log(q) - (1 - p) * np.log(1 - q)

def sgd(X_train, y_train, lr, batch_size, epochs):
	N, d = X_train.shape
	beta = np.zeros(d) 
	L_vals = []

	for e in range(epochs):
		indices = np.random.permutation(N)
		total_loss = 0 

		for i in range(0, N, batch_size):
			batch_index = indices[i:i + batch_size]
			X_batch = X_train[batch_index]
			y_batch = y_train[batch_index]

			y_pred = sigmoid(np.dot(X_batch, beta))
			gradient = np.dot(X_batch.T, (y_pred - y_batch)) / batch_size
			beta -= lr *gradient

			total_loss += np.sum(bce(y_batch, y_pred)) / batch_size
		L_vals.append(total_loss / (N // batch_size))

		if e % 10 == 0:
			print(f'Epoch {e}: Loss = {L_vals[-1]}')
	return beta, L_vals

# Train the model and record the loss values
lr = 0.01 
batch_size = 50 
epochs = 10  

beta, L_vals = sgd(X_train, y_train, lr=lr, batch_size=batch_size, epochs=epochs)

# Plot the cost function value vs. epoch
plt.plot(L_vals)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss vs Epoch (Learning Rate = {lr}, Batch Size = {batch_size})')
plt.show()

#Make predictions
y_val_pred = sigmoid(np.dot(X_val, beta)) > 0.5

#Check accuracy 
accuracy = np.mean(y_val_pred == y_val)
print(f'Accuracy: {accuracy:.4f}')

