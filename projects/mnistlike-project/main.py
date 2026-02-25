from sklearn.model_selection import train_test_split #library for splitting the data
from sklearn import datasets #library for acquiring the dataset
from sklearn.neural_network import MLPClassifier #for classifying the data*
from sklearn.metrics import accuracy_score #for measuring accuracy
import matplotlib.pyplot as plt #for data visualisation

#loading the dataset
digits = datasets.load_digits()

#extracting the features and labels
X = digits.data #this contains the input images
y = digits.target #this contains the digit values

#viewing the shape of the data
print(X.shape)
print(y.shape)

#splitting the data
#random state is for reproducibility, shuffle is 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#creating, training and testing the model 
model = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', max_iter=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the model: {accuracy:.4f}")

#visualizing the data
#working with images so use imshow
#using a single test sample and reshaping it to its original form
X_test_sample = X_test[0]
X_test_reshape = X_test_sample.reshape(-1, 1)
plt.imshow(X_test_reshape, cmap='gray')
plt.axis("off")
plt.show()
