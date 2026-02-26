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
print(f"The shape of the samples and features is: {X.shape}")
print(f"The shape of the labels: {y.shape}")

#splitting the data
#random state is for reproducibility, shuffle is 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#creating, training and testing the model 
model = MLPClassifier(hidden_layer_sizes=1000, activation='relu', solver='adam', max_iter=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the model: {accuracy:.4f}")

#visualizing the data
#working with images so use imshow
#using a single test sample and reshaping it to its original form
X_test_sample = X_test[60]
print(f"The shape of the sample is: {X_test_sample.shape}")
X_test_reshape = X_test_sample.reshape(8, 8)
plt.imshow(X_test_reshape, cmap='gray')
plt.axis("off")
plt.show()
y_test_sample = y_test[60]
y_pred_sample = y_pred[60]
print(f"The actual number displayed is: {y_test_sample}")
print(f"The number the model predicted is: {y_pred_sample}")