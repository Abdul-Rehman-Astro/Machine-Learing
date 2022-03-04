# Machine-Learing
This is where my machine learning journey begins.

# This is first Logistic Regression code to tell whether a given flower is iris or virginica or versicolor

#Train a loigistic Regression classifier to predict whether a flower is 
#iris virginica or not
from sklearn import datasets
import numpy as np

iris= datasets.load_iris()
# print(list(iris.keys()))
X= iris["data"][:,3:]
# print(X)
Y = (iris["target"])
#Y= (iris["target"] ==2 )
# print(Y)
Y= (iris["target"] ==2).astype(np.int64)# either true or false 
print(Y)


import numpy as np

#Y= (iris["target"] ==2).astype(np.int64)
print(Y)


#train a logistic regression classifier
from sklearn.linear_model import LogisticRegression 

clf = LogisticRegression()  # clf is classifier 

clf.fit(X,Y)# it will fit all the data form X and Y into LogisticRegression


example = clf.predict(([[2.6]]))

print(example)

#using matplotlib for  plot the visulization of data
import matplotlib.pyplot as plt

X_new= np.linspace(0,3,1000)# it will give 1000 points in between (0 , 3)
#print(X_new)

X_new= np.linspace(0,3,1000).reshape(-1,1)# to convert all those points into 1 column
                                          #array reshaped with -1 rows and 1 columns :
Y_prob= clf.predict_proba(X_new)# predict_proba predict probability
# print(Y_prob)

plt.plot(X_new,Y_prob[:,1],"g-",label="virginica")
plt.show()
