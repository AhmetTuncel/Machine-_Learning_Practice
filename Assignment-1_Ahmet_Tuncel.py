# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:12:49 2018

@author: AHMET.TUNCEL

"""
"""
1. Please run the code for digits dataset that we have worked during the last class. Compare and discuss the outputs for the raw and scaled data as we did in the lab. Please assign the random state value randomly (you need to provide a random integer assignment).
"""
# Answer:


#Loading lib

import pylab as pl
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pylab as plt



# Unscale version 

#Load digits data
np.random.seed(123)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted


# Split train test data
y = digits.target
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# Set gnb for naive bayes algorith
gnb = GaussianNB(priors=None)

# Fit model with
fit = gnb.fit(X_train, y_train)
predicted = fit.predict(X_test)

print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy 





# Scale version 

data = scale(digits.data)  # the data is scaled with the use of z-score

X1 = data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.3, random_state=10)


fit = gnb.fit(X_train1, y_train1)
predicted = fit.predict(X_test1)
print(confusion_matrix(y_test1, predicted))
print(accuracy_score(y_test1, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)

"""
As you can see raw data most higher accuracy output thats why feature data is not larger numeric magnitude.  
"""


"""
2. Please run the same code for 5 different test/train sizes such as 0.1, 0.3, 0.5 for raw dataset. Compare and discuss the obtained results. Which one is the best and why do you think so?
"""


#Load digits data
np.random.seed(123)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted


# Split train test data
y = digits.target
X = digits.data


#test/train size 0.1

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.1, random_state=10)


#test/train size 0.3

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=10)


#test/train size 0.5

X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.5, random_state=10)


# results 0.1

# Set gnb for naive bayes algorithm
gnb = GaussianNB(priors=None)

# Fit model with
fit = gnb.fit(X_train1, y_train1)
predicted = fit.predict(X_test1)

print(confusion_matrix(y_test1, predicted))
print(accuracy_score(y_test1, predicted)) # the use of another function for calculating the accuracy 


fit = gnb.fit(X_train1, y_train1)
predicted = fit.predict(X_test1)

print(confusion_matrix(y_test1, predicted))
print(accuracy_score(y_test1, predicted)) # the use of another function for calculating the accuracy 



# results 0.3
gnb = GaussianNB(priors=None)

# Fit model with
fit = gnb.fit(X_train2, y_train2)
predicted = fit.predict(X_test2)

print(confusion_matrix(y_test2, predicted))
print(accuracy_score(y_test2, predicted)) # the use of another function for calculating the accuracy 




# results 0.5

gnb = GaussianNB(priors=None)

# Fit model with
fit = gnb.fit(X_train3, y_train3)
predicted = fit.predict(X_test3)

print(confusion_matrix(y_test3, predicted))
print(accuracy_score(y_test3, predicted)) # the use of another function for calculating the accuracy 


"""

Result accurcy score so close (also test/train size 0.1 has higher accurcy) but if you can trust this result then you probably decide incorrect result. As much as possible you find the representing all the data in your test data set. Also if you have small data set and also not proporly seperetad to test and train , your classification or prediction result may be bias or over fitting. 

"""


"""
3. Please explain how the Gaussian Naive Bayesian model. How does Gaussian Naive Bayes algorithm work in the digits dataset? Assume that you are explaining it to a close friend of yours who has no background in data science but she is highly willing to learn it. Please do not forget to refer to the features, independence assumption, and labels.
"""

# Answer:

"""

Gaussian (Normal distribution) fitted the mean and the standard deviation from training data. In the digits data set; we calculate mean and standart devitation of input values for every feature and then probabilities for input values for every label that using a frequency. This importing thing you remove outliers data when you want to 
If the input variables are real-valued, a Gaussian distribution is assumed. In which case the algorithm will perform better if the univariate distributions Gaussian. 

"""


"""
# Question :

4a. Please run the SVC (linear) code for iris dataset (that we worked during the class). Please perform the same algorithm for 4 different C values that you will decide. Then you should plot the outputs in the 2 x 2 plots (you need to get rid of the plots for label propagation but instead use SVC).
"""

# Load library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Load iris data set and creating data and target object (features and label)
iris = load_iris()
X = iris.data
y = iris.target

# Create for X_2d and y_2d object for drawing plot (features and label)

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# scale feature data set for fitting model
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# An initial search, a logarithmic grid with basis
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)


# Set C and gamma values and create empty classifiers list
C_2d_range = [1e-2, 1, 10, 1e2]
gamma_2d_range = [1e-1, 1,10, 1e2]
classifiers = []

# fit in C_2d_range list values and append to classifiers list 
for C in C_2d_range:
    clf = SVC(C=C)
    clf.fit(X_2d, y_2d)
    classifiers.append((C, clf))
    

# prepera sketch of graphic
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))

# draw a plot for every C value from model

for (k, (C, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("C=10^%d" % (np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')



"""
# Question :

4b. Please run the same SVC (linear kernel) for the iris dataset (that we worked during the class). Please perform the same algorithm for 4 different gamma values that you will decide. Then you should plot the outputs in the 2 x 2 plots (again you need to get rid of the plots for label propagation but instead use SVC).
"""
# Loading library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# Load iris data set and creating data and target object (features and label)
iris = load_iris()
X = iris.data
y = iris.target


# Create for X_2d and y_2d object for drawing plot (features and label)

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1


# scale feature data set for fitting model
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# An initial search, a logarithmic grid with basis
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)


# Set C and gamma values and create empty classifiers list

C_2d_range = [1e-2, 1, 10, 1e2]
gamma_2d_range = [1e-1, 1,10, 1e2]
classifiers = []


# fit in gamma_2d_range list values and append to classifiers list 
#    
for gamma in gamma_2d_range:
    clf = SVC(gamma=gamma)
    clf.fit(X_2d, y_2d)
    classifiers.append((gamma, clf))



# prepera sketch of graphic
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))

# draw a plot for every C value from model


for (k, (gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d" % (np.log10(gamma)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')




'''
Question :

5. List all the outputs for accuracy values that you have found in question-4a and question-4b. Compare and discuss the results of the 8 models that you obtained.

'''

# Answer :

# scores come from GridSearchCV function
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),

                                                     len(gamma_range))
# print best scores                                                     
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# visualize parameters and Validation accuracy
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

"""
About accury observation ; we can easily observe on "Validation accuracy" plot. If you want to high accuracy score , 

The best parameters are 'C': 1.0, 'gamma': 0.10000000000000001 with a score of 0.97

"""



"""
Question :

6. Please explain the need for semi-supervised learning methods. Please try to find a concrete example and discuss it. (At most 100 words).
"""

# Answer :

"""
Semi-supervised learning, some examples are not labeled. So if you want to better understand new sample data ;you may use unlabelled data. In generally well perform we have a very small amount of labeled data and also a large amount of unlabeled data.


"""

"""
7. Please provide two concrete examples: one of them is more suitable for applying Naïve Bayes algorithm and the other for Support Vector Machine algorithm. Provide a very rough comparison accordingly.
"""

# Answer :

"""
- Naive Bayes is widely used in fields such as text classification and spam filtering.
- SVM is generally used in fields such as face recognition, handwriting recognation and time-series forecasting.

- If your feature data is a linear property and then you choose SVM algorithm.
- SVM algorithm is effective in high dimensional spaces
- If your feature data is a probability model with a specific independence structure and then you choose Naive Bayes algorithm.
- Also if you have a small data set ,bias data set and low variance then Naive Bayes is more advantageous.
"""