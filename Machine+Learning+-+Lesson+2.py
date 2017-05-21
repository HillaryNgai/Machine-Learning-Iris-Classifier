
# coding: utf-8

# In[22]:

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

print(iris.data[0])
print(iris.target[0])
"""
for i in range(len(iris.target)):
    print("Example %d: Label %s, Features %s" % (i, iris.target[i], iris.data[i]))
"""  
#testing data is separate from training data so that we can test the classifer with data that it has never seen before
#test index
test_idx = [0,50,100]

#training data 
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testin data 
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()

clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

