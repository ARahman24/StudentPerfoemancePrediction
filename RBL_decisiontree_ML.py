import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score

start_time = time.time()


df = pd.read_csv("C:\\Users\\LATITUDE\\Desktop\\RBL\\dataaa.csv")
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1})

predictors = df.values[:, 0:11]
targets = df.values[:,12]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size= 0.25)


#print(pred_train.shape)
#print(pred_test.shape)
#print(tar_train.shape)
#print(tar_test.shape)

features = list(df.columns[:11])

classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')

classifier = classifier.fit(pred_train,tar_train)

predictions = classifier.predict(pred_test)

#print(sklearn.metrics.confusion_matrix(tar_test, predictions))

#classification accuracy
print("Accuracy of training dataset is {:.2f}".format(classifier.score(pred_train,tar_train)))
a="{:.2f}".format(classifier.score(pred_train,tar_train))
print("Accuracy of test dataset is {:.2f}".format(classifier.score(pred_test,tar_test)))
b="{:.2f}".format(classifier.score(pred_test,tar_test))

#print(accuracy_score(tar_test, predictions, normalize = True))

#error rate
print("Error rate is",1- accuracy_score(tar_test, predictions, normalize = True))
c=1- accuracy_score(tar_test, predictions, normalize = True)
c="{:.2f}".format(c)

#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))
d=sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None)
d="{:.2f}".format(d)

#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))
e=1 - sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None)
e="{:.2f}".format(e)

b=float(b)*100
c=float(c)*100
d=float(d)*100
e=float(e)*100


x = np.array(["Accuracy", "ClassificationError", "Sensitivity", "Specificity"])
y = np.array([b,c,d,e])
plt.bar(x,y)
plt.show()
