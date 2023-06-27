import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score


adult_df = pd.read_csv("C:\\Users\\LATITUDE\\Desktop\\RBL\\dataaa.csv")

adult_df.columns = ['sex','age','Medu','Fedu','studytime','failures','famrel','freetime','health','absences','address','guardian','G3' ]

adult_df_rev = adult_df

le = preprocessing.LabelEncoder()
sex_cat = le.fit_transform(adult_df.sex)
address_cat = le.fit_transform(adult_df.address)
guardian_cat   = le.fit_transform(adult_df.guardian)

adult_df_rev['sex_cat'] = sex_cat
adult_df_rev['address_cat'] = address_cat
adult_df_rev['guardian_cat'] = guardian_cat

dummy_fields = ['sex','address','guardian']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

adult_df_rev = adult_df_rev.reindex(['sex_cat','age','Medu','Fedu','studytime','failures','famrel','freetime','health','absences','address_cat','guardian_cat','G3'], axis=1)

features = adult_df_rev.values[:, :11]
target = adult_df_rev.values[:, 11]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=10)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)
#print(target_pred)

#accuracy
print("Accuracy is",accuracy_score(target_test, target_pred, normalize = True))
a=accuracy_score(target_test, target_pred, normalize = True)
a="{:.2f}".format(a)
#classification error
print("Classification error is",1- accuracy_score(target_test, target_pred, normalize = True))
b=1- accuracy_score(target_test, target_pred, normalize = True)
b="{:.2f}".format(b)
#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(target_test, target_pred, labels=None, average =  'micro', sample_weight=None))
c=sklearn.metrics.recall_score(target_test, target_pred, labels=None, average =  'micro', sample_weight=None)
c="{:.2f}".format(c)
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(target_test, target_pred,labels=None, average =  'micro', sample_weight=None))
d=1 - sklearn.metrics.recall_score(target_test, target_pred,labels=None, average =  'micro', sample_weight=None)
d="{:.2f}".format(d)


a=float(a)*100
b=float(b)*100
c=float(c)*100
d=float(d)*100

x = np.array(["Accuracy", "ClassificationError", "Sensitivity", "Specificity"])
y = np.array([a,b,c,d])
plt.bar(x,y)
plt.show()
