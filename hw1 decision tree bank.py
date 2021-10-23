import pandas as pd
import numpy as np
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv


#Dtree=open(r"TEST new.csv","r")
Dtree=open(r"Train new.csv","r")

reader=csv.reader(Dtree)
headers=reader.__next__()
print(headers)
featureList=[]
labelList=[]

for row in reader:
    labelList.append(row[-1])
    rowDict={}
    for i in range (0,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)
print(featureList)

vec=DictVectorizer()
x_data=vec.fit_transform(featureList).toarray()
print("x_data:"+str(x_data))

print("Labelist:"+str(labelList))

lb=preprocessing.LabelBinarizer()
y_data=lb.fit_transform(labelList)
print("y_data:"+str(y_data))

model=tree.DecisionTreeClassifier(criterion="entropy")
model.fit(x_data,y_data)

#model=tree.DecisionTreeClassifier(criterion="gini")
#model.fit(x_data,y_data)

x_test=x_data[0]
print("x_test:"+str(x_test))
predict=model.predict(x_test.reshape(1,-1))
print("predict:"+str(predict))

import graphviz
dot_data=tree.export_graphviz(model,out_file=None,
                              feature_names=vec.get_feature_names(),
                              class_names=lb.classes_,
                              filled=True,
                              rounded=True,
                              special_characters=True)
graph=graphviz.Source(dot_data)
#graph.render("label entropy")
#graph.render("label gini")

graph.render("label train entropy")
#graph.render("label train gini")

graph


