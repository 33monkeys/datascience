from sklearn import tree
import numpy as np
import pandas as pd

data = pd.read_csv('heights_weights_genders.csv')

X = pd.concat([data.Height * 0.0254, data.Weight * 0.453592], axis=1)
y = data['Gender']

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

with open("dtree2.dot", 'w') as file:
    tree.export_graphviz(clf, out_file=file, feature_names=X.columns, filled=True, rounded=True,
                         special_characters=True, class_names=['Male', 'Female'])

print(clf.predict([[1.60, 55]]))
