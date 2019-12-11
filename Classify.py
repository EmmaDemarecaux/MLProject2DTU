from Processing import classX, y, N, attributeNames
from sklearn import tree
import numpy as np
import graphviz


classNames = ['Poor', 'Lower', 'Middle', 'Upper']
classIndices = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,25):
        classIndices[i] = 0
    elif y[i] <= np.percentile(y,50):
        classIndices[i] = 1
    elif y[i] <= np.percentile(y,75):
        classIndices[i] = 2
    else: 
        classIndices[i] = 3
        
C = len(classNames)

dtcGini = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtcGini = dtcGini.fit(classX,classIndices)
out = tree.export_graphviz(dtcGini, out_file='2sampleGini.gvz', feature_names=attributeNames)
graphviz.render('dot','png','2sampleGini.gvz')
