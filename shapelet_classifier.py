import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RanF
from c45 import C45
from sklearn.naive_bayes import GaussianNB
#from rotation_forest import RotationForestClassifier as RotF
from sktime.classification.sklearn import RotationForest as RotF
#from sklearn.ensemble import VotingClassifier
    
class VotingClassifier:
    def __init__(
        self,
        estimators,
        weights,
    ):
        self.estimators=estimators
        self.weights=weights
        
    def fit(self, X, y):
        self.estimators = [(name, clone(clf).fit(X,y)) for (name, clf) in self.estimators]
        return self
    
    def predict(self,X):        
        predictions = np.asarray([est.predict(X) for (_,est) in self.estimators]).transpose()
        maj = np.apply_along_axis(
            lambda x:np.argmax(np.bincount(x, weights=self.weights)),
            axis=1,
            arr=predictions,
        )
        
        return maj
        
def _get_weight(X,y):
    clf1 = svm.SVC(kernel='linear')
    clf2 = svm.SVC(kernel='poly')
    clf3 = RanF(n_estimators=200,max_depth=10)
    clf4 = KNN(n_neighbors=3)
    clf5 = C45()
    clf6 = GaussianNB()
    clf7 = RotF(n_estimators=100)
    
    estimators = [('SVM_L',clf1),('SVM_Q',clf2),('RanF',clf3),('KNN',clf4),('C45',clf5),('NB',clf6),('RotF',clf7)]
    #box = np.array([1,1,2,2,4,4])
    #box = np.array([1,1,4,4,2,2])
    #box = np.array([4,4,2,2,1,1])
    #box = np.array([1,2,4,1,2,4])
    acc = []
    for (label,clf) in estimators:
        acc.append(np.mean(cross_val_score(clf,X,y,scoring='accuracy',cv=5)))
    #weights = np.array([box[np.argwhere(np.argsort(acc)==i)] for i in range(7)]).reshape(-1)
    weights = np.array(acc)
    #classifier = estimators[np.argmax(weights)][0]
    return estimators,weights,np.mean(acc*weights)

def classifier(X,y):
    estimators,weights,score = _get_weight(X,y)
    eclf = VotingClassifier(estimators=estimators,
                            #voting='hard',
                            weights=weights,
                           )
    return score,eclf

def get_srank(ls):
    result=[0]
    for i in range(len(ls)):
        if ls[i]>ls[result[-1]]:
            result+=[i]
    return np.array(result)        
        
        
        
    