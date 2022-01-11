import numpy as np
from numpy.core.fromnumeric import mean
from clasificacion_laparoscopia.preprocesing import get_data_from_JIWSAWS, get_data_from_eScope
from clasificacion_laparoscopia.preprocesing import get_features
from clasificacion_laparoscopia.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from statistics import stdev as std
import matplotlib.pyplot as plt
from sklearn.svm import SVC


data=get_data_from_eScope()
clasificadores={
    'SVM':make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True)),
    'KNN':KNeighborsClassifier(n_neighbors=3),
    'RF':RandomForestClassifier(max_depth=2, random_state=0)
}


X=[]
Y=[]

for i,datum in enumerate(data):
    print("processing ",i+1," of ",len(data))
    x=get_features(X=datum['x'],Y=datum['y'],dT=1/30)
    X.append(x)
    Y.append(datum['y'])
    

kf = KFold(n_splits=4)
kf.get_n_splits(X)
X=np.array(X)
Y=np.array(Y)
for name,clasificador in clasificadores.items():
    ACC=[]
    AUC=[]
    F1=[]
    for train_index, test_index in kf.split(X):
        try:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            Clf=Classifier(clasificador)
            Clf.fit(X_train)
            Y_predict,Y_predict_global=Clf.predict(X_test)
            Proba=Clf.proba(X_test)
            acc = accuracy_score(y_test, Y_predict_global)
            auc = roc_auc_score(y_test,Y_predict_global)
            f1 = f1_score(y_test,Y_predict_global)
            ACC.append(acc)
            AUC.append(auc)
            F1.append(f1)
        except Exception as e:
            print(e)
    plt.plot(Proba[0])
    plt.show()
    print("acc=",mean(ACC),"+-",std(ACC))
    print("AUC=",mean(AUC),"+-",std(AUC))
    print("F1=",mean(F1),"+-",std(F1))