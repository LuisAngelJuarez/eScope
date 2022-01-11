from clasificacion_laparoscopia.preprocesing import get_data_from_JIWSAWS
from clasificacion_laparoscopia.preprocesing import get_features
from clasificacion_laparoscopia.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC


data=get_data_from_JIWSAWS()
clasificadores={
    'SVM':make_pipeline(StandardScaler(), SVC(gamma='auto')),
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
for name,clasificador in clasificadores.items():
    print(name)
    Clf=Classifier(clasificador)
    Clf.fit(X_train)
    Y_predict,Y_predict_global=Clf.predict(X_test)
    acc = accuracy_score(y_test, Y_predict_global)
    AUC = roc_auc_score(y_test,Y_predict_global)
    F1 = roc_auc_score(y_test,Y_predict_global)
    print(Y_predict_global,y_test)
    print("acc=",acc)
    print("AUC=",AUC)
    print("F1=",F1)
    plt.plot(Y_predict[0])
    plt.show()