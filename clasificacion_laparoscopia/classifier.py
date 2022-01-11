import pandas as pd
from scipy import stats


class Classifier:
    def __init__(self,clf):
        self.clf=clf

    def fit(self,X):
        X,Y=self.transform_data(X)
        self.clf.fit(X,Y)

    
    def predict(self,X):
        import numpy as np
        import statistics
        Y_predict=[]
        Y_predict_global=[]
        for element in X:
            df=element.dropna()
            df=df.drop(columns=['target'])
            y_predict=self.clf.predict(df.values)
            y_predict_global=stats.mode(y_predict)[0][0]
            Y_predict.append(y_predict)
            Y_predict_global.append(y_predict_global)
        return Y_predict,Y_predict_global

    def proba(self,X):
        import numpy as np
        import statistics
        Proba=[]
        for element in X:
            df=element.dropna()
            df=df.drop(columns=['target'])
            proba=self.clf.predict_proba(df.values)[:,1]
            Proba.append(proba)
        return Proba
    
    def transform_data(self,X):
        df=pd.DataFrame()
        for element in X:
            df=pd.concat([df,element],axis=0)
        df=df.reset_index(drop=True)
        df=df.dropna()
        Y=df['target'].values
        X=df.drop(columns=['target']).values
        return X,Y