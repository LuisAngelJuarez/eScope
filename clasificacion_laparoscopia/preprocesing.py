from clasificacion_laparoscopia.Features import *
from pandas.core.indexes.base import Index
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
import random

def get_data_from_eScope():
    from os import listdir
    from os.path import isfile, join
    mypath='Report\Orientacion espacial\Experto'
    Expert = [mypath+"\\"+f for f in listdir(mypath) if isfile(join(mypath, f))]
    mypath='Report\Orientacion espacial\\Novato'
    Novice = [mypath+"\\"+f for f in listdir(mypath) if isfile(join(mypath, f))]
    Y=[0]*len(Novice)+[1]*len(Expert)
    paths=Novice+Expert
    data=[]
    for path,y in zip(paths,Y):
        df=pd.read_csv(path)[["x","y","z"]]
        datum={
                'x':df,
                'y':y
            }
        data.append(datum)
    random.shuffle(data)
    return data

def get_data_from_JIWSAWS():
    files=['Knot_Tying/meta_file_Knot_Tying.txt']
    #,'Needle_Passing/meta_file_Needle_Passing.txt','Suturing/meta_file_Suturing.txt']
    data=[]
    map={
        "N":0,
        "E":1
    }
    for i,metafile in enumerate(files):
        mat = np.loadtxt(metafile, 'str')
        for element in mat:
            if element[1]=="I":
                continue
            file=metafile.split('/')[0] + '/kinematics/AllGestures/' + element[0] + '.txt'
            df= pd.DataFrame(np.loadtxt(file))[[19,20,21]]
            df=df.rename(columns={19 : "x", 20 : "y", 21 : "z"})
            datum={
                'x':df,
                'y':map[element[1]]
            }
            data.append(datum)
    random.shuffle(data)
    return data

def get_features(X,Y,dT,NumFrames=100):
    dX=(X.diff()/dT).rename(columns={"x":"dx","y":"dy","z":"dz"})
    dX2=(dX.diff()/dT).rename(columns={"dx":"dx2","dy":"dy2","dz":"dz2"})
    dX3=(dX2.diff()/dT).rename(columns={"dx2":"dx3","dy2":"dy3","dz2":"dz3"})
    X=pd.concat([X,dX,dX2,dX3],axis=1)
    features=[]
    for i in range(X.shape[0]-NumFrames+1):
        kinSegm=X[i:i+NumFrames]
        AX=np.mean(kinSegm['x'].values)
        AY=np.mean(kinSegm['y'].values)
        AZ=np.mean(kinSegm['z'].values)
        ADX=np.mean(kinSegm['dx'].values)
        ADY =np.mean(kinSegm['dy'].values)
        ADZ =np.mean(kinSegm['dz'].values)
        PL=pathLenght(kinSegm['dx'].values,kinSegm['dy'].values,kinSegm['dz'].values,dT)
        DP=depthPerception(kinSegm['dy'].values,kinSegm['dz'].values,dT)
        MS=motionSmoothness(kinSegm['dx3'].values,kinSegm['dy3'].values,kinSegm['dz3'].values,dT)
        AV=averageVelocity(kinSegm['dx'],kinSegm['dy'],kinSegm['dz'])
        AA = averageAceleration(kinSegm['dx2'], kinSegm['dy2'], kinSegm['dz2'])
        features.append([AX,AY,AZ,ADX,ADY,ADZ,PL,DP,MS,AV,AA,Y])
    features=np.array(features)
    columns=["AX","AY","AZ","ADX","ADY","ADZ","PL","DP","MS","AV","AA","target"]
    features=pd.DataFrame(features,columns=columns)
    return features