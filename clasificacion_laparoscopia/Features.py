import  numpy as np
from math import sqrt
from math import pow
def pathLenght(vx,vy,vz,dT):
    PL=np.power(vx,2)+np.power(vy,2)+np.power(vz,2)
    PL=np.sum(PL)*dT
    return PL

def depthPerception(vy,vz,dT):
    DP=np.power(vy,2)+np.power(vz,2)
    DP=np.sum(DP)*dT
    return DP

#def depthAlongTrocar()
def motionSmoothness(dx3,dy3,dz3,dT):
    T=dx3.shape[0]
    J=sqrt((1/2)*np.sum(np.power(np.power(dx3,2)+np.power(dy3,2)+np.power(dz3,2),2)*dT))
    MS=J/T;
    return MS
#Aceleration
def averageVelocity(dx,dy,dz):
    return np.mean(np.sqrt(dx*dx+dy*dy+dz*dz))

def averageAceleration(dx2,dy2,dz2):
    return np.mean(np.sqrt(dx2*dx2+dy2*dy2+dz2*dz2))

def economyOfArea(X,Y,PL):
    EOA=sqrt((np.max(X)-np.min(X))*(np.max(Y)-np.min(Y)))/PL
    return EOA

def economyOfVolume(X,Y,Z,PL):
    EOV=((np.max(X)-np.min(X))*(np.max(Y)-np.min(Y))*(np.max(Z)-np.min(Z)))**(1/3)
    EOV=EOV/PL
    return EOV

def energyInTheArea(X,Y):
    EA=(np.sum(X*X)+np.sum(Y*Y))
    EA=EA/((np.max(X)-np.min(X))*(np.max(Y)-np.min(Y)))
    return EA;

def energyInTheVolume(X,Y,Z):
    EV=np.sum(X*X)+np.sum(Y*Y)+np.sum(Z*Z)
    EV=EV/((np.max(X)-np.min(X))*(np.max(Y)-np.min(Y))*(np.max(Z)-np.min(Z)))
    return EV