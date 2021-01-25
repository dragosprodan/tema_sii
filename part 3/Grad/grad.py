import numpy as np

def citire2(elem):
    f = open(elem, "r")
    a = f.read().split("\n")
    b = a[2:]
    c = []
    d = []
    for x in b:
        aux = x.split(",")
        for i in range(len(aux)-1):
            c.append([float(aux[i])])
        d.append([float(aux[-1])])
    return c,d

def averagesquare(results,trueResults):
    sum = 0
    for i in range(len(results)):
        sum = sum + (results[i]-trueResults[i])**2
    sum /= len(results)
    return sum

class Gradient:
    def __init__(self,X,Y,learnR,epoci):
        self.x = X
        self.y = Y
        self.learn = learnR
        self.epoci = epoci
        self.coef = np.zeros(shape=(len(self.x[0])+1,1))
        self.norm = None
        self.history = []

    def getErr(self):
        return (self.coef[-1]+np.dot(self.x,self.coef[:-1])[0],) - self.y

    def update(self):
        error = self.getErr()
        self.coef[:-1] = self.coef[:-1]-self.learn*np.dot(self.x.T, error.T)
        self.coef[-1] = self.coef[-1]-self.learn*sum(error.T)
        self.history.append(np.mean(error))


    def updateEpoch(self):
        for i in range(self.epoci):
            self.update()

    def listToY(self,lista):
        y = 0
        for i in range(len(lista)):
            y = y + (self.coef[i]*lista[i])
        return y