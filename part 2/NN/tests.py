from sklearn import neural_network
import numpy as np


class ANN:
    def __init__(self,  matrix, nrCol, n, matrixT, nrColT, nT):
        self.matrix = matrix
        self.nrCol = nrCol
        self.n = n
        self.matrixT = matrixT
        self.nrColT = nrColT
        self.nT = nT

def read_input(elem):
    f = open(elem, "r")
    a = f.read().split("\n")
    b = a[2:]
    c = []
    for x in b:
        if len(x) > 1:
            aux = x.split(",")
            c.append(np.array([float(i) for i in aux]))
    f.close()
    return np.array(c)

def norm(x):
    normX = np.linalg.norm(x)
    x = x / normX
    return x, normX

def main():
    x = read_input("hard_parkinson_train")
    y = np.array([np.array(i[-1]) for i in x.tolist()])
    x = np.array([np.array(i[:-1]) for i in x.tolist()])

    x2 = read_input("hard_parkinson_test")
    y2 = np.array([np.array(i[-1]) for i in x2.tolist()])
    x2 = np.array([np.array(i[:-1]) for i in x2.tolist()])

    x, normX = norm(x)
    y, normY = norm(y)
    x2, normX2 = norm(x2)
    y2, normY2 = norm(y2)

    network = neural_network.MLPRegressor(hidden_layer_sizes=5, activation='relu', max_iter=10000)
    network.fit(x, y)
    rez = network.predict(x2)
    sol = 0

    for i in range(len(rez)):
        rez[i] = rez[i] * normY2
        y2[i] = y2[i] * normY2
        sol += abs(rez[i] - y2[i]) ** 2

    print(sol/len(rez))

main()