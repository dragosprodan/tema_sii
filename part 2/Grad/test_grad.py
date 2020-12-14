import numpy as np
import grad as grad
import matplotlib.pyplot as plt

def read_input(elem):
    f = open(elem, "r")
    a = f.read().split("\n")
    b = a[2:]
    c = []
    d = []
    for x in b:
        aux = x.split(",")
        test = []
        for i in range(len(aux)-1):
            test.append(float(aux[i]))
        try:
            d.append(float(aux[-1]))
            c.append(test)
        except:
            print("")


    return c,d

def toFile(elem, data):
    f = open(elem, "w")
    out = ""
    out += str(data[0])
    out += "\n"
    out += str(data[1])
    f.write(out)

def averagesquare(results,trueResults):
    sum = 0
    for i in range(len(results)):
        sum = sum + (results[i]-trueResults[i])**2
    sum /= len(results)
    return sum

fin = []

lista = read_input("hard_parkinson_train")
rez = grad.Gradient(np.array(lista[0]),np.array(lista[1]),0.00000000002,1000)
rez.updateEpoch()
aux = []
for x in read_input("hard_parkinson_test")[0]:
    aux.append(rez.listToY(x))

fin.append(averagesquare(aux,read_input("hard_parkinson_test")[-1]))

plt.plot(rez.history)
plt.show()
print(fin)
