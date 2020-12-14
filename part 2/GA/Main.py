import numpy as np

        #print("Predicted result: ",result)
        #print("Actual result: ",realResult)
        #print()

from Population import Population

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

def main():
    x2 = read_input("hard_parkinson_train")
    y2 = np.array([[np.array(i[-1])] for i in x2.tolist()])

    population = Population(x2,y2,100,10) # 10000 1000
    chromosome=population.findSolution()

    x2 = read_input("hard_parkinson_test")
    y2 = np.array([[np.array(i[-1])] for i in x2.tolist()])

    x2=x2/chromosome.normX
    totalError=0
    for i in range(len(x2)):
        result=chromosome.getOutput(x2[i])
        realResult=y2[i]
        aux = abs(result-realResult) ** 2
        totalError=totalError+(aux)

        print("Predicted result: ",result)
        print("Actual result: ",realResult)
        print()

    print("Error: ",(totalError/len(x2)))

main()

