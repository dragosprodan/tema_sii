import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Population import Population

def main():
    df = pd.read_csv("parkinsons_updrs.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    X = df[
        ["age", "sex", "Jitter_proc", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP", "Shimmer", "Shimmer_dB",
         "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]]
    Y = df[["total_UPDRS"]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    population = Population(X_train.values,Y_train.values,100,10) # 10000 1000
    chromosome=population.findSolution()

    X_test = X_test.values/chromosome.normX
    Y_test = Y_test.values
    totalError=0
    for i in range(len(X_test)):
        result=chromosome.getOutput(X_test[i])
        realResult=Y_test[i]
        aux = abs(result-realResult) ** 2
        totalError=totalError+(aux)

        print("Predicted result: ",result)
        print("Actual result: ",realResult)
        print()

    print("Error: ",(totalError/len(X_test)))

main()

