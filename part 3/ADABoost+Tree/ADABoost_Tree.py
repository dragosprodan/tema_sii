from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import tree
import sys
import time

df = pd.read_csv("parkinsons_updrs.csv")
df = df.sample(frac=1).reset_index(drop=True)
X = df[["age", "sex", "Jitter_proc", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP", "Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]]
Y = df[["total_UPDRS"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

n_splits = 10
crossvalidation = KFold(n_splits, shuffle=True)

best_depth = 1
best_score = - sys.float_info.max
for depth in range (1,16):
     tree_regressor=tree.DecisionTreeRegressor(max_depth=depth,random_state=1)
     if tree_regressor.fit(X,Y).tree_.max_depth<depth:
         break
     score=cross_val_score(tree_regressor,X,Y,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1)
     if score.mean() > best_score:
         best_score = score.mean()
         best_depth = depth

start_time = time.time()
tree = tree.DecisionTreeRegressor(max_depth=best_depth,random_state=1)
ada2=AdaBoostRegressor(base_estimator=tree, n_estimators=500,learning_rate=0.001,random_state=1)
ada2.fit(X_train,Y_train.values.ravel())
score=cross_val_score(ada2,X_train,Y_train.values.ravel(),scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)

print("cross_val_mean_squared_error=", score.mean())

Y_pred = ada2.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
print("mean_squared_error=", mean_squared_error(Y_test, Y_pred))