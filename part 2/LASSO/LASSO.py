from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

df = pd.read_csv("parkinsons_updrs.csv")
df = df.sample(frac=1).reset_index(drop=True)
X = df[["age", "sex", "Jitter_proc", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP", "Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]]
Y = df[["total_UPDRS"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

model = Lasso(alpha=0.7, max_iter= 100)

start_time = time.time()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("--- %s seconds ---" % (time.time() - start_time))
print("mean_squared_error=", mean_squared_error(Y_test, Y_pred))


print("x")