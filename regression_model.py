import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import data

lr = LogisticRegression(C = 100.0, solver = 'lbfgs')
lr.fit(data.X_train, data.y_train)
y_pred = lr.predict(data.X_test)

accuracy = np.mean(y_pred == data.y_test)
print(f"sklearn Accuracy: {accuracy * 100:.2f}%")




