import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("all_one_hot.csv")

y = df.loc[:, 'win'].values
X = df.iloc[:, 1:340].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify = y
)