from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import data

tree_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, random_state = 1)

tree_model.fit(data.X_train, data.y_train)

y_pred = tree_model.predict(data.X_test)
accuracy = np.mean(y_pred == data.y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

forest = RandomForestClassifier(n_estimators = 50, random_state = 1, n_jobs = 2)
forest.fit(data.X_train, data.y_train)

y_pred = forest.predict(data.X_test)
accuracy = np.mean(y_pred == data.y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")