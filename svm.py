from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import data


svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)
svm.fit(data.X_train, data.y_train)

y_pred = svm.predict(data.X_test)
accuracy = np.mean(y_pred == data.y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")