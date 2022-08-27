from sklearn.model_selection import train_test_split
from transformer import FeatureDataset
from sklearn import svm
from sklearn import metrics
import numpy as np


transformer_data = FeatureDataset("NewHI-Modnew.csv")

x_train, X_test, y_train, y_test = train_test_split(transformer_data.data,
                                                    transformer_data.target,
                                                    test_size=0.3,
                                                    random_state=25,
                                                    stratify=transformer_data.target)

clf = svm.SVC(kernel="linear")

clf.fit(x_train.astype('int'), y_train.astype('int'))

y_pred = clf.predict(X_test)

print(clf.predict(np.asarray([[30, 120, 1000, 2, 4, 3.4, 1.2, 0.1, 8, 0.012, 60, 0.035,0.02]])))
print(y_pred)
#TODO: Works but accuracy is only 0.765 76.95%. Okay but not great
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


