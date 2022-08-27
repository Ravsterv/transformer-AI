from sklearn.model_selection import train_test_split
from transformer import FeatureDataset
from sklearn.neighbors import KNeighborsClassifier
#split dataset into train and test data

transformer_data = FeatureDataset("NewHI-Modnew.csv")

x_train, X_test, y_train, y_test = train_test_split(transformer_data.data,
                                                    transformer_data.target,
                                                    test_size=0.3,
                                                    random_state=12,
                                                    stratify=transformer_data.target)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train.astype("int"), y_train.astype("int"))

print(knn.predict(X_test))

# Accuracy Test
print("Accuracy", knn.score(X_test, y_test))


