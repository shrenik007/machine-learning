import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Open the training dataset as a dataframe and perform preprocessing
dataset = np.genfromtxt("data/dataFile.csv", delimiter=",")
feature_data = dataset[:, :-1]
label_data = dataset[:, -1]

# standardization of the data
scaler = StandardScaler()
scaler.fit(feature_data)
feature_data = scaler.transform(feature_data)

train_features, test_features, train_labels, test_labels = train_test_split(feature_data, label_data, test_size=0.2,
                                                                            random_state=0)
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(train_features, train_labels)

predicted = neigh.score(test_features, test_labels)
print('Initial kNN accuracy=', predicted)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(train_features, train_labels)

feature_importance_list = clf.feature_importances_
sorting_indices = np.argsort(feature_importance_list)

indices_to_remove = [sorting_indices[0]]
number_of_features = [0]
knn_accuracy = [predicted]
for indices in sorting_indices:

    new_train_features = np.delete(train_features, indices_to_remove, axis=1)
    new_test_features = np.delete(test_features, indices_to_remove, axis=1)

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(new_train_features, train_labels)

    predicted = neigh.score(new_test_features, test_labels)
    # print(f'kNN accuracy={predicted}, indicies removed: {indices_to_remove}')
    number_of_features.append(len(indices_to_remove))
    knn_accuracy.append(predicted)
    indices_to_remove.append(indices)

print(number_of_features)
print(knn_accuracy)

plt.figure()
plt.xlabel("Number of features removed")
plt.ylabel("Cross validation score ")
plt.plot(number_of_features, knn_accuracy)
plt.show()