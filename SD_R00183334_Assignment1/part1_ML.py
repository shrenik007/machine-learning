#  importing required libraries
import numpy as np

def calculateDistances(numpy_training_array, test_array):
    """
    This function calculates euclidean distance between test instance and training instances.
    :param numpy_training_array: numpy array of training dataset
    :param test_array: numpy array of one of the instance from test dataset
    :return: distances_np_array: numpy array of distance between all training instances with the test instance.
    :return: indices_np_array: NumPy array containing the indices that would sort the distances array.
    """
    result1 = (test_array - numpy_training_array[:]) ** 2  # (q1 - p1)^2, (q2 - p2)^2, ..., (qn - pn)^2
    distances_np_array = np.sqrt(np.sum(result1[:, :], axis=1))  # sum and square root of above result
    indices_np_array = np.argsort(distances_np_array)  # indices that would sort the distances array
    return distances_np_array, indices_np_array


def calculateDistancesManhattan(numpy_training_array, test_array):
    """
    This function calculates manhattan distance between test instance and training instances.
    :param numpy_training_array: numpy array of training dataset
    :param test_array: numpy array of one of the instance from test dataset
    :return: distances_np_array: numpy array of distance between all training instances with the test instance.
    :return: indices_np_array: NumPy array containing the indices that would sort the distances array.
    """
    result1 = np.abs(test_array - numpy_training_array[:])  # (q1 - p1), (q2 - p2), ..., (qn - pn)
    distances_np_array = np.sum(result1[:, :], axis=1)  # sum of above result
    indices_np_array = np.argsort(distances_np_array)  # indices that would sort the distances array
    return distances_np_array, indices_np_array


def calculateDistancesMinkowski(numpy_training_array, test_array):
    """
    This function calculates minkowski distance between test instance and training instances.
    :param numpy_training_array: numpy array of training dataset
    :param test_array: numpy array of one of the instance from test dataset
    :return: distances_np_array: numpy array of distance between all training instances with the test instance.
    :return: indices_np_array: NumPy array containing the indices that would sort the distances array.
    """
    a = 2
    result1 = np.abs(test_array - numpy_training_array[:]) ** a  # (q1 - p1)^a, (q2 - p2)^a, ..., (qn - pn)^a
    distances_np_array = np.power(np.sum(result1[:, :], axis=1), a)  # sum and a root of above result
    indices_np_array = np.argsort(distances_np_array)  # indices that would sort the distances array
    return distances_np_array, indices_np_array


def main(k_value, training_data_csv, test_data_csv, distance_type='euclidean'):
    """
    This is the main driving function which takes in all the training and test data and calculates accuracy based on the
    value of k provided.
    :param k_value: list of different k values. Ranges from 1 to n where n>0
    :param training_data_csv: numpy array of training dataset
    :param test_data_csv: numpy array of one of the instance from test dataset
    :param distance_type: type of distance metric
    :return: None
    """
    # iterate on each value in k_values to get different accuracies for different 'k'
    for k in k_value:
        accuracy_boolean_list = []
        # iterating on each instance of test dataset
        for row in test_data_csv:
            # Deciding which type of distance metric function to be called
            if distance_type == 'manhattan':
                distance_np_array, indices_np_array = calculateDistancesManhattan(training_data_csv[:, 0:10], row[:10])
            elif distance_type == 'minkowski':
                distance_np_array, indices_np_array = calculateDistancesMinkowski(training_data_csv[:, 0:10], row[:10])
            else:
                distance_np_array, indices_np_array = calculateDistances(training_data_csv[:, 0:10], row[:10])

            min_distance_row_index = indices_np_array[0:k]  # fetch minimum distance entries from indices array.
            #  fetching those rows from training data for indices fetched in the above step
            required_row = training_data_csv[np.array(min_distance_row_index)]
            #  forming group classes if k > 1 to get max occurred class
            class_groups = np.bincount(required_row[:, -1].astype(int))
            #  obtaining maximum value class from the above result
            max_element = np.where(class_groups == np.max(class_groups))[0][0]
            accuracy_boolean_list.append(int(row[-1]) == max_element)

        #  calculating accuracy from maximum True values occurring in the list.
        accuracy = accuracy_boolean_list.count(True) / len(accuracy_boolean_list) * 100
        print('-----------PART 1-----------')
        print(f'k:{k}, Distance metric: {distance_type.capitalize()}, Accuracy: {accuracy}%')


if __name__ == '__main__':
    #  fetching data from csv file and storing into the numpy array.
    training_data_csv = np.genfromtxt('data/classification/trainingData.csv', delimiter=",")
    test_data_csv = np.genfromtxt('data/classification/testData.csv', delimiter=",")

    #  uncomment below if you want value of k ranging from 1 to 100 at intervals of 10
    # k_value = [x for x in range(1, 101) if (x % 10 == 0 or x == 1)]

    #  uncomment below if you want value of k ranging from 1 to 10
    # k_value = []
    # k_range = 10
    # k_value.extend(range(1, k_range+1))

    #  single value of k. By default it is only 1 in the list
    k_value = [1]
    n = 2
    main(k_value, training_data_csv, test_data_csv, 'euclidean')
