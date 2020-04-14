#  importing required libraries
import numpy as np

#  all the distance calculation functions are imported from part 1 file
from part1_ML import calculateDistancesManhattan, calculateDistancesMinkowski, calculateDistances


def get_distance_knn(min_distance_row_indexes, distance_np_array, training_data_csv, unique_class, n):
    """
    This function will calculate a distance-knn based on the selected training rows using value of k.
    :param min_distance_row_indexes: lowest distance indexes as a list
    :param distance_np_array: numpy array of euclidean distances
    :param training_data_csv: numpy array of training distances
    :param unique_class:  different classes in the datasets
    :param n: power value for distance knn formula
    :return: max occurrence of a class
    """
    #  fetching those rows from training data for indices fetched in the above step
    required_rows = training_data_csv[np.array(min_distance_row_indexes)]
    #  fetching those values from euclidean distance array for lowest k indices
    required_ed_array = distance_np_array[np.array(min_distance_row_indexes)]
    #  normalising the k euclidean distances
    normalised_distance = 1 / (required_ed_array ** n)
    new_list = []
    for row_class in unique_class:
        #  calculating final knn value for each class
        final_output = np.sum(normalised_distance * np.where(required_rows[:, -1] == row_class, 1, 0))
        new_list.append(final_output)
    #  return max value's index as the predicted class from this function
    return unique_class[new_list.index(max(new_list))]


def main(k_value, n, training_data_csv, test_data_csv, unique_class, distance_type='euclidean'):
    """
    This is the main driving function which takes in all the training and test data and calculates accuracy based on the
    value of k provided.
    :param k_value: list of different k values. Ranges from 1 to n where n>0
    :param n: power value for distance knn formula
    :param training_data_csv: numpy array of training dataset
    :param test_data_csv: numpy array of one of the instance from test dataset
    :param unique_class: different classes in the datasets
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
            #  calculating distance knn using the euclidean distances calculated in the above step
            max_element = get_distance_knn(min_distance_row_index, distance_np_array, training_data_csv, unique_class, n)
            accuracy_boolean_list.append(int(row[-1]) == max_element)
        accuracy = accuracy_boolean_list.count(True) / len(accuracy_boolean_list) * 100
        print('-----------PART 2-----------')
        print(f'k={k}, Distance metric={distance_type.capitalize()}, Accuracy={round(accuracy, 3)}% ')


if __name__ == '__main__':
    #  fetching data from csv file and storing into the numpy array.
    training_data_csv = np.genfromtxt('data/classification/trainingData.csv', delimiter=",")
    test_data_csv = np.genfromtxt('data/classification/testData.csv', delimiter=",")
    #  unique class list which will be used if k > 1
    unique_class = [0, 1, 2]
    #  uncomment below if you want value of k ranging from 1 to 100 at intervals of 10
    # k_value = [x for x in range(1, 101) if (x % 10 == 0 or x == 1)]

    #  uncomment below if you want value of k ranging from 1 to 10
    # k_value = []
    # k_range = 10
    # k_value.extend(range(1, k_range+1))

    #  single value of k. By default it is only 1 in the list
    k_value = [10]
    n = 2
    main(k_value, n, training_data_csv, test_data_csv, unique_class, 'euclidean')
