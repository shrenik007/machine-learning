#  importing required libraries
import numpy as np
#  the euclidean distance calculation function is imported from part 1 file
from part1_ML import calculateDistances


def get_regression_value(training_data_csv, min_distance_row_indexes, distance_np_array):
    """
    This function return regression value for each test instance based on the euclidean distances between
    training data and each test data row instance
    :param min_distance_row_indexes: lowest distance indexes as a list
    :param distance_np_array: numpy array of euclidean distances
    :return: return a single regression value f(xi) for each test instance
    """
    k_indexes = np.array(min_distance_row_indexes)  # fetching list of smallest distance k indexes
    required_rows = training_data_csv[k_indexes]  # fetching rows from training using the above indices
    required_ed_array = distance_np_array[k_indexes]  # fetching euclidean distances from array using above indices
    numerator_req = np.sum(((1 / required_ed_array) ** 2) * required_rows[:, -1])  # calculating numerator for f(x)
    sum_inversion_ed = np.sum(1 / (required_ed_array ** 2))  # calculating denominator for f(x)
    regression_fx = numerator_req / sum_inversion_ed  # calculating actual f(x) using numerator and denominator.
    return regression_fx


def main(k_value, training_data_csv, test_data_csv):
    """
    This is the main driving function which takes in all the training and test data and calculates accuracy based on the
    value of k provided.
    :param k_value: list of different k values. Ranges from 1 to n where n>0
    :param training_data_csv: numpy array of training dataset
    :param test_data_csv: numpy array of one of the instance from test dataset
    :param distance_type: type of distance metric
    :return: r_square
    """
    r_square = 0
    # iterate on each value in k_values to get different accuracies for different 'k'
    for k in k_value:
        total_sum = 0
        sum_of_residuals = 0
        y_bar = np.mean(test_data_csv[:, -1])  # calculating y_bar value
        # iterating on each instance of test dataset
        for row in test_data_csv:
            # fetching euclidean distances of training data with each test instance
            distance_np_array, indices_np_array = calculateDistances(training_data_csv[:, 0:-1], row[:-1])
            min_distance_row_index = indices_np_array[0:k]
            # fetching regression value for each test instance
            regression_value = get_regression_value(training_data_csv, min_distance_row_index, distance_np_array)
            # taking sum of residuals
            sum_of_residuals += (regression_value - row[-1]) ** 2
            # calculating total sum
            total_sum += (y_bar - row[-1]) ** 2
        # calculating rsquare value using sum of residuals and total sum
        r_square = 1 - (sum_of_residuals / total_sum)
        print('-----------PART 3-----------')
        print(f'k={k}, Accuracy={r_square}')
    return r_square


if __name__ == '__main__':
    #  make the below variable True to enable feature selection using wrapper backward search
    feature_selection = False
    accuracies = []

    #  fetching data from csv file and storing into the numpy array.
    training_data_csv = np.genfromtxt('data/regression/trainingData.csv', delimiter=",")
    test_data_csv = np.genfromtxt('data/regression/testData.csv', delimiter=",")

    #  uncomment below if you want value of k ranging from 1 to 100 at intervals of 10
    # k_value = [x for x in range(1, 101) if (x % 10 == 0 or x == 1)]

    #  uncomment below if you want value of k ranging from 1 to 10
    # k_value = []
    # k_range = 10
    # k_value.extend(range(1, k_range+1))

    #  single value of k. By default it is only 1 in the list
    k_value = [10]

    best_r_square_index = 0
    #  initial rsquare value when all the features are considered
    best_r_square = main(k_value, training_data_csv, test_data_csv)
    #  Feature selection using backward search (wrapper method)
    if feature_selection:
        for index in range(12):
            # print(index + 1)
            indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            #  each time remove one feature from the above indexes list
            indexes.remove(index)
            #  fetch only the subset of features as per the above list
            new_training_data_csv = training_data_csv[:, indexes]
            new_test_data_csv = test_data_csv[:, indexes]

            #  fetch r_square value from the function
            r_square = main(k_value, new_training_data_csv, new_test_data_csv)

            # if the new calculated value is greater than previous one, than update it as the best value
            if r_square > best_r_square:
                best_r_square = r_square
                best_r_square_index = index + 1
                # print(best_r_square, best_r_square_index, index)
        print(f'It can proved that by removing feature {best_r_square_index} (column {best_r_square_index}) '
              f'we get the best accuracy')
