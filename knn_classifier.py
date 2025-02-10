import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

# Function to calculate distance


def distance(point1, point2, p) -> float:
    # By the lecture:
    if p == float('inf'):
        return np.max(np.abs(point1 - point2))
    return (np.sum(np.abs(point1 - point2) ** p)) ** (1 / p)


# k-NN algorithm  supporting p as a distance metric parameter
def k_nn(train_data, train_labels, test_data, k, p) -> np.array:
    predictions = []
    for test_point in test_data:
        # Calculate the distance from test_point to all training points using the given p
        distances = [distance(test_point, train_point, p) for train_point in train_data]

        # Get indices of the k closest training points
        k_neighbors_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbors
        k_neighbors_labels = [train_labels[i] for i in k_neighbors_indices]

        # Vote for the most common label among the k neighbors
        common_lab = np.bincount(k_neighbors_labels).argmax()
        predictions.append(common_lab)

    return np.array(predictions)


# Function to prepare data
def prepare_data(file_path) -> Tuple[np.array, np.array]:
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[4] == "Iris-versicolor":
                data.append([float(parts[1]), float(parts[2])])
                labels.append(0)
            elif parts[4] == "Iris-virginica":
                data.append([float(parts[1]), float(parts[2])])
                labels.append(1)
    return np.array(data), np.array(labels)


# Function to calculate accuracy
def accuracy(predictions, true_labels) -> float:
    return np.sum(predictions == true_labels) / len(true_labels)


# Main function
def main():
    # Prepare data
    data, labels = prepare_data('iris.txt')

    # Repeat the experiment 100 times to compute average errors
    k_values = [1, 3, 5, 7, 9]
    p_values = [1, 2, np.inf]
    n_repeats = 100

    empirical_errors = {k: {p: [] for p in p_values} for k in k_values}
    true_errors = {k: {p: [] for p in p_values} for k in k_values}
    error_differences = {k: {p: [] for p in p_values} for k in k_values}

    for i in range(n_repeats):
        # Split data into training and testing sets (50% for each)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, shuffle=True)

        for k in k_values:
            for p in p_values:
                # k-NN classifier
                predictions_train = k_nn(x_train, y_train, x_train, k, p)
                predictions_test = k_nn(x_train, y_train, x_test, k, p)

                # Calculate errors
                empirical_error = 1 - accuracy(predictions_train, y_train)
                true_error = 1 - accuracy(predictions_test, y_test)

                # Store errors
                empirical_errors[k][p].append(empirical_error)
                true_errors[k][p].append(true_error)

                # Calculate and store the difference between empirical and true errors
                error_difference = true_error - empirical_error
                error_differences[k][p].append(error_difference)

    # Compute averages and differences
    average_empirical_errors = {
        k: {p: np.mean(empirical_errors[k][p]) for p in p_values} for k in k_values
    }
    average_true_errors = {
        k: {p: np.mean(true_errors[k][p]) for p in p_values} for k in k_values
    }
    average_error_differences = {
        k: {p: np.mean(error_differences[k][p]) for p in p_values} for k in k_values
    }

    # Output the results
    print(f"{'round':^15} {'Average Empirical Errors':^25} {'Average True Errors':^15} {'Error Differences':^15}")
    for p in p_values:
        for k in k_values:
            print(f"{'p = ' + str(p):<7} {'k = ' + str(k):<7} {average_empirical_errors[k][p]:^25.6f} {average_true_errors[k][p]:<15.6f} {average_error_differences[k][p]:<15.6f}")
        print()


if __name__ == "__main__":
    main()
