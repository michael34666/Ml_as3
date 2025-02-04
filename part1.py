# k-NN Average Errors:
# Average Empirical Errors:
# k = 1, p = 1: Empirical Error = 0.0134
# k = 1, p = 2: Empirical Error = 0.0134
# k = 1, p = inf: Empirical Error = 0.0134
# k = 3, p = 1: Empirical Error = 0.0568
# k = 3, p = 2: Empirical Error = 0.0544
# k = 3, p = inf: Empirical Error = 0.0554
# k = 5, p = 1: Empirical Error = 0.0626
# k = 5, p = 2: Empirical Error = 0.0628
# k = 5, p = inf: Empirical Error = 0.0612
# k = 7, p = 1: Empirical Error = 0.0666
# k = 7, p = 2: Empirical Error = 0.0654
# k = 7, p = inf: Empirical Error = 0.0640
# k = 9, p = 1: Empirical Error = 0.0674
# k = 9, p = 2: Empirical Error = 0.0644
# k = 9, p = inf: Empirical Error = 0.0640
#
# Average True Errors:
# k = 1, p = 1: True Error = 0.1226
# k = 1, p = 2: True Error = 0.1210
# k = 1, p = inf: True Error = 0.1326
# k = 3, p = 1: True Error = 0.0890
# k = 3, p = 2: True Error = 0.0906
# k = 3, p = inf: True Error = 0.0904
# k = 5, p = 1: True Error = 0.0850
# k = 5, p = 2: True Error = 0.0852
# k = 5, p = inf: True Error = 0.0846
# k = 7, p = 1: True Error = 0.0860
# k = 7, p = 2: True Error = 0.0832
# k = 7, p = inf: True Error = 0.0868
# k = 9, p = 1: True Error = 0.0882
# k = 9, p = 2: True Error = 0.0814
# k = 9, p = inf: True Error = 0.0852

# k = 1, p = 1: Error Difference = -0.1092
# k = 1, p = 2: Error Difference = -0.1076
# k = 1, p = inf: Error Difference = -0.1192
# k = 3, p = 1: Error Difference = -0.0322
# k = 3, p = 2: Error Difference = -0.0362
# k = 3, p = inf: Error Difference = -0.0350
# k = 5, p = 1: Error Difference = -0.0224
# k = 5, p = 2: Error Difference = -0.0224
# k = 5, p = inf: Error Difference = -0.0234
# k = 7, p = 1: Error Difference = -0.0194
# k = 7, p = 2: Error Difference = -0.0178
# k = 7, p = inf: Error Difference = -0.0228
# k = 9, p = 1: Error Difference = -0.0208
# k = 9, p = 2: Error Difference = -0.0170
# k = 9, p = inf: Error Difference = -0.0212

import numpy as np
from sklearn.model_selection import train_test_split

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
def prepare_data(file_path) -> (np.array, np.array):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure there are enough parts in the line
                if parts[4] in ['Iris-versicolor', 'Iris-virginica']:
                    data.append([float(parts[1]), float(parts[2])])  # Using the second and third features
                    labels.append(1 if parts[4] == 'Iris-virginica' else 0)
            else:
                print(f"Skipping invalid line: {line}")
    return np.array(data), np.array(labels)


# Function to calculate accuracy
def accuracy(predictions, true_labels) -> float:
    return np.sum(predictions == true_labels) / len(true_labels)


# Main function
def main():
    # Prepare data
    data, labels = prepare_data('iris.txt')

    # Check if there is any valid data
    if data.shape[0] == 0:
        raise ValueError("No valid data found after processing the input file.")

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
                error_difference = true_error-empirical_error
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
    for k in k_values:
        for p in p_values:
            print(f" p = {p} , k = {k}: Empirical Error = {average_empirical_errors[k][p]:.4f}"
                  f"  True Error = {average_true_errors[k][p]:.4f}"
                  f"  Error Difference = {average_error_differences[k][p]:.4f}")



if __name__ == "__main__":
    main()

