import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class DecisionTree:
    def __init__(self, max_depth, strategy="brute_force"):
        self.max_depth = max_depth
        self.strategy = strategy
        self.tree = None

    def fit(self, x, y):
        if self.strategy == "brute_force":
            self.tree = self._fit_brute_force(x, y)
        elif self.strategy == "binary_entropy":
            self.tree = self._fit_binary_entropy(x, y)
        else:
            raise ValueError("Unknown strategy")

    def _fit_brute_force(self, X, y):
        n_features = X.shape[1]

        def all_trees(depth, current_split):
            if depth == 0 or len(current_split) == 2 * self.max_depth - 1:
                return [current_split]

            trees = []
            for feature in range(n_features):
                left = current_split + [(feature, 0)]
                right = current_split + [(feature, 1)]
                trees += all_trees(depth - 1, left)
                trees += all_trees(depth - 1, right)
            return trees

        min_error = float("inf")
        best_tree = None

        for tree_structure in all_trees(self.max_depth):
            error = self._calculate_error(tree_structure, X, y)
            if error < min_error:
                min_error = error
                best_tree = tree_structure

        return best_tree

    def _fit_binary_entropy(self, X, y):
        def entropy(labels):
            proportions = np.bincount(labels) / len(labels)
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])

        def best_split(X, y):
            best_feature = None
            best_score = float("inf")
            for feature in range(X.shape[1]):
                left = y[X[:, feature] == 0]
                right = y[X[:, feature] == 1]

                score = len(left) * entropy(left) + len(right) * entropy(right)
                if score < best_score:
                    best_score = score
                    best_feature = feature
            return best_feature

        def build_tree(depth, X, y):
            if depth == 0 or len(set(y)) == 1:
                return np.bincount(y).argmax()

            feature = best_split(X, y)
            left_indices = X[:, feature] == 0
            right_indices = X[:, feature] == 1

            left = build_tree(depth - 1, X[left_indices], y[left_indices])
            right = build_tree(depth - 1, X[right_indices], y[right_indices])

            return {"feature": feature, "left": left, "right": right}

        return build_tree(self.max_depth, X, y)

    def _calculate_error(self, tree_structure, X, y):
        def predict_single(tree, x):
            for feature, value in tree:
                if x[feature] != value:
                    return 0
            return 1

        predictions = np.array([predict_single(tree_structure, x) for x in X])
        return np.mean(predictions != y)

    def predict(self, X):
        def predict_single(tree, x):
            for feature, value in tree:
                if isinstance(tree, dict):
                    if x[tree["feature"]] == 0:
                        tree = tree["left"]
                    else:
                        tree = tree["right"]
                else:
                    return tree

        return np.array([predict_single(self.tree, x) for x in X])

    def draw_tree(self):
        def plot_node(node, depth, x, dx, ax):
            if isinstance(node, dict):
                feature = node["feature"]
                ax.text(x, -depth, f"Feature {feature}", ha="center", bbox=dict(facecolor='white', edgecolor='black'))
                left_x = x - dx
                right_x = x + dx
                ax.plot([x, left_x], [-depth, -(depth + 1)], "k-")
                ax.plot([x, right_x], [-depth, -(depth + 1)], "k-")
                plot_node(node["left"], depth + 1, left_x, dx / 2, ax)
                plot_node(node["right"], depth + 1, right_x, dx / 2, ax)
            else:
                ax.text(x, -depth, f"Label {node}", ha="center", bbox=dict(facecolor='lightgray', edgecolor='black'))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-2 ** self.max_depth, 2 ** self.max_depth)
        ax.set_ylim(-self.max_depth - 1, 1)
        ax.axis("off")
        plot_node(self.tree, 0, 0, 2 ** (self.max_depth - 1), ax)
        plt.show()


# Sample usage
X = np.random.randint(0, 2, (100, 5))  # Example data
y = np.random.randint(0, 2, 100)  # Example labels

# Brute-force strategy
tree_brute = DecisionTree(max_depth=3, strategy="brute_force")
tree_brute.fit(X, y)
tree_brute.draw_tree()

# Binary entropy strategy
tree_entropy = DecisionTree(max_depth=3, strategy="binary_entropy")
tree_entropy.fit(X, y)
tree_entropy.draw_tree()
