from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class DecisionNode:
    feature_index: Optional[int] = None  # Index of the feature to split on.
    split_value: Optional[float] = None  # Threshold value for the feature.
    prediction: Optional[int] = None
    left_child: Optional["DecisionNode"] = None
    right_child: Optional["DecisionNode"] = None
    depth: int = 0

    def classify(self, sample: np.ndarray) -> int:
        # If this is a leaf node, return its prediction.
        if self.prediction is not None:
            return self.prediction
        # Otherwise, traverse the tree.
        if sample[self.feature_index] <= self.split_value:
            return self.left_child.classify(sample)
        else:
            return self.right_child.classify(sample)


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 2,
        strategy: Literal["brute_force", "entropy"] = "entropy",
    ):
        self.max_depth = max_depth
        self.strategy = strategy
        self.root: Optional[DecisionNode] = None

    def _calc_entropy(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        # Calculate the proportion of 1s.
        p = np.mean(labels)
        # If all samples are the same, the entropy is zero.
        if p == 0 or p == 1:
            return 0.0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def show_tree(self) -> None:
        """
        Displays the structure of the tree using matplotlib.
        *This is AI generated code, so we can draw the tree using GUI, and not just print it.*
        """

        def _draw_node(
            node: DecisionNode, level: int, x_pos: float, x_offset: float, ax: plt.Axes
        ):
            if node.prediction is not None:
                ax.text(
                    x_pos,
                    -level,
                    f"Predict: {node.prediction}",
                    ha="center",
                    bbox=dict(facecolor="lightblue", edgecolor="black"),
                )
            else:
                label_text = f"X[{node.feature_index}] <= {node.split_value:.2f}"
                ax.text(
                    x_pos,
                    -level,
                    label_text,
                    ha="center",
                    bbox=dict(facecolor="white", edgecolor="black"),
                )
                # Calculate positions for children nodes.
                left_x = x_pos - x_offset
                right_x = x_pos + x_offset
                ax.plot([x_pos, left_x], [-level, -(level + 1)], "k-")
                ax.plot([x_pos, right_x], [-level, -(level + 1)], "k-")
                _draw_node(node.left_child, level + 1, left_x, x_offset / 2, ax)
                _draw_node(node.right_child, level + 1, right_x, x_offset / 2, ax)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-(2**self.max_depth), 2**self.max_depth)
        ax.set_ylim(-self.max_depth - 1, 1)
        ax.axis("off")
        if self.root:
            _draw_node(
                self.root, level=0, x_pos=0, x_offset=2 ** (self.max_depth - 1), ax=ax
            )
        plt.show()

    def _split_entropy(
        self, left_labels: np.ndarray, right_labels: np.ndarray
    ) -> float:
        """
        calculate the entropy of the split - we need to calculate the weighted average of the entropy of the left and right splits.
        """
        total = left_labels.size + right_labels.size
        weight_left = left_labels.size / total
        weight_right = right_labels.size / total
        return weight_left * self._calc_entropy(
            left_labels
        ) + weight_right * self._calc_entropy(right_labels)

    def _candidate_splits(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """
        Generate candidate splits for each feature, we need to find the best split for each feature, we need to consider all possible thresholds.
        """
        splits = []
        num_features = data.shape[1]
        for idx in range(num_features):
            sorted_vals = sorted(data[:, idx])
            # Use midpoints between consecutive values as candidate thresholds.
            thresholds = [
                (a + b) / 2 for a, b in zip(sorted_vals[:-1], sorted_vals[1:])
            ]
            splits.extend([(idx, thr) for thr in thresholds])
        return splits

    def _optimal_split(
        self, data: np.ndarray, labels: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        best_entropy = float("inf")
        best_feature, best_threshold = None, None

        for feat_idx, thr in self._candidate_splits(data):
            left_labels = []
            right_labels = []

            # Iterate over each sample to partition labels based on the threshold.
            for i in range(data.shape[0]):
                if data[i, feat_idx] <= thr:
                    left_labels.append(labels[i])
                else:
                    right_labels.append(labels[i])

            # Ensure that both partitions have at least one sample.
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            # Convert lists to numpy arrays for entropy computation.
            left_labels = np.array(left_labels)
            right_labels = np.array(right_labels)

            curr_entropy = self._split_entropy(left_labels, right_labels)
            if curr_entropy < best_entropy:
                best_entropy = curr_entropy
                best_feature = feat_idx
                best_threshold = thr

        return best_feature, best_threshold


    def enumerate_all_trees(
        self, data: np.ndarray, labels: np.ndarray, curr_depth: int = 1
    ) -> List[DecisionNode]:
        # Base case: if we've reached max depth or all labels are identical, create a leaf node.
        if curr_depth >= self.max_depth or len(set(labels)) == 1:
            pred = 1 if np.mean(labels) >= 0.5 else 0
            return [DecisionNode(prediction=pred, depth=curr_depth)]

        all_trees = []
        # Also consider making the current node a leaf.
        leaf_prediction = 1 if np.mean(labels) >= 0.5 else 0
        all_trees.append(DecisionNode(prediction=leaf_prediction, depth=curr_depth))

        # Try every candidate split for the current data.
        for feat_idx, thr in self._candidate_splits(data):
            left_data_list, left_labels_list = [], []
            right_data_list, right_labels_list = [], []

            # Partition the data and labels manually.
            for i in range(data.shape[0]):
                if data[i, feat_idx] <= thr:
                    left_data_list.append(data[i])
                    left_labels_list.append(labels[i])
                else:
                    right_data_list.append(data[i])
                    right_labels_list.append(labels[i])

            # Skip splits that do not actually partition the data.
            if len(left_data_list) == 0 or len(right_data_list) == 0:
                continue

            # Convert the lists back to numpy arrays.
            left_data = np.array(left_data_list)
            left_labels = np.array(left_labels_list)
            right_data = np.array(right_data_list)
            right_labels = np.array(right_labels_list)

            # Recursively generate subtrees for the left and right partitions.
            left_subtrees = self.enumerate_all_trees(left_data, left_labels, curr_depth + 1)
            right_subtrees = self.enumerate_all_trees(right_data, right_labels, curr_depth + 1)

            # Combine each pair of left and right subtrees into a new candidate tree.
            for left_tree in left_subtrees:
                for right_tree in right_subtrees:
                    # Skip splits where both children are leaves with the same prediction.
                    if (left_tree.prediction is not None and
                        right_tree.prediction is not None and
                        left_tree.prediction == right_tree.prediction):
                        continue

                    new_node = DecisionNode(
                        feature_index=feat_idx,
                        split_value=thr,
                        left_child=left_tree,
                        right_child=right_tree,
                        depth=curr_depth,
                    )
                    all_trees.append(new_node)

        return all_trees

    def _build_tree_entropy(
    self, data: np.ndarray, labels: np.ndarray, curr_depth: int = 1
    ) -> DecisionNode:
        # Base case: if maximum depth is reached or all labels are identical, return a leaf node.
        if curr_depth >= self.max_depth or len(set(labels)) == 1:
            final_pred = 1 if np.mean(labels) >= 0.5 else 0
            return DecisionNode(prediction=final_pred, depth=curr_depth)

        best_feat, best_thr = self._optimal_split(data, labels)
        if best_feat is None:
            final_pred = 1 if np.mean(labels) >= 0.5 else 0
            return DecisionNode(prediction=final_pred, depth=curr_depth)

        # Manually partition the data and labels without using boolean masks.
        left_data_list, left_labels_list = [], []
        right_data_list, right_labels_list = [], []

        for i in range(data.shape[0]):
            if data[i, best_feat] <= best_thr:
                left_data_list.append(data[i])
                left_labels_list.append(labels[i])
            else:
                right_data_list.append(data[i])
                right_labels_list.append(labels[i])

        # If either partition is empty, return a leaf node.
        if len(left_data_list) == 0 or len(right_data_list) == 0:
            final_pred = 1 if np.mean(labels) >= 0.5 else 0
            return DecisionNode(prediction=final_pred, depth=curr_depth)

        # Convert the lists back into numpy arrays.
        left_data = np.array(left_data_list)
        left_labels = np.array(left_labels_list)
        right_data = np.array(right_data_list)
        right_labels = np.array(right_labels_list)

        # Recursively build the tree for each branch.
        left_branch = self._build_tree_entropy(left_data, left_labels, curr_depth + 1)
        right_branch = self._build_tree_entropy(right_data, right_labels, curr_depth + 1)

        # If both branches yield the same prediction, merge them by returning a leaf node.
        if (left_branch.prediction is not None and
            right_branch.prediction is not None and
            left_branch.prediction == right_branch.prediction):
            return DecisionNode(prediction=left_branch.prediction, depth=curr_depth)

        # Otherwise, return a decision node with the best split.
        return DecisionNode(
            feature_index=best_feat,
            split_value=best_thr,
            left_child=left_branch,
            right_child=right_branch,
            depth=curr_depth,
        )


    def train(self, data: np.ndarray, labels: np.ndarray) -> Tuple[DecisionNode, float]:
        if self.strategy == "brute_force":
            candidate_trees = self.enumerate_all_trees(data, labels)
            best_error = float("inf")
            best_tree = None

            for tree in candidate_trees:
                predictions = np.array([tree.classify(sample) for sample in data])
                curr_error = np.mean(predictions != labels)
                if curr_error < best_error:
                    best_error = curr_error
                    best_tree = tree

            self.root = best_tree
            return best_tree, best_error
        else:  # Use entropy-based splitting.
            self.root = self._build_tree_entropy(data, labels)
            predictions = np.array([self.root.classify(sample) for sample in data])
            error = np.mean(predictions != labels)
            return self.root, error

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The tree must be trained before prediction!")
        return np.array([self.root.classify(sample) for sample in data])


def prepare_data(file_path) -> Tuple[np.array, np.array]:
    data = []
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts[4] == "Iris-versicolor":
                data.append([float(parts[1]), float(parts[2])])
                labels.append(0)
            elif parts[4] == "Iris-virginica":
                data.append([float(parts[1]), float(parts[2])])
                labels.append(1)
    return np.array(data), np.array(labels)


def main():
    file_path = "iris.txt"
    X, y = prepare_data(file_path)

    print("Training decision tree with brute force strategy")
    tree_brute = DecisionTree(max_depth=3, strategy="brute_force")
    tree_brute.train(X, y)

    print("Drawing brute force decision tree")
    tree_brute.show_tree()

    print("Training decision tree with binary entropy strategy")
    tree_entropy = DecisionTree(max_depth=3, strategy="binary_entropy")
    tree_entropy.train(X, y)

    print("Drawing entropy-based decision tree")
    tree_entropy.show_tree()

    y_pred_brute = tree_brute.predict(X)
    y_pred_entropy = tree_entropy.predict(X)

    accuracy_brute = np.mean(y_pred_brute == y)
    accuracy_entropy = np.mean(y_pred_entropy == y)

    print(f"\nAccuracy with brute force strategy: {accuracy_brute:.4f}")
    print(f"Accuracy with binary entropy strategy: {accuracy_entropy:.4f}")


if __name__ == "__main__":
    main()
