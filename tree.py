import numpy as np
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
from rule import Inequality, Rule


class Node:
    def __init__(self, idx, idxleft, idxright, idxancestor, feature, threshold, label):
        """
        Parameters
        ----------
        idx : int
            node index in the _tree scikit structure
        idxleft : int
            index of the left node
        idxright : int
            index of the right node
        idxancestor : int
            index of the ancestor
        feature : int
            idx of the feature in the dataset (latent dimension in our case)
        threshold : float
            threshold value for the tree node
        label : int
            majority class for instances passing through that node
        """
        self.idx = idx
        self.idxleft = idxleft
        self.idxright = idxright
        self.idxancestor = idxancestor
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.ancestor = None


class SklearnDecisionTreeConverter(object):
    def __init__(self, decision_tree: DecisionTreeClassifier):
        """
        Parameters
        ----------
        decision_tree : DecisionTreeClassifier object
            scikit decision tree classifier trained on latent vectors
        """
        self.n_nodes = np.array(decision_tree.tree_.node_count)
        self.children_left = np.array(decision_tree.tree_.children_left)
        self.children_right = np.array(decision_tree.tree_.children_right)
        self.features = np.array(decision_tree.tree_.feature)
        self.thresholds = np.array(decision_tree.tree_.threshold)
        labels_idxs = np.array(decision_tree.tree_.value.argmax(axis=2).ravel())
        self.labels = []
        for idx in labels_idxs:
            self.labels.append(decision_tree.classes_[idx])
        self._build()

    def _build(self):
        nodes = []
        for node_idx in range(self.n_nodes):
            if (len(np.argwhere(self.children_right == node_idx)) == 0) and (
                len(np.argwhere(self.children_left == node_idx)) == 0
            ):  # if the node isn't ever a child (ancestor)
                idxancestor = None
            else:
                if len(np.argwhere(self.children_right == node_idx)) != 0:
                    idxancestor = np.argwhere(self.children_right == node_idx).ravel()[0]
                else:
                    idxancestor = np.argwhere(self.children_left == node_idx).ravel()[0]
            new_node = Node(
                idx=node_idx,
                idxleft=self.children_left[node_idx],
                idxright=self.children_right[node_idx],
                idxancestor=idxancestor,
                feature=self.features[node_idx],
                threshold=self.thresholds[node_idx],
                label=self.labels[node_idx],
            )
            nodes.append(new_node)
        for node in nodes:
            node.left = nodes[node.idxleft] if node.idxleft != -1 else None
            node.right = nodes[node.idxright] if node.idxright != -1 else None
            node.ancestor = (
                nodes[node.idxancestor] if node.idxancestor is not None else None
            )
        self.nodes = nodes
        return self

    def _get_rule(self, root_leaf_path, as_contained=False, labels=None):
        thresholds_signs = []
        for i, node_idx in enumerate(root_leaf_path["path"][:-1]):
            node = self.nodes[node_idx]
            if node.left.idx == root_leaf_path["path"][i + 1]:
                thresholds_signs.append("<=")
            else:
                thresholds_signs.append(">")
        root_leaf_path["thresholds_signs"] = thresholds_signs
        conditions = []
        for i, node_idx in enumerate(root_leaf_path["path"][:-1]):
            condition = Inequality(
                root_leaf_path["features"][i],
                root_leaf_path["thresholds_signs"][i],
                root_leaf_path["thresholds"][i],
                as_contained=as_contained,
            )
            conditions.append(condition)
        rule = Rule(conditions, root_leaf_path["labels"][-1], labels=labels)
        return rule

    def get_factual_rule_by_idx(self, idx, as_contained=False, labels=None):
        """
        Parameters
        ----------
        labels : list
            Optional list of label names
        idx : int
            leaf index obtained via scikit .apply method
        as_contained : bool
            flag to handle containment explanations

        Returns
        -------
        rule : Rule
            Factual rule for the sample at leaf idx
        """
        return self.get_factual_rule(
            self._get_node_by_idx(idx), as_contained=as_contained, labels=labels
        )

    def get_factual_rule(self, node: Node, as_contained=False, labels=None):
        path = []
        features = []
        majority_labels = []
        thresholds = []
        while node is not None:
            path.append(node.idx)
            features.append(node.feature)
            majority_labels.append(node.label)
            thresholds.append(node.threshold)
            node = node.ancestor

        rule = self._get_rule(
            {
                "path": path[::-1],
                "features": features[::-1],
                "labels": majority_labels[::-1],
                "thresholds": thresholds[::-1],
                "thresholds_signs": None,
            },
            as_contained=as_contained,
            labels=labels,
        )
        return rule

    def print_tree(self):
        """ Print the decision tree structure """
        self._print_subtree(self.nodes[0])

    def _print_subtree(self, node: Node, level=0):
        if node is not None:
            self._print_subtree(node.left, level + 1)
            print(
                "%s -> %s %.2f %s"
                % (" " * 12 * level, node.feature, node.threshold, node.label)
            )
            self._print_subtree(node.right, level + 1)

    def _minimum_distance(self, x: Node):
        return minimum_distance(self.nodes[0], x)

    def get_counterfactual_rule(self, factual_node: Node, as_contained=False, labels=None):
        _, nearest_leaf = self._minimum_distance(factual_node)
        counterfactual = self.get_factual_rule(
            self.nodes[nearest_leaf], as_contained=as_contained, labels=labels
        )
        return counterfactual

    def _get_node_by_idx(self, idx):
        return self.nodes[idx]

    def get_counterfactual_rule_by_idx(self, factual_idx, as_contained=False, labels=None):
        return self.get_counterfactual_rule(
            self._get_node_by_idx(factual_idx), as_contained=as_contained, labels=labels
        )


# Helper methods to find the nearest leaf for counterfactual rule extraction

def find_leaf_down(root: Node, lev, min_dist, min_idx, x):
    if root is None:
        return
    if (root.left is None and root.right is None) and root.label != x.label:
        if (lev < min_dist[0]) and lev > 0:
            min_dist[0] = lev
            min_idx[0] = root.idx
        return
    find_leaf_down(root.left, lev + 1, min_dist, min_idx, x)
    find_leaf_down(root.right, lev + 1, min_dist, min_idx, x)

def find_through_parent(root: Node, x: Node, min_dist, min_idx):
    if root is None:
        return -1
    if root == x:
        return 0
    l = find_through_parent(root.left, x, min_dist, min_idx)
    if l != -1:
        find_leaf_down(root.right, l + 2, min_dist, min_idx, x)
        return l + 1
    r = find_through_parent(root.right, x, min_dist, min_idx)
    if r != -1:
        find_leaf_down(root.left, r + 2, min_dist, min_idx, x)
        return r + 1
    return -1

def minimum_distance(root: Node, x: Node):
    min_dist = [np.inf]
    min_idx = [None]
    find_leaf_down(x, 0, min_dist, min_idx, x)
    find_through_parent(root, x, min_dist, min_idx)
    return min_dist[0], min_idx[0]

