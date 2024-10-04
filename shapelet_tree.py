import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tslearn.shapelets import LearningShapelets
from joblib import dump, load
import pathlib

from lasts.base import Plotter, Evaluator, Surrogate, RuleBasedExplanation
from lasts.utils import (
    make_path,
)
from lasts.plots import (
    plot_subsequences,
    plot_shapelet_heatmap,
)

def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_classes, l, r):
    base_size = int(l * ts_sz)
    base_size = max(base_size, 1)
    r = min(r, ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        n_shapelets = int(np.log10(n_ts *
                                      (ts_sz - shp_sz + 1) *
                                      (n_classes - 1)))
        n_shapelets = max(1, n_shapelets)
        d[shp_sz] = n_shapelets
    return d

class ShapeletTree(Surrogate):
    def __init__(
        self,
        labels=None,
        random_state=None,
        tau=None,
        tau_quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        shapelet_model_kwargs={
            "l": 0.1,
            "r": 2,
            "optimizer": "sgd",
            "n_shapelets_per_size": "heuristic",
            "weight_regularizer": 0.01,
            "max_iter": 100,
        },
        decision_tree_grid_kwargs={
            "min_samples_split": [0.002, 0.01, 0.05, 0.1, 0.2],
            "min_samples_leaf": [0.001, 0.01, 0.05, 0.1, 0.2],
            "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
        },
        prune_duplicate_tree_leaves=True,
    ):
        self.labels = labels
        self.random_state = random_state
        self.tau = tau
        self.tau_quantiles = tau_quantiles
        self.shapelet_model_kwargs = shapelet_model_kwargs
        self.decision_tree_grid_kwargs = decision_tree_grid_kwargs
        self.prune_duplicate_tree_leaves = prune_duplicate_tree_leaves
        self.plotter = ShapeletTreePlotter(self)
        self.evaluator = ShapeletTreeEvaluator(self)

        self.decision_tree_ = None
        self.shapelet_model_ = None
        self.X_transformed_ = None
        self.X_thresholded_ = None

    def fit(self, X, y):
        """Fit the shapelet model and decision tree."""
        self.X_ = X
        self.y_ = y
        n_shapelets_per_size = grabocka_params_to_shapelet_size_dict(
            n_ts=X.shape[0],
            ts_sz=X.shape[1],
            n_classes=len(set(y)),
            l=self.shapelet_model_kwargs.get("l", 0.1),
            r=self.shapelet_model_kwargs.get("r", 2),
        )

        shp_clf = LearningShapelets(
            n_shapelets_per_size=n_shapelets_per_size,
            optimizer=self.shapelet_model_kwargs.get("optimizer", "sgd"),
            weight_regularizer=self.shapelet_model_kwargs.get("weight_regularizer", 0.01),
            max_iter=self.shapelet_model_kwargs.get("max_iter", 100),
            random_state=self.random_state,
            verbose=self.shapelet_model_kwargs.get("verbose", 0),
        )

        shp_clf.fit(X, y)
        self.shapelet_model_ = shp_clf
        X_transformed = shp_clf.transform(X)
        self.X_transformed_ = X_transformed

        self._train_decision_tree(X_transformed, y)

    def _train_decision_tree(self, X_transformed, y):
        if self.tau is not None:
            self.X_thresholded_ = 1 * (X_transformed < self.tau)
            clf = DecisionTreeClassifier()
            clf.fit(self.X_thresholded_, y)
            self.decision_tree_ = clf
        else:
            grids = []
            grid_scores = []
            for quantile in self.tau_quantiles:
                X_thresholded = 1 * (X_transformed < np.quantile(X_transformed, quantile))
                clf = DecisionTreeClassifier()
                param_grid = self.decision_tree_grid_kwargs
                grid = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy", n_jobs=1, verbose=0)
                grid.fit(X_thresholded, y)
                grids.append(grid)
                grid_scores.append(grid.best_score_)
            best_grid = grids[np.argmax(grid_scores)]
            self.tau = np.quantile(X_transformed, self.tau_quantiles[np.argmax(grid_scores)])
            self.X_thresholded_ = 1 * (X_transformed < self.tau)

            clf = DecisionTreeClassifier(**best_grid.best_params_)
            clf.fit(self.X_thresholded_, y)
            self.decision_tree_ = clf

    def explain(self, x):
        """Generate explanations for a given sample."""
        factual_id = self.find_leaf_id(x)
        factual_rule = self.decision_tree_queryable_.get_factual_rule_by_idx(factual_id, as_contained=True)
        counterfactual_rule = self.decision_tree_queryable_.get_counterfactual_rule_by_idx(factual_id, as_contained=True)
        explanation = RuleBasedExplanation(factual_rule=factual_rule, counterfactual_rule=counterfactual_rule)
        return explanation

    def predict(self, X):
        """Predict the class labels for the provided data."""
        X_transformed = self.shapelet_model_.transform(X)
        X_thresholded = 1 * (X_transformed < self.tau)
        y_pred = self.decision_tree_.predict(X_thresholded)
        return y_pred

    def score(self, X, y):
        """Score the model with accuracy."""
        return accuracy_score(y, self.predict(X))

    def save(self, folder, name=""):
        path = make_path(folder)
        self.shapelet_model_.to_pickle(path=path / (name + "_shapeletmodel.joblib"))
        self.shapelet_model_ = None
        dump(self, path / (name + "_shapelettree.joblib"))
        return self

    @classmethod
    def load(cls, folder, name=""):
        path = pathlib.Path(folder)
        shapelet_tree = load(path / (name + "_shapelettree.joblib"))
        shapelet_model = LearningShapelets().from_pickle(path / (name + "_shapeletmodel.joblib"))
        shapelet_tree.shapelet_model_ = shapelet_model
        return shapelet_tree


class ShapeletTreePlotter(Plotter):
    def __init__(self, shapelet_tree: ShapeletTree):
        self.shapelet_tree = shapelet_tree

    def plot_subsequences(self, **kwargs):
        plot_subsequences(
            shapelets=self.shapelet_tree.shapelet_model_.shapelets_.copy(),
            ts_length=self.shapelet_tree.X_.shape[1],
            ts_max=self.shapelet_tree.X_.max(),
            ts_min=self.shapelet_tree.X_.min(),
            **kwargs
        )

    def plot_shapelet_heatmap(self, **kwargs):
        plot_shapelet_heatmap(
            X=self.shapelet_tree.X_transformed_,
            y=self.shapelet_tree.y_,
            **kwargs
        )


class ShapeletTreeEvaluator(Evaluator):
    def __init__(self, shapelet_tree: ShapeletTree):
        self.shapelet_tree = shapelet_tree

    def evaluate(self, metric, **kwargs):
        if metric == "accuracy_score":
            return self.shapelet_tree.score(kwargs["X"], kwargs["y"])
        raise ValueError(f"Metric {metric} not supported.")


def save_shapelet_tree(shapelet_tree: ShapeletTree, folder="./"):
    path = make_path(folder)
    shapelet_tree.shapelet_model_.to_pickle(path=path / "shapelet_model.joblib")
    shapelet_tree.shapelet_model_ = None
    dump(shapelet_tree, path / "shapelet_tree.joblib")
    return


def load_shapelet_tree(folder):
    path = pathlib.Path(folder)
    shapelet_model = LearningShapelets().from_pickle(path / "shapelet_model.joblib")
    shapelet_tree = load(path / "shapelet_tree.joblib")
    shapelet_tree.shapelet_model_ = shapelet_model
    return shapelet_tree