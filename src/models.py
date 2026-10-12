"""Model zoo mirroring the reference paper: Logistic (none/L1/L2), SVM (poly/RBF/sigmoid), GDA (LDA/QDA), FFNN."""
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# scikit-learn >= 1.8 deprecates `penalty` (removed in 1.10) in favour of C / l1_ratio
_NEW_LR_API = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)


def logistic(penalty, **kw):
    """Same model on old and new scikit-learn: no penalty = C=inf, L1 = l1_ratio 1, L2 = l1_ratio 0."""
    if _NEW_LR_API:
        kw.update({None: {"C": np.inf}, "l1": {"l1_ratio": 1.0}, "l2": {"l1_ratio": 0.0}}[penalty])
    else:
        kw["penalty"] = penalty
    return LogisticRegression(**kw)


def build_models(seed=0):
    sc = StandardScaler
    return {
        "Logistic (baseline)": make_pipeline(sc(), logistic(None, max_iter=500)),
        "Logistic (L1)": make_pipeline(sc(), logistic("l1", solver="saga", C=1.0, max_iter=300, random_state=seed)),
        "Logistic (L2)": make_pipeline(sc(), logistic("l2", C=1.0, max_iter=500)),
        "GDA (Linear)": make_pipeline(sc(), LinearDiscriminantAnalysis()),
        "GDA (Quadratic)": make_pipeline(sc(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "SVM (Poly)": make_pipeline(sc(), SVC(kernel="poly", degree=3, class_weight="balanced")),
        "SVM (RBF)": make_pipeline(sc(), SVC(kernel="rbf", class_weight="balanced")),
        "SVM (Sigmoid)": make_pipeline(sc(), SVC(kernel="sigmoid", class_weight="balanced")),
        "Neural Network": make_pipeline(sc(), MLPClassifier(hidden_layer_sizes=(128, 256, 128), activation="relu",
                                                            solver="adam", learning_rate_init=1e-3, max_iter=40,
                                                            early_stopping=True, n_iter_no_change=5, random_state=seed)),
    }


def is_kernel_svm(name):
    return name.startswith("SVM")


def oversample_positives(X, y, factor=10, seed=0):
    """Stand-in for the paper's positive-class weight of 10 (sklearn MLP has no class_weight)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y.values == 1)[0]
    idx = np.concatenate([np.arange(len(y)), rng.choice(pos, len(pos) * (factor - 1), replace=True)])
    rng.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]


def scores(model, X):
    """Continuous score for ROC/PR: probability if available, else decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
