# define config values for the different metrics

from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference
# import cvm.utils_cvm as cvm

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm



def evaluate_metrics(metric_registry, selected_metrics=None, **kwargs):
    """
    Compute selected metrics using dynamically supplied inputs.

    Parameters:
        metric_registry (dict): Dictionary with metric configurations. Each entry should map
                                metric name to a dict with 'function' and 'requires' keys.
        selected_metrics (list of str, optional): Names of metrics to compute. If None, compute all.
        **kwargs: Inputs required by the metric functions (e.g., y_true, y_pred, sensitive_features).

    Returns:
        dict: Metric name → result

    Raises:
        KeyError: If a required input is missing for any selected metric.
        ValueError: If a metric name is unknown.
        Exception: For any errors raised during metric computation.
    """
    results = {}
    to_run = selected_metrics or list(metric_registry.keys())

    for name in to_run:
        if name not in metric_registry:
            raise ValueError(f"Unknown metric: '{name}'")

        config = metric_registry[name]
        func = config["function"]
        required_args = config["requires"]
        optional_args = config["optional"]

        try:
            required_inputs = [kwargs[arg] for arg in required_args]
            optional_inputs = {arg:kwargs[arg] for arg in optional_args}
        except KeyError as e:
            raise KeyError(f"Missing required input for metric '{name}': {e}")

        try:
            results[name] = func(*required_inputs, **optional_inputs)
        except Exception as e:
            raise RuntimeError(f"Error computing metric '{name}': {e}")

    return results

def evaluate_multiple_models(models, metric_registry, selected_metrics=None, **kwargs):
    """
    Evaluate multiple models and compare them on specified metrics.

    Parameters:
        models (dict): Dict of model names → (y_pred, optional y_prob)
        metric_registry (dict): Metric definitions for evaluate_metrics.
        selected_metrics (list of str, optional): Subset of metrics to compute.
        **kwargs: Shared inputs like y_true, sensitive_features, etc.

    Returns:
        pd.DataFrame: Metrics per model
    """
    from collections import defaultdict

    results = defaultdict(dict)

    for model_name, model_outputs in models.items():
        model_kwargs = kwargs.copy()

        if isinstance(model_outputs, tuple):
            # assume (y_pred, y_prob) tuple
            model_kwargs['y_pred_binary'], model_kwargs['y_prob'] = model_outputs
        else:
            model_kwargs['y_pred_binary'] = model_outputs

        metric_results = evaluate_metrics(metric_registry, selected_metrics, **model_kwargs)

        for metric_name, value in metric_results.items():
            results[model_name][metric_name] = value

    return pd.DataFrame(results).T  # Models as rows, metrics as columns

def plot_model_comparison(df, inclinated_names=False,filename = None):
    """
    Plot a grouped bar chart comparing models across metrics.

    Parameters:
        df (pd.DataFrame): Output from `evaluate_multiple_models`
    """

    labels = [
        "Accuracy",
        "F1 score",
        "DP",
        "EoO",
        "CvM",
        "MI",
        "Differentiable CvM",
    ]

    # increase font size
    plt.rcParams.update({'font.size': 13})
    ax = df.plot(kind="bar", figsize=(10, 6+2*int(inclinated_names)))
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Models")
    # ax.set_xlabel("Fine-tuning Epoch x20")
    # ax.set_title(title)
    ax.set_xticklabels(df.index, rotation=0+65*int(inclinated_names))
    ax.legend(title="Metrics", ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.20),handlelength=1.5,handletextpad=0.4,columnspacing=0.3)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    if filename is not None:
        plt.savefig("plots."+filename+".pdf", bbox_inches="tight")
    plt.show()

def plot_model_comparison_errors(df_mean: pd.DataFrame,
                          df_std = None,
                          inclinated_names: bool = False,
                          filename = None,
                          show_grid: bool = True,
                          capsize: float = 4.0):
    """
    Plot a grouped bar chart comparing models across metrics, with optional std error bars.

    Parameters:
        df_mean (pd.DataFrame): Mean metric values (models as index, metrics as columns).
        df_std  (pd.DataFrame|None): Std dev for the same metrics (same shape/index/columns as df_mean).
        inclinated_names (bool): Rotate x tick labels by ~65 degrees if True.
        filename (str|None): If provided, saves a PDF to 'plots.{filename}.pdf'.
        show_grid (bool): Whether to show a dashed grid.
        capsize (float): Size of the error bar caps.
    """

    # Optional: keep a specific metric order if these columns exist
    preferred_order = [
        "Accuracy", "F1 score", "DP", "EoO", "CvM", "MI", "Differentiable CvM",
    ]
    cols_in_both = [c for c in preferred_order if c in df_mean.columns]
    other_cols = [c for c in df_mean.columns if c not in cols_in_both]
    ordered_cols = cols_in_both + other_cols
    df_mean = df_mean[ordered_cols]

    # If std is provided, align to the same index/columns and validate
    yerr = None
    if df_std is not None:
        # Reindex to match and ensure there are no negative stds
        df_std = df_std.reindex(index=df_mean.index, columns=df_mean.columns)
        if not df_std.index.equals(df_mean.index) or not df_std.columns.equals(df_mean.columns):
            raise ValueError("df_std must have the same index and columns as df_mean.")
        if (df_std < 0).any().any():
            raise ValueError("df_std contains negative values, which is not valid for std.")
        yerr = df_std

    # Increase font size
    plt.rcParams.update({"font.size": 13})

    # Draw bars (pandas forwards yerr/error_kw to matplotlib)
    ax = df_mean.plot(
        kind="bar",
        figsize=(10, 6 + 2 * int(inclinated_names)),
        yerr=yerr,
        error_kw={"elinewidth": 1.2, "capsize": capsize, "capthick": 1.2}
    )

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Models")
    ax.set_xlabel("λ value")  # kept from your original code
    ax.set_xticklabels(df_mean.index, rotation=0 + 65 * int(inclinated_names))

    # Legend: one entry per metric column
    ax.legend(
        title="Metrics",
        ncol=len(df_mean.columns),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.3,
    )

    if show_grid:
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")  # grid on y usually looks cleaner

    plt.tight_layout()

    if filename is not None:
        plt.savefig(f"plots.{filename}.pdf", bbox_inches="tight")

    plt.show()

# wrapper for cvm.compute_cvm_chatterjee_autodiff function
def get_differentiable_cvm(X,Y,regularization_strength=1e-5,modify_ranks=True):
    return cvm.compute_cvm_chatterjee_autodiff(X,Y,regularization_strength=regularization_strength,modify_ranks=modify_ranks).item()

# wrapper for cvm.compute_cvm_classic function
def get_classic_cvm(X,Y):
    return cvm.compute_cvm_classic(X,Y).item()

# wrapper for sklearn's mutual_info_regression
def get_mutual_info_regression(X, y_prob):
    return mutual_info_regression(X.reshape(-1,1), y_prob)[0]

# function to plot the losses
import matplotlib.pyplot as plt
def plot_losses(val_total_loss_list, val_reg_list,model_name=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(val_total_loss_list, label='Total Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('(V) Total Loss over Epochs for ' + model_name if model_name else '(V) Total Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_reg_list, label='Validation Regularization Term', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Regularization Term')
    plt.title('(V) Regularization Term over Epochs for ' + model_name if model_name else  '(V) Regularization Term over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# get the models' accuracies
def compute_accuracies(prediction_dict, labels):
    for model_name, (preds, probs) in prediction_dict.items():
        accuracy = (preds == labels.numpy().squeeze()).mean()
        acceptance_rate = probs.mean().item()
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Acceptance Rate: {acceptance_rate:.4f}, imporvement over random: {accuracy-dataset.labels.numpy().squeeze().mean():.4f}")
    return
    
# function to evaluate the model predicitons [works properly]
def evaluate_model(model, test_inputs, test_labels):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(test_inputs)
        y_pred = torch.sigmoid(y_pred_tensor).numpy().flatten()
        y_pred_binary = (y_pred > 0.5).astype(int)

    acc = accuracy_score(test_labels, y_pred_binary)
    print("Accuracy:", acc)
    acceptance_rate = y_pred_binary.mean()
    print(f"Positive prediction rate: {acceptance_rate:.4f}")
    return acc
    
# average the results for each multiplier in a dataframe containing the results for different multipliers and repetitions
def average_results(df_results):
    """Averages the results for each multiplier in the index of df_results.
    Important: the indexes of df_results must be a string in the format "model_name multiplier repetition"
    """
    df_results_avg = df_results.copy()
    df_results_avg['to_group'] = np.array(df_results.index.str.extract(r'(\d+)')[0], dtype=int)
    df_results_avg = df_results_avg.groupby('to_group').mean()
    return df_results_avg

def std_results(df_results):
    """Computes the standard deviation of the results for each multiplier in the index of df_results.
    Important: the indexes of df_results must be a string in the format "model_name multiplier repetition"
    """
    df_results_std = df_results.copy()
    df_results_std['to_group'] = np.array(df_results.index.str.extract(r'(\d+)')[0], dtype=int)
    df_results_std = df_results_std.groupby('to_group').std()
    return df_results_std

def load_metric_registry():
    """Load the predefined metric registry.
    The predefined metrics include:
        - Accuracy
        - F1 score
        - Demographic Parity Difference
        - Equal Opportunity Difference
        - Differentiable CVM
        - Classic CVM
        - Mutual Information
    Returns:
        dict: Metric name → configuration dict
        with the following keys and metrics:
            - "function": callable
            - "requires": list of required input names
            - "optional": list of optional input names
    """
    metric_registry = {
        "Accuracy": {
            "function": accuracy_score,
            "requires": ["y_true", "y_pred_binary"],
            "optional": []
        },
        "F1 score": {
            "function": f1_score,
            "requires": ["y_true", "y_pred_binary"],
            "optional": []
        },
        "DP": {
            "function": demographic_parity_difference,
            "requires": ["y_true", "y_pred_binary"],
            "optional": ["sensitive_features"]
        },
        "EoO": {
            "function": equal_opportunity_difference,
            "requires": ["y_true", "y_pred_binary"],
            "optional": ["sensitive_features"]
        },
        "EO": {
            "function": equalized_odds_difference,
            "requires": ["y_true", "y_pred_binary"],
            "optional": ["sensitive_features"]
        },
        "MI": {
            "function": get_mutual_info_regression,
            "requires": ["X", "y_prob"],
            "optional": []
        },
        "CvM": {
            "function": get_classic_cvm,
            "requires": ["X", "y_prob"],
            "optional": []
        },
        "Diff CvM": {
            "function": get_differentiable_cvm,
            "requires": ["X", "y_prob"],
            "optional": ["regularization_strength", "modify_ranks"]
        },
    }
    return metric_registry

def predict(model, inputs):
    """Given a model and some inputs, return a dictionary containing the predictions (binary,probabilistic) for the model on the inputs"""
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).squeeze()
        probs = torch.sigmoid(outputs).flatten()
        preds = (probs.numpy() >= 0.5).astype(int)
    return preds, probs

def preDICT(model_dict, inputs):
    """Given a model DICTIONARY and some inputs, return a dictionary containing the predictions (binary,probabilistic) for each model on the inputs"""
    prediction_dict = {}
    for model_name, model in model_dict.items():
        model.eval()
        with torch.no_grad():
            preds, probs = predict(model, inputs)
        prediction_dict[model_name] = (preds, probs)
    return prediction_dict

from sklearn.metrics import confusion_matrix
def get_confusion_matrix(model, X_test_tensor, y_test_tensor,cutoff=0.5):
    """Given a model and a test dataset, return the confusion matrix"""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = torch.sigmoid(test_outputs).numpy()
        test_preds = (test_probs >= cutoff).astype(int)
        cm = confusion_matrix(y_test_tensor.numpy(), test_preds)
    return cm

def get_confusion_matrix_from_predictions(preds, y_test_tensor):

    """Given predictions and true labels, return the confusion matrix"""
    cm = confusion_matrix(y_test_tensor.numpy(), preds)
    return cm

def plot_prob_histogram(grouping, probabilities=None, model=None, X_test_tensor=None,n_bins=20, normalize=False):
    """Plot the histogram of the predicted probabilities for each group in the grouping variable"""
    if probabilities is None:
        assert model is not None and X_test_tensor is not None, "Either probabilities or (model and X_test_tensor) must be provided"
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    else:
        test_probs = probabilities
    
    groups = np.unique(grouping)
    plt.figure(figsize=(10, 6))
    for group in groups:
        group_probs = test_probs[grouping == group]
        plt.hist(group_probs, bins=n_bins, alpha=0.5, label=f'Group {group}', density=normalize)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities by Group')
    plt.legend()
    plt.show()

def plot_accuracy_fairness_approx(
    df: pd.DataFrame,
    title: str,
    measure: str):
    """
    Plot Accuracy vs Demographic Parity from an approximate dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['Method', 'Accuracy', measure]
    title : str
        Plot title
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    # Methods that should be plotted as curves
    curve_methods = [
        "DC",
        "CDC",
        "HGR",
        "FairMixup",
        "FERMI",
        "Soft-HGR",
    ]

    # Standalone baseline points
    baseline_methods = ["Unfair", "p_1", "p_2", "p_3"]

    # Plot curves
    for method in curve_methods:
        data = df[df["Method"] == method].sort_values("Accuracy")

        ax.plot(
            data["Accuracy"],
            data[measure],
            marker="o",
            linewidth=2,
            label=method,
        )

    data = df[df["Method"] == "CvM"].sort_values("Accuracy")
    ax.plot(
        data["Accuracy"],
        data[measure],
        marker="o",
        linewidth=2,
        label="CvM",
        color='black'
    )

    # Plot baselines
    for method in baseline_methods:
        data = df[df["Method"] == method]
        ax.scatter(
            data["Accuracy"],
            data[measure],
            s=120,
            marker="^" if method == "Unfair" else "D",
            label=method,
            zorder=5,
        )

    

    # Axes and styling
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel(measure, fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Methods", frameon=True)

    plt.tight_layout()
    # plt.show()

def plot_accuracy_fairness_approx_inverse(
    df: pd.DataFrame,
    title: str,
    measure: str,
    cvm_std = None):
    """
    Plot Accuracy vs Demographic Parity from an approximate dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['Method', 'Accuracy', measure]
    title : str
        Plot title
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    # Methods that should be plotted as curves
    curve_methods = [
        "DC",
        "CDC",
        "HGR",
        "FairMixup",
        "FERMI",
        "Soft-HGR",
    ]

    # Standalone baseline points
    baseline_methods = ["Unfair"]

    # Plot curves
    for method in curve_methods:
        data = df[df["Method"] == method].sort_values("Accuracy")

        ax.plot(
            data[measure],
            data["Accuracy"],
            marker="o",
            linewidth=2,
            label=method,
        )

    data = df[df["Method"] == "CvM"].sort_values("Accuracy")
    ax.plot(
        data[measure],
        data["Accuracy"],
        marker="o",
        linewidth=2,
        label="CvM",
        color='black'
    )

    if cvm_std is not None:
        ax.fill_between(
            data[measure],
            data["Accuracy"] - cvm_std,
            data["Accuracy"] + cvm_std,
            alpha=0.25,
            color='gray'
        )
        

    # Plot baselines
    for method in baseline_methods:
        data = df[df["Method"] == method]
        ax.scatter(
            data[measure],
            data["Accuracy"],
            s=120,
            marker="^" if method == "Unfair" else "D",
            label=method,
            zorder=5,
        )

    

    # Axes and styling
    ax.set_xlabel(measure, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Methods", frameon=True)

    plt.tight_layout()
    # plt.show()