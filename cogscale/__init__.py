import cogscale.evals as evals
import numpy as np

tasks = [
    'sinus_forecasting',
    'chaotic_forecasting',
    'discrete_postcasting',
    'continuous_postcasting',
    'discrete_pattern_completion',
    'continuous_pattern_completion',
    'bracket_matching',
    'simple_copy',
    'selective_copy',
    'adding_problem',
    'sorting_problem',
    'cross_situation',
    'associative_recall',
    'induction_heads',
]

def compute_score(Y, Y_hat, category, threshold=0.5):
    """
    Compute the accuracy of the model using -100 as a mask.

    Parameters:
    - Y (np.ndarray): Target array [B, T, O] (contains -100 where masked)
    - Y_hat (np.ndarray): Predicted array [B, T, O]
    - category (str): Category of the task -> 'classification' (acc) or 'regression' (mse) or 'multi_classification'

    Returns:
    - accuracy (float): Accuracy value
    """
    # Make sure Y_hat and Y are numpy arrays
    if not isinstance(Y_hat, np.ndarray) or not isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype=np.float32)
        Y_hat = np.array(Y_hat, dtype=np.float32)

    # Create the mask: True for timesteps where the target is not completely -100
    mask = np.any(Y != -100.0, axis=-1)
    
    # Select only the non-masked prediction timesteps
    preds = Y_hat[mask]
    truths = Y[mask]

    if category=='classification':
        # Compute the accuracy
        preds_class = np.argmax(preds, axis=-1)  # [N] int: class
        truths_class = np.argmax(truths, axis=-1)  # [N] int: class
        score = np.sum(preds_class == truths_class) / truths_class.shape[0]
        score = 1 - score

    elif category=='multi_classification':
        # Compute the accuracy
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        preds_bin = (sigmoid(preds) >= threshold).astype(int)
        correct_samples = (preds_bin == truths)
        score = 1 - np.mean(correct_samples)

    elif category=='regression':
        # Compute the MSE
        score = np.mean((preds - truths) ** 2)

    else:
        raise ValueError(f"Unknown category {category}. Must be 'classification', 'multi_classification' or 'regression'.")

    return score

def build_task(task_name, difficulty='small', seed=None, **kwargs):
    """
    Build the task.

    Parameters:
    - task_name (str): Name of the task between 'sinus_forecasting', 'chaotic_forecasting', 'discrete_postcasting',
        'continuous_postcasting', 'discrete_pattern_completion', 'continuous_pattern_completion', 'bracket_matching',
        'simple_copy', 'selective_copy', 'adding_problem', 'sorting_problem', and 'cross_situation'.
    - difficulty (str): Difficulty level of the task ('small', 'medium' or 'large')
    - seed (int, optional): Seed for reproducibility. Default is None.

    The other optional parameters are given as arguments in the task generation function.

    Returns:
    - Task: Task object
    """
    # Check if the task name is valid 
    if task_name not in evals.cogscale_small:
        raise ValueError(f"Task {task_name} not found. Available tasks are: {list(evals.cogscale_small.keys())}")
    # Check if the difficulty level is valid
    if difficulty not in ['small', 'medium', 'large']:
        raise ValueError("Difficulty level must be 'small', 'medium' or 'large'.")

    # Get the corresponding cogscale configuration
    cogscale = {
        'small': evals.cogscale_small,
        'medium': evals.cogscale_medium,
        'large': evals.cogscale_large,
    }[difficulty]

    # Get the function and parameters from the cogscale config
    fct = cogscale[task_name]['fct']
    params = cogscale[task_name]['params']
    params['seed'] = seed

    # Update params with optional arguments
    params |= kwargs

    # Generate the task
    return fct(**params)