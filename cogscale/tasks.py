import numpy as np
from cogscale.libs.CSL import TwoSituationCSLDataset


# ------------ USEFUL FUNCTIONS ------------ #

def _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category):
    """
    Generate the samples and split them into training, validation and testing sets.
    
    Parameters:
    - n_train (int): Number of training samples
    - n_valid (int): Number of validation samples
    - n_test (int): Number of testing samples
    - generate_one_sample (function): Function to generate one sample
    
    Returns:
    - data (dict): Dictionary containing the training, testing and validation sets and their respective prediction timesteps.
    It also contains the category flag.
    """
    # Generate the samples
    n_samples = n_train + n_test + n_valid
    input, target, timesteps = zip(*[generate_one_sample() for _ in range(n_samples)])
    input, target, timesteps = np.array(input), np.array(target), np.array(timesteps)
    
    # Split the data into training and testing set
    X_train = input[:n_train, :, :]
    Y_train = target[:n_train, :, :]
    T_train = timesteps[:n_train, :] # timesteps to predict
    X_valid = input[n_train:n_train+n_valid, :, :]
    Y_valid = target[n_train:n_train+n_valid, :, :]
    T_valid = timesteps[n_train:n_train+n_valid, :] # timesteps to predict
    X_test = input[n_train+n_valid:, :, :]
    Y_test = target[n_train+n_valid:, :, :]
    T_test = timesteps[n_train+n_valid:, :] # timesteps to predict

    # Create the data dictionary
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'T_train': T_train,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'T_valid': T_valid,
        'X_test': X_test,
        'Y_test': Y_test,
        'T_test': T_test,
        'category': category
    }

    return data



# ------------ SIMPLE MEMORY TEST ------------ #

def generate_discrete_postcasting(n_train=1000, n_valid=200, n_test=200, sequence_length=1000, delay=10, n_symbols=8, seed=None):
    """
    [Multi sequence]
    Generates a copy task: the model must reproduce the input sequence 
    (one-hot) after a delay.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - delay (int): delay before reproducing the sequence
    - n_symbols (int): number of possible symbols
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate the sequence
        sequence = rng.integers(0, n_symbols, size=sequence_length)
        input = np.eye(n_symbols)[sequence].reshape(sequence_length, n_symbols)
        target = np.concatenate([np.zeros((delay, n_symbols)), input[:-delay, :]], axis=0)
        timesteps = np.arange(delay, sequence_length)

        return input, target, timesteps
    
    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')




def generate_continuous_postcasting(n_train=1000, n_valid=200, n_test=200, sequence_length=1000, delay=10, seed=None):
    """
    [Multi sequence]
    Generates a copy task: the model must reproduce the input sequence 
    (continuous) after a delay.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - delay (int): delay before reproducing the sequence
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate the sequence
        input = rng.uniform(-0.8, 0.8, size=sequence_length).reshape(sequence_length, 1)
        target = np.concatenate([np.zeros((delay, 1)), input[:-delay, :]], axis=0)
        timesteps = np.arange(delay, sequence_length)

        return input, target, timesteps
    
    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='regression')

# ------------ SIGNAL PROCESSING TEST ------------ #

def generate_sinus_forecasting(sequence_length=1000, forecast_length=1, training_ratio=0.8, validation_ratio=0.1, testing_ratio=0.1, seed=None):
    """
    [Single sequence]
    Generates a frequency-modulated sinusoidal signal.
    The model must predict the signal frequency at the next timestep.
    The signal is deterministic and there is no random train-test split, no seed is needed.

    Args:
    - sequence_length (int): sequence length
    - forecast_length (int): prediction length
    - training_ratio (float): proportion of the sequence used for training
    - validation_ratio (float): proportion of the sequence used for validation
    - testing_ratio (float): proportion of the sequence used for testing

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    # Check the ratios
    if training_ratio + testing_ratio + validation_ratio != 1:
        raise ValueError("The sum of the ratios must be equal to 1.")

    # Generate the signal
    length = sequence_length + forecast_length
    max_value = length / 100
    t = np.linspace(0, max_value, length)
    carrier_freq = 10
    modulator_freq = 0.5
    modulator = np.sin(2 * np.pi * modulator_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)

    # Create the input & target
    input = carrier[:-forecast_length].reshape(1, -1, 1)
    target = carrier[forecast_length:].reshape(1, -1, 1)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    validation_size = int(sequence_length * (training_ratio + validation_ratio))
    X_train = input[:, :training_size, :]
    Y_train = target[:, :training_size, :]
    X_valid = input[:, :validation_size, :]
    Y_valid = target[:, :validation_size, :]
    X_test = input
    Y_test = target

    # Prediction timestep
    T_train = np.arange(forecast_length, training_size).reshape(1, -1)
    T_valid = np.arange(training_size, validation_size).reshape(1, -1)
    T_test = np.arange(validation_size, sequence_length).reshape(1, -1)

    # Create the data dictionary
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'T_train': T_train,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'T_valid': T_valid,
        'X_test': X_test,
        'Y_test': Y_test,
        'T_test': T_test,
        'category': 'regression'
    }

    return data

def generate_chaotic_forecasting(sequence_length=1000, forecast_length=1, training_ratio=0.8, validation_ratio=0.1, testing_ratio=0.1, seed=None):
    """
    [Single sequence]
    Generates a chaotic time series (Lorenz system).
    The model must predict the system state at the next timestep.
    The signal is deterministic and there is no random train-test split, no seed is needed.

    Args:
    - sequence_length (int): sequence length
    - forecast_length (int): prediction length
    - training_ratio (float): proportion of samples used for training
    - validation_ratio (float): proportion of samples used for validation
    - testing_ratio (float): proportion of samples used for testing

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    # Check the ratios
    if training_ratio + testing_ratio + validation_ratio != 1:
        raise ValueError("The sum of the ratios must be equal to 1.")

    # Define the Lorenz system
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz
    
    # Generate the Lorenz system
    dt = 0.01
    stepCnt = sequence_length + forecast_length
    
    xs = np.zeros(stepCnt)
    ys = np.zeros(stepCnt)
    zs = np.zeros(stepCnt)
    
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    for i in range(stepCnt-1):
        dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (dx * dt)
        ys[i + 1] = ys[i] + (dy * dt)
        zs[i + 1] = zs[i] + (dz * dt)
    
    # Normalize the data
    xs = (xs - np.mean(xs)) / (3*np.std(xs))
    ys = (ys - np.mean(ys)) / (3*np.std(ys))
    zs = (zs - np.mean(zs)) / (3*np.std(zs))

    # Create the input & target
    input = np.column_stack((xs[:-forecast_length], ys[:-forecast_length], zs[:-forecast_length])).reshape(1, -1, 3)
    target = np.column_stack((xs[forecast_length:], ys[forecast_length:], zs[forecast_length:])).reshape(1, -1, 3)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    validation_size = int(sequence_length * (training_ratio + validation_ratio))
    X_train = input[:, :training_size, :]
    Y_train = target[:, :training_size, :]
    X_valid = input[:, :validation_size, :]
    Y_valid = target[:, :validation_size, :]
    X_test = input
    Y_test = target

    # Prediction timestep
    T_train = np.arange(forecast_length, training_size).reshape(1, -1)
    T_valid = np.arange(training_size, validation_size).reshape(1, -1)
    T_test = np.arange(validation_size, sequence_length).reshape(1, -1)

    # Create the data dictionary
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'T_train': T_train,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'T_valid': T_valid,
        'X_test': X_test,
        'Y_test': Y_test,
        'T_test': T_test,
        'category': 'regression'
    }

    return data

# ------------ LONG-TERM DEPENDENCY TEST ------------ #

def generate_discrete_pattern_completion(n_train=1000, n_valid=200, n_test=200, sequence_length=1000, n_symbols=8, base_length=5, mask_ratio=0.2, seed=None):
    """
    [Multi sequence]
    The model must identify and complete repetitive patterns.
    The sequence consists of repeating a pattern of length base_length and dimension n_symbols + 1.
    The first symbol is a marker indicating when the model should predict the pattern.
    The other symbols are elements of the pattern.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - n_symbols (int): number of possible symbols
    - base_length (int): pattern length
    - mask_ratio (float): proportion of symbols to mask
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate a base pattern
        base_pattern = rng.integers(0, n_symbols, size=base_length)
        sequence = np.tile(base_pattern, sequence_length // base_length + 1)[:sequence_length]

        # Mask some parts so that the model predicts them
        nb_masked = int(sequence_length * mask_ratio)
        mask = rng.choice(sequence_length, nb_masked, replace=False)
        masked_sequence = sequence.copy()
        masked_sequence[mask] = n_symbols

        # One-hot encoding
        input = np.eye(n_symbols+1)[masked_sequence]
        target = np.eye(n_symbols)[sequence]
        timesteps = mask

        return input, target, timesteps

    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')

def generate_continuous_pattern_completion(n_train=1000, n_valid=200, n_test=200, sequence_length=100, base_length=5, mask_ratio=0.2, seed=None):
    """
    [Multi sequence]
    The model must identify and complete repetitive patterns.
    The sequence consists of repeating a pattern of length base_length and dimension 1.
    Some values in the sequence are masked by the value -1 and the model must predict them.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - base_length (int): pattern length
    - mask_ratio (float): proportion of masked symbols
    - training_ratio (float): proportion of samples used for training
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate a base pattern
        base_pattern = rng.uniform(0, 1, size=base_length)
        sequence = np.tile(base_pattern, sequence_length // base_length + 1)[:sequence_length]

        # Mask some parts so that the model predicts them
        nb_masked = int(sequence_length * mask_ratio)
        mask = rng.choice(sequence_length, nb_masked, replace=False)
        masked_sequence = sequence.copy()
        masked_sequence[mask] = -1

        # One-hot encoding
        input = masked_sequence.reshape(-1, 1)
        target = sequence.reshape(-1, 1)
        timesteps = mask

        return input, target, timesteps
    
    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='regression')

def generate_simple_copy(n_train=1000, n_valid=200, n_test=200, sequence_length=100, delay=10, n_symbols=8, seed=None):
    """
    [Multi sequence]
    Generates a copy task: the model must read an entire sequence, 
    memorize it and reproduce it after a delay, when a trigger warns it.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - delay (int): delay before reproducing the sequence
    - n_symbols (int): number of possible symbols
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate a random sequence
        sequence = rng.integers(0, n_symbols, size=sequence_length)  # 8 possible symbols
        sequence_onehot = np.eye(n_symbols)[sequence]

        # Create the input & target
        input_sequence = np.zeros((sequence_length + delay + 1 + sequence_length, n_symbols + 1))
        input_sequence[:sequence_length, :n_symbols] = sequence_onehot
        input_sequence[sequence_length + delay, n_symbols] = 1  # marker

        target_sequence = np.zeros((sequence_length + delay + 1 + sequence_length, n_symbols))
        target_sequence[sequence_length + delay + 1:, :] = sequence_onehot

        timesteps = np.arange(sequence_length + delay + 1, sequence_length + delay + 1 + sequence_length)

        return input_sequence, target_sequence, timesteps

    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')

def generate_selective_copy(n_train=1000, n_valid=200, n_test=200, sequence_length=100, delay=2, n_markers=2, n_symbols=8, seed=None):
    """
    [Multi sequence]
    The model must read an entire sequence, memorize the marked elements,
    and reproduce only the marked elements in the sequence, once the trigger signal is received.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - delay (int): delay before reproducing the sequence
    - n_markers (int): number of elements to memorize < sequence_length
    - n_symbols (int): number of possible symbols
    - seed (int): random seed for reproducibility

    Return: 
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # generate random sequence
        sequence = rng.integers(0, n_symbols, size=sequence_length)
        sequence_onehot = np.eye(n_symbols)[sequence]
        selected_indices = rng.choice(sequence_length, n_markers, replace=False)
        selected_indices.sort()

        # Create the input
        input = np.zeros((sequence_length + delay + 1 + n_markers, n_symbols + 2))
        input[:sequence_length, :n_symbols] = sequence_onehot # sequence
        input[selected_indices, n_symbols] = 1 # markers
        input[sequence_length + delay, n_symbols + 1] = 1

        # Create the target
        target = np.zeros((sequence_length + delay + 1 + n_markers, n_symbols))
        target[-n_markers:, :] = sequence_onehot[selected_indices, :]

        # Create the timesteps
        timesteps = np.arange(sequence_length + delay + 1, sequence_length + delay + 1 + n_markers)

        return input, target, timesteps
    
    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')

# ------------ TEST FOR MANIPULATION OF RETAINED INFORMATION ------------ #

def generate_adding_problem(n_train=1000, n_valid=200, n_test=200, sequence_length=100, max_number=9, seed=None):
    """
    [Multi sequence]
    The model must read a sequence of random numbers, 
    then add the numbers at the marked positions once it receives the trigger signal.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - max_number (int): maximum possible number
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Generate the sequence
        sequence = rng.integers(0, max_number, sequence_length)
        selected_indices = rng.choice(sequence_length, 2, replace=False)
        result = (sequence[selected_indices] + 1).sum()

        # Create input
        input = np.zeros((sequence_length+2, max_number+2))
        input[:sequence_length, :max_number] = np.eye(max_number)[sequence] # One-hot encoding
        input[selected_indices, max_number] = 1 # Markers
        input[sequence_length, max_number+1] = 1 # Trigger

        # Create target
        target = np.zeros((sequence_length+2, max_number*2-1))
        target[-1, result-2] = 1

        # Create timesteps
        timesteps = np.arange(sequence_length+1, sequence_length+2)

        return input, target, timesteps

    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')

def generate_sorting_problem(n_train=1000, n_valid=200, n_test=200, sequence_length=100, n_symbols=8, seed=None):
    """
    [Multi sequence]
    Generates a sequence of symbols (one-hot) randomly, each associated with a position (one-hot). 
    The model must sort the sequence according to positions, once the trigger signal is received.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - n_symbols (int): number of possible symbols
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_one_sample():
        # Create a sequence of symbols & a random order
        sequence = rng.integers(0, n_symbols, sequence_length)
        order = rng.permutation(sequence_length)

        # One-hot encode the sequence and order
        sequence_onehot = np.eye(n_symbols)[sequence]
        order_onehot = np.eye(sequence_length + 1)[order]
        sequence_order = np.concatenate([sequence_onehot, order_onehot], axis=1)

        # Create other input parts   
        marker = np.zeros((1, n_symbols + sequence_length + 1))
        marker[0, n_symbols+sequence_length] = 1
        zero_input_pad = np.zeros((sequence_length, n_symbols + sequence_length + 1))

        # Create the input & target
        input = np.concatenate([sequence_order, marker, zero_input_pad], axis=0)
        target = np.zeros((sequence_length+1+sequence_length, n_symbols))
        target[sequence_length + 1 + order] = sequence_onehot

        # Create the timesteps
        timesteps = np.arange(sequence_length + 1, sequence_length + 1 + sequence_length)

        return input, target, timesteps
    
    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')


def generate_bracket_matching(n_train=1000, n_valid=200, n_test=200, sequence_length=100, max_depth=5, seed=None):
    """
    [Multi sequence]
    Generates a sequence of parentheses that the model must validate.
    Tests the ability to maintain hierarchical context.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - sequence_length (int): sequence length
    - max_depth (int): maximum depth of parentheses
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets as well as
    their respective prediction timesteps. It also contains the category flag.
    """
    def generate_valid_sequence(length, max_depth):
        sequence = []
        stack = []
        remaining = length
        
        while remaining > 0:
            if len(stack) == 0 or (remaining > len(stack) and len(stack) < max_depth and rng.random() > 0.5):
                sequence.append('(')
                stack.append('(')
            else:
                sequence.append(')')
                stack.pop()
            remaining -= 1
        
        return sequence

    def check_validity(sequence):
        stack = []
        for bracket in sequence:
            if bracket == '(':
                stack.append(bracket)
            elif len(stack) == 0:
                return 0
            else:
                stack.pop()
        return int(len(stack) == 0)

    def mutate_sequence(sequence, proba=0.35):
        nb_mutated = int(len(sequence) * proba)
        index = rng.choice(len(sequence), nb_mutated, replace=False)
        mutation = ['(' if rng.random() > 0.5 else ')' for _ in range(nb_mutated)]
        for i, bracket in zip(index, mutation):
            sequence[i] = bracket
        return sequence

    def generate_one_sample():
        # Generate a sequence
        sequence = generate_valid_sequence(sequence_length, max_depth)
        sequence = sequence if rng.random() < 0.5 else mutate_sequence(sequence)
        validity = check_validity(sequence)

        # One-hot encode the sequence
        sequence_onehot = np.zeros((sequence_length+2, 3))
        for i, bracket in enumerate(sequence):
            sequence_onehot[i, 0 if bracket == '(' else 1] = 1
        sequence_onehot[-2, 2] = 1 # marker

        # Create the input & target
        input = sequence_onehot
        target = np.zeros((sequence_length+2, 2))
        target[-1, int(validity)] = 1

        # Create the timesteps
        timesteps = np.arange(sequence_length+1, sequence_length+2)

        return input, target, timesteps

    # Generate the samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')

def generate_csl(n_train=1000, n_valid=200, n_test=200, objects=None, colors=None, positions=None, seed=None):
    """
    [Multi sequence]
    Generates a Cross Situational Learning (CSL) task.
    The model must read a sentence (encoded in one-hot) describing two situations 
    and predict the corresponding roles/attributes at the end of the sequence.

    Args:
    - n_train (int): number of training samples
    - n_valid (int): number of validation samples
    - n_test (int): number of test samples
    - objects (list): list of objects in the CSL dataset
    - colors (list): list of colors in the CSL dataset
    - positions (list): list of positions in the CSL dataset
    - seed (int): random seed for reproducibility

    Return:
    - data (dict): dictionary containing the training, validation and test sets 
    as well as their respective prediction timesteps. It also contains the category flag.
    """
    
    # Default parameters
    if objects is None:
        objects = ['glass', 'orange', 'cup', 'bowl']
    if colors is None:
        colors = ['blue', 'orange', 'green', 'red']
    if positions is None:
        positions = ['left', 'right', ('center', 'middle')]

    # Instantiation of the global combinatorial generator
    csl_dataset = TwoSituationCSLDataset(objects=objects, colors=colors, positions=positions)
    X_all = csl_dataset.X
    Y_all = csl_dataset.Y

    N_total = X_all.shape[0]
    n_steps = X_all.shape[1]
    n_labels = Y_all.shape[1]

    def generate_one_sample():
        # Random draw of an index in the exhaustive CSL dataset
        idx = rng.integers(0, N_total)
        
        # Retrieval of the input sequence (encoded sentence)
        input_seq = X_all[idx]  # Shape: (n_steps, n_symbols)
        
        # We fill with zeros and place the true target only at the end
        target_seq = np.zeros((n_steps, n_labels))
        target_seq[-1, :] = Y_all[idx]
        
        # The model must predict only at the end of the sentence
        timesteps = np.array([n_steps - 1])

        return input_seq, target_seq, timesteps

    # Generation of samples
    rng = np.random.default_rng(seed)
    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='multi_classification')

def generate_associative_recall(n_train=1000, n_valid=200, n_test=200, sequence_length=16, num_pairs=4, n_symbols=16, seed=None):
    """
    [Multi sequence]
    Associative recall task. The model must memorize key-value pairs 
    and recall the value associated with a specific query key.
    """
    rng = np.random.default_rng(seed)
    
    SEP = n_symbols - 1
    PAD = n_symbols - 2
    vocab_kv = n_symbols - 2
    
    min_len = 2 * num_pairs + 2
    if sequence_length < min_len:
        raise ValueError(f"sequence_length={sequence_length} trop court. Minimum requis : {min_len}.")
    if vocab_kv < num_pairs:
        raise ValueError(f"n_symbols={n_symbols} insuffisant. Requis : {num_pairs + 2}.")
        
    def generate_one_sample():
        tokens = np.full(sequence_length, PAD, dtype=np.int64)
        keys = rng.choice(vocab_kv, size=num_pairs, replace=False)
        values = rng.integers(0, vocab_kv, size=num_pairs)
        
        # 1. Remplissage des paires
        for i, (k, v) in enumerate(zip(keys, values)):
            tokens[2 * i] = k
            tokens[2 * i + 1] = v
            
        # 2. Distracteurs
        n_distract = sequence_length - 2 * num_pairs - 2
        if n_distract > 0:
            tokens[2 * num_pairs : 2 * num_pairs + n_distract] = rng.integers(0, vocab_kv, size=n_distract)
            
        # 3. SEP + Requête
        tokens[-2] = SEP
        query_idx = rng.integers(0, num_pairs)
        tokens[-1] = keys[query_idx]
        
        # 4. Encodage One-hot de l'entrée
        input_seq = np.eye(n_symbols)[tokens]
        
        # 5. Création de la cible (uniquement valide à la fin)
        target_seq = np.zeros((sequence_length, n_symbols))
        target_value = values[query_idx]
        target_seq[-1, target_value] = 1
        
        # 6. Définition du timestep à évaluer (le dernier token)
        timesteps = np.array([sequence_length - 1])
        
        return input_seq, target_seq, timesteps

    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')


def generate_induction_heads(n_train=1000, n_valid=200, n_test=200, sequence_length=100, n_symbols=10, seed=None):
    """
    [Multi sequence]
    Induction heads task. The second half of the sequence is an exact copy 
    of the first half. The model must predict the next token in the second half.
    """
    rng = np.random.default_rng(seed)
    
    if sequence_length % 2 != 0:
        raise ValueError("sequence_length doit être pair pour la tâche induction_heads.")
        
    half_len = sequence_length // 2
    
    def generate_one_sample():
        # Generate the first half of the sequence with random tokens
        first_half = rng.integers(0, n_symbols, size=half_len)
        
        # Duplicate the first half to create the full sequence
        sequence = np.concatenate([first_half, first_half])
        
        # One-hot encoding of the input sequence
        input_seq = np.eye(n_symbols)[sequence]
        
        # Create the target sequence: the model must predict the next token in the second half
        target_seq = np.zeros((sequence_length, n_symbols))

        # The model must predict the next token in the second half, which is the same as the first half        
        for t in range(half_len, sequence_length - 1):
            target_seq[t, sequence[t+1]] = 1
            
        timesteps = np.arange(half_len, sequence_length - 1)
        
        return input_seq, target_seq, timesteps

    return _generate_train_test_samples(n_train, n_valid, n_test, generate_one_sample, category='classification')