"""
neural_network.py — Dense neural-network classifier for mass-ordering discrimination.

Architecture
------------
Input → Dense(64, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)

The network is trained as a binary classifier:
    label = 1  →  Normal Ordering (NO, Δm²₃₂ > 0)
    label = 0  →  Inverted Ordering (IO, Δm²₃₂ < 0)

Training uses binary cross-entropy loss and a cosine-decay learning-rate
schedule with warm restarts, which works well for this type of problem.

Functions
---------
create_nn_model    : Build the Keras model.
train_nn_model     : Train on labelled histogram data.
evaluate_nn_model  : Evaluate loss (and optionally accuracy) on a dataset.
load_nn_model      : Load a previously saved model from disk.
plot_classifier_output : Histogram of classifier scores by true class.
plot_roc_curve     : ROC curve from training/validation predictions.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def create_nn_model(num_bins, learning_rate=0.01, decay_steps=125):
    """
    Build the dense neural-network classifier.

    Parameters
    ----------
    num_bins : int
        Total number of input features (concatenated histogram bins across all
        four detection channels).
    learning_rate : float
        Initial learning rate for the cosine-decay schedule.  Default 0.01.
    decay_steps : int
        Number of steps in one cosine-decay cycle.  Default 125.

    Returns
    -------
    model : keras.Model
        Compiled Keras model ready for training.
    """
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=decay_steps,
        t_mul=1.0,   # cycle length stays constant
        m_mul=1.0,   # peak learning rate stays constant
        alpha=0.0    # decay all the way to zero
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model = models.Sequential([
        layers.Input(shape=(num_bins,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1,  activation='sigmoid'),   # output ∈ (0, 1); 1 = NO
    ])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_nn_model(model, train_data, train_labels,
                   epochs=10, batch_size=32, validation_split=0.1,
                   patience=20, log_dir=None):
    """
    Train the model on labelled histogram data.

    When ``validation_split > 0`` an EarlyStopping callback is attached: it
    monitors ``val_loss``, stops training after ``patience`` epochs without
    improvement (min_delta = 1e-6), and restores the weights from the
    epoch with the best validation loss.  Set ``patience`` to a very large
    number to effectively disable it.

    Parameters
    ----------
    model : keras.Model
        Model created by create_nn_model.
    train_data : ndarray, shape (N, num_bins, 1)
        Flattened and reshaped histogram arrays.
    train_labels : ndarray, shape (N,)
        Binary labels (1 = NO, 0 = IO).
    epochs : int
        Maximum number of training epochs (early stopping may terminate sooner).
    batch_size : int
        Mini-batch size.
    validation_split : float
        Fraction of training data held out for validation loss monitoring.
        Required to be > 0 for early stopping to engage.
    patience : int
        EarlyStopping patience (epochs without val_loss improvement).
    log_dir : str or None
        If given, TensorBoard logs are written to this directory.
        Set to None (default) to skip TensorBoard logging.

    Returns
    -------
    model : keras.Model
        Trained model (same object, modified in place; best-val weights restored).
    """
    callbacks = []
    if validation_split > 0:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=1,
        ))
    if log_dir is not None:
        from tensorflow.keras.callbacks import TensorBoard
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

    model.fit(
        train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )
    return model


def evaluate_nn_model(model, data, labels):
    """
    Evaluate the model and return (loss, accuracy).

    Parameters
    ----------
    model : keras.Model
    data : ndarray
    labels : ndarray

    Returns
    -------
    loss : float
    accuracy : float
    """
    results = model.evaluate(data, labels, verbose=0)
    loss, accuracy = results[0], results[1]
    return loss, accuracy


def load_nn_model(model_path):
    """
    Load a previously saved Keras model from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to a .h5 file or a SavedModel directory.

    Returns
    -------
    model : keras.Model

    Notes
    -----
    Loaded with ``compile=False`` because the model is only used for inference
    here (no further training or ``evaluate``).  This also avoids the harmless
    Keras warning about compiled metrics not yet being built.
    """
    return models.load_model(str(model_path), compile=False)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_classifier_output(y_pred_prob, labels, output_path=None):
    """
    Plot histograms of the classifier score separated by true class.

    Parameters
    ----------
    y_pred_prob : ndarray, shape (N,)
        Classifier output scores in [0, 1].
    labels : ndarray, shape (N,)
        True binary labels (1 = NO, 0 = IO).
    output_path : str or None
        File path to save the figure.  Set to None to show interactively.
    """
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 6))
    # NO — filled Wong-blue with solid outline
    ax.hist(y_pred_prob[labels == 1], bins=30, alpha=0.35,
            color='#0072B2', edgecolor='#0072B2', linewidth=1.5,
            histtype='stepfilled', label='NO (label=1)')
    # IO — Wong-vermillion dashed contour-only histogram
    ax.hist(y_pred_prob[labels == 0], bins=30,
            histtype='step', color='#D55E00',
            linewidth=2.5, linestyle='--', label='IO (label=0)')
    ax.set_xlabel('NN Classifier Output')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)


def plot_roc_curve(y_pred_prob, labels, output_path=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_pred_prob : ndarray, shape (N,)
        Classifier output scores in [0, 1].
    labels : ndarray, shape (N,)
        True binary labels (1 = NO, 0 = IO).
    output_path : str or None
        File path to save the figure.  Set to None to show interactively.
    """
    plt.rcParams.update({'font.size': 20})
    fpr, tpr, _ = roc_curve(labels, y_pred_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('NO False Positive Rate')
    ax.set_ylabel('NO True Positive Rate')
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)
