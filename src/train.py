from src.model import LinearModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time

def train_model_with_early_stopping(model, optimizer, X_train, y_train, X_val, y_val, num_epochs=10000, patience=100):
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    patience_counter = 0
    epoch_times = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Forward pass on training data
        predictions_train = model.forward(X_train)
        loss_train = model.compute_loss(predictions_train, y_train)

        # Update the model using the optimizer
        optimizer.update(model, X_train, y_train, predictions_train)

        # Forward pass on validation data
        predictions_val = model.forward(X_val)
        loss_val = model.compute_loss(predictions_val, y_val)

        # Record the training and validation losses
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # Calculate F1 scores
        train_predictions_binary = (predictions_train >= 0.5).astype(int)
        val_predictions_binary = (predictions_val >= 0.5).astype(int)
        
        train_f1 = f1_score(y_train.get(), train_predictions_binary.get())
        val_f1 = f1_score(y_val.get(), val_predictions_binary.get())
        
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0  # Reset counter if improvement
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    train_losses = interpolate_missing_data([arr.get() for arr in train_losses]).tolist()
    val_losses = interpolate_missing_data([arr.get() for arr in val_losses]).tolist()
    train_f1_scores = interpolate_missing_data(train_f1_scores).tolist()
    val_f1_scores = interpolate_missing_data(val_f1_scores).tolist()

    assert len(train_losses) == len(val_losses) == len(train_f1_scores) == len(val_f1_scores) == len(epoch_times), "Lengths of the lists are not equal"

    if len(train_losses) != len(val_losses) or len(train_f1_scores) != len(val_f1_scores) or len(epoch_times) != len(train_losses):
        print(f"Length of train_losses: {len(train_losses)}")
        print(f"Length of val_losses: {len(val_losses)}")
        print(f"Length of train_f1_scores: {len(train_f1_scores)}")
        print(f"Length of val_f1_scores: {len(val_f1_scores)}")
        print(f"Length of epoch_times: {len(epoch_times)}")

    return train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_times

def interpolate_missing_data(data):
    """Interpolate missing data points in a list."""
    data = np.array(data)
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data
