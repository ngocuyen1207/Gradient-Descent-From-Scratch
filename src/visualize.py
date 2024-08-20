import matplotlib.pyplot as plt
import numpy as np
from src.model import LinearModel
from src.train import train_model_with_early_stopping
import sys
import json
from tqdm import tqdm
sys.path.append('.')

# Experiment with different optimizers and learning rates
def optimizers_with_lrs_plots(X_train, y_train, X_val, y_val, optimizers, learning_rates):
    input_size = X_train.shape[1]
    num_epochs = 10000
    patience = 100

    for optimizer_name, optimizer_class in tqdm(optimizers, desc=f'Visualizing optimizers'):
        train_loss_results = {}
        val_loss_results = {}
        train_f1_results = {}
        val_f1_results = {}
        epoch_times = {}

        for lr in learning_rates:
            model = LinearModel(input_size)
            optimizer = optimizer_class(learning_rate=lr)
            
            train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
                model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
            )
            
            # Store results
            train_loss_results[lr] = train_losses
            val_loss_results[lr] = val_losses
            train_f1_results[lr] = train_f1_scores
            val_f1_results[lr] = val_f1_scores
            epoch_times[lr] = epoch_time
            
        # Save results to JSON
        results = {
            'train_loss': train_loss_results,
            'val_loss': val_loss_results,
            'train_f1': train_f1_results,
            'val_f1': val_f1_results,
            'epoch_times': epoch_times
        }

        with open(f'results/plots/optimizers/{optimizer_name}_results.json', 'w') as f:
            json.dump(results, f)
                
        # Plot training loss for different learning rates
        plt.figure(figsize=(12, 6))
        for lr, losses in train_loss_results.items():
            plt.plot(losses, label=f'LR={lr}')
        plt.title(f'{optimizer_name}: Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/optimizers/{optimizer_name}_train_loss.png')

        # Plot validation loss for different learning rates
        plt.figure(figsize=(12, 6))
        for lr, losses in val_loss_results.items():
            plt.plot(losses, label=f'LR={lr}')
        plt.title(f'{optimizer_name}: Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/optimizers/{optimizer_name}_val_loss.png')

        # Plot training F1 score for different learning rates
        plt.figure(figsize=(12, 6))
        for lr, f1_scores in train_f1_results.items():
            plt.plot(f1_scores, label=f'LR={lr}')
        plt.title(f'{optimizer_name}: Training F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/optimizers/{optimizer_name}_train_f1.png')

        # Plot validation F1 score for different learning rates
        plt.figure(figsize=(12, 6))
        for lr, f1_scores in val_f1_results.items():
            plt.plot(f1_scores, label=f'LR={lr}')
        plt.title(f'{optimizer_name}: Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/optimizers/{optimizer_name}_val_f1.png')

def plot_optimizer_performance(X_train, y_train, X_val, y_val, optimizers, learning_rate=0.01, num_epochs=100, patience=5):
    """
    Trains models with different optimizers and plots their performance on training loss, validation loss, training F1, and validation F1.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        optimizers (list): A list of tuples containing optimizer names and classes.
        learning_rate (float): The learning rate to use for all optimizers.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before early stopping.
    """
    optimizer_results = {}
    epoch_times= {}

    for optimizer_name, optimizer_class in tqdm(optimizers, desc='Comparing optimizers'):
        optimizer_results[optimizer_name] = {}
        model = LinearModel(X_train.shape[1])  # Initialize model
        optimizer = optimizer_class(learning_rate=learning_rate)
        
        train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
            model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
        )
        
        optimizer_results[optimizer_name]['train_losses'] = train_losses
        optimizer_results[optimizer_name]['val_losses'] = val_losses
        optimizer_results[optimizer_name]['train_f1_scores'] = train_f1_scores
        optimizer_results[optimizer_name]['val_f1_scores'] = val_f1_scores
        epoch_times[optimizer_name] = epoch_time
    # Save results to JSON
    results = {
        'optimizer_results': optimizer_results,
        'epoch_times': epoch_times
    }
    with open(f'results/plots/compare_optimizers/optimizer_results.json', 'w') as f:
        json.dump(results, f)

    # Plot training loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, (train_losses, _, _, _) in optimizer_results.items():
        plt.plot(train_losses, label=optimizer_name)
    plt.title('Training Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Use logarithmic scale
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_train_loss.png')
    plt.show()

    # Plot validation loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, (_, val_losses, _, _) in optimizer_results.items():
        plt.plot(val_losses, label=optimizer_name)
    plt.title('Validation Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Use logarithmic scale
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_val_loss.png')
    plt.show()

    # Plot training F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, (_, _, train_f1_scores, _) in optimizer_results.items():
        plt.plot(train_f1_scores, label=optimizer_name)
    plt.title('Training F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_train_f1.png')
    plt.show()

    # Plot validation F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, (_, _, _, val_f1_scores) in optimizer_results.items():
        plt.plot(val_f1_scores, label=optimizer_name)
    plt.title('Validation F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_val_f1.png')
    plt.show()

def visualize_minibatch_optimizer(X_train, y_train, X_val, y_val, optimizer_class, batch_sizes, learning_rates):
    input_size = X_train.shape[1]
    num_epochs = 10000
    patience = 100

    train_loss_results = {}
    val_loss_results = {}
    train_f1_results = {}
    val_f1_results = {}
    epoch_times = {}

    for batch_size in tqdm(batch_sizes, desc='Visualizing minibatch optimizer'):
        for lr in learning_rates:
            model = LinearModel(input_size)
            optimizer = optimizer_class(learning_rate=lr, batch_size=batch_size)
            
            train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
                model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
            )
            
            # Store results
            key = f"{batch_size}_{lr}"
            train_loss_results[key] = train_losses
            val_loss_results[key] = val_losses
            train_f1_results[key] = train_f1_scores
            val_f1_results[key] = val_f1_scores
            epoch_times[key] = epoch_time
            
    # Save results to JSON
    results = {
        'train_loss': train_loss_results,
        'val_loss': val_loss_results,
        'train_f1': train_f1_results,
        'val_f1': val_f1_results,
        'epoch_times': epoch_times
    }

    with open(f'results/plots/optimizers/minibatch_optimizer/results.json', 'w') as f:
        json.dump(results, f)
            
    # Plot training loss for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for key, losses in train_loss_results.items():
        batch_size, lr = key.split('_')
        plt.plot(losses, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer_train_loss.png')
    plt.show()

    # Plot validation loss for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for (batch_size, lr), losses in val_loss_results.items():
        plt.plot(losses, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer_val_loss.png')
    plt.show()

    # Plot training F1 score for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for (batch_size, lr), f1_scores in train_f1_results.items():
        plt.plot(f1_scores, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer_train_f1.png')
    plt.show()

    # Plot validation F1 score for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for (batch_size, lr), f1_scores in val_f1_results.items():
        plt.plot(f1_scores, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer_val_f1.png')
    plt.show()

