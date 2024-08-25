import matplotlib.pyplot as plt
import numpy as np
from src.model import LinearModel
from src.train import train_model_with_early_stopping
import sys
import json
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append('.')

# Experiment with different optimizers and learning rates
def optimizers_with_lrs_plots(X_train, y_train, X_val, y_val, optimizers, learning_rates):
    input_size = X_train.shape[1]
    num_epochs = 100000
    patience = 100

    for optimizer_name, optimizer_class in tqdm(optimizers, desc=f'Visualizing learning rates'):
        # Load results from JSON
        print(f'Running {optimizer_name}')
        with open(f'results/plots/optimizers/{optimizer_name}_results.json', 'r') as f:
            results = json.load(f)
        
        train_loss_results = results['train_loss']
        val_loss_results = results['val_loss']
        train_f1_results = results['train_f1']
        val_f1_results = results['val_f1']
        epoch_times = results['epoch_times']

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

def plot_optimizer_performance(X_train, y_train, X_val, y_val, optimizers, num_epochs=100, patience=5):
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
    # Load optimizer_results from JSON
    with open('results/plots/compare_optimizers/optimizer_results.json', 'r') as f:
        optimizer_results = json.load(f)
        
    for optimizer_name, optimizer_class in tqdm(optimizers, desc='Comparing optimizers'):
        optimizer_results[optimizer_name] = {}
        model = LinearModel(X_train.shape[1])  # Initialize model
        optimizer = optimizer_class()
        
        train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
            model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
        )
        
        optimizer_results[optimizer_name]['train_losses'] = train_losses
        optimizer_results[optimizer_name]['val_losses'] = val_losses
        optimizer_results[optimizer_name]['train_f1_scores'] = train_f1_scores
        optimizer_results[optimizer_name]['val_f1_scores'] = val_f1_scores
        optimizer_results[optimizer_name]['epoch_times'] = epoch_time
        

    with open(f'results/plots/compare_optimizers/optimizer_results.json', 'w') as f:
        json.dump(optimizer_results, f)

    # Load optimizer_results from JSON
    with open('results/plots/compare_optimizers/optimizer_results.json', 'r') as f:
        optimizer_results = json.load(f)

    # Plot training loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['train_losses'], label=optimizer_name)
    plt.title('Training Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_train_loss.png')

    # Plot validation loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['val_losses'], label=optimizer_name)
    plt.title('Validation Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_val_loss.png')

    # Plot training F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['train_f1_scores'], label=optimizer_name)
    plt.title('Training F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_train_f1.png')

    # Plot validation F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['val_f1_scores'], label=optimizer_name)
    plt.title('Validation F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/compare_optimizers/optimizer_val_f1.png')

def visualize_minibatch_optimizer(X_train, y_train, X_val, y_val, optimizer_class, batch_sizes, learning_rates):
    input_size = X_train.shape[1]
    num_epochs = 10000
    patience = 100
    with open(f'results/plots/optimizers/minibatch_optimizer/results.json', 'r') as f:
        results = json.load(f)

    train_loss_results = results['train_loss']
    val_loss_results = results['val_loss']
    train_f1_results = results['train_f1']
    val_f1_results = results['val_f1']
    epoch_times = results['epoch_times']

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
    plt.savefig('results/plots/optimizers/minibatch_optimizer/train_loss.png')

    # Plot validation loss for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for key, losses in val_loss_results.items():
        batch_size, lr = key.split('_')
        plt.plot(losses, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer/val_loss.png')

    # Plot training F1 score for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for key, f1_scores in train_f1_results.items():
        batch_size, lr = key.split('_')
        plt.plot(f1_scores, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer/train_f1.png')

    # Plot validation F1 score for different batch sizes and learning rates
    plt.figure(figsize=(12, 6))
    for key, f1_scores in val_f1_results.items():
        batch_size, lr = key.split('_')
        plt.plot(f1_scores, label=f'Batch Size={batch_size}, LR={lr}')
    plt.title('Minibatch Optimizer: Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/optimizers/minibatch_optimizer/val_f1.png')

def compare_time(result_path):
    with open(result_path, 'r') as f:
        optimizers = json.load(f)

    # Plotting Train Loss
    plt.figure(figsize=(12, 6))
    for optimizer, results in optimizers.items():
        plt.plot(results['epoch_times'], results['train_losses'], label=optimizer)
    plt.title('Training Loss vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/plots/compare_optimizers/optimizer_train_loss_time.png')

    # Plotting Validation Loss
    plt.figure(figsize=(12, 6))
    for optimizer, results in optimizers.items():
        plt.plot(results['epoch_times'], results['val_losses'], label=optimizer)
    plt.title('Validation Loss vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/compare_optimizers/optimizer_val_loss_time.png')

    # Plotting Train F1 Score
    plt.figure(figsize=(12, 6))
    for optimizer, results in optimizers.items():
        plt.plot(results['epoch_times'], results['train_f1_scores'], label=optimizer)
    plt.title('Training F1 Score vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/compare_optimizers/optimizer_train_f1_time.png')

    # Plotting Validation F1 Score
    plt.figure(figsize=(12, 6))
    for optimizer, results in optimizers.items():
        plt.plot(results['epoch_times'], results['val_f1_scores'], label=optimizer)
    plt.title('Validation F1 Score vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/compare_optimizers/optimizer_val_f1_time.png')


def evaluate_with_regularization(X_train, y_train, X_val, y_val, optimizers, num_epochs=100, patience=5, l1_lambdas=[0.0], l2_lambdas=[0.0]):
    """
    Trains models with different optimizers and plots their performance on training loss, validation loss, training F1, and validation F1.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        optimizers (list): A list of tuples containing optimizer names and classes.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before early stopping.
        l1_lambdas (list): A list of regularization strengths for L1.
        l2_lambdas (list): A list of regularization strengths for L2.
    """
    optimizer_results_l1, optimizer_results_l2 = {}, {}

    for optimizer_name, optimizer_class in optimizers:
        optimizer_results_l1[optimizer_name] = {
            'train_losses': {},
            'val_losses': {},
            'train_f1_scores': {},
            'val_f1_scores': {},
            'epoch_times': {}
        }
        
        for l1_lambda in tqdm(l1_lambdas, desc=f'{optimizer_name} regularization:'):
            lambda_key = f"L1={l1_lambda}"
            model = LinearModel(X_train.shape[1], l1_lambda=l1_lambda, l2_lambda=0)  # Initialize model with L1 regularization
            optimizer = optimizer_class()
            
            train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
            model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
            )
            
            # Store results in the dictionary
            optimizer_results_l1[optimizer_name]['train_losses'][lambda_key] = train_losses
            optimizer_results_l1[optimizer_name]['val_losses'][lambda_key] = val_losses
            optimizer_results_l1[optimizer_name]['train_f1_scores'][lambda_key] = train_f1_scores
            optimizer_results_l1[optimizer_name]['val_f1_scores'][lambda_key] = val_f1_scores
            optimizer_results_l1[optimizer_name]['epoch_times'][lambda_key] = epoch_time

        # Save results to JSON
        with open(f'results/plots/regularization/optimizer_results_l1_{optimizer_name}.json', 'w') as f:
            json.dump(optimizer_results_l1, f)

    for optimizer_name, optimizer_class in optimizers:
        optimizer_results_l2[optimizer_name] = {
            'train_losses': {},
            'val_losses': {},
            'train_f1_scores': {},
            'val_f1_scores': {},
            'epoch_times': {}
        }
        
        for l2_lambda in tqdm(l2_lambdas, desc=f'{optimizer_name} regularization:'):
            lambda_key = f"L2={l2_lambda}"
            model = LinearModel(X_train.shape[1], l2_lambda=l2_lambda, l1_lambda=0)  # Initialize model with L1 regularization
            optimizer = optimizer_class()
            
            train_losses, val_losses, train_f1_scores, val_f1_scores, epoch_time = train_model_with_early_stopping(
            model, optimizer, X_train, y_train, X_val, y_val, num_epochs=num_epochs, patience=patience
            )
            
            # Store results in the dictionary
            optimizer_results_l2[optimizer_name]['train_losses'][lambda_key] = train_losses
            optimizer_results_l2[optimizer_name]['val_losses'][lambda_key] = val_losses
            optimizer_results_l2[optimizer_name]['train_f1_scores'][lambda_key] = train_f1_scores
            optimizer_results_l2[optimizer_name]['val_f1_scores'][lambda_key] = val_f1_scores
            optimizer_results_l2[optimizer_name]['epoch_times'][lambda_key] = epoch_time

        # Save results to JSON
        with open(f'results/plots/regularization/optimizer_results_l2_{optimizer_name}.json', 'w') as f:
            json.dump(optimizer_results_l2, f)

    # Plot training loss for all optimizers and lambda combinations
    for optimizer_name, results_l1 in optimizer_results_l1.items():
        results_l2 = optimizer_results_l2[optimizer_name]  # Get corresponding L2 results

        plt.figure(figsize=(12, 6))
        # Plot L1 results
        for lambda_key, train_losses in results_l1['train_losses'].items():
            plt.plot(train_losses, label=lambda_key)
        # Plot L2 results
        for lambda_key, train_losses in results_l2['train_losses'].items():
            if lambda_key != 'L2=0.0':
                plt.plot(train_losses, label=lambda_key, linestyle='--')
        plt.title(f'Training Loss for {optimizer_name} - L1 and L2')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/regularization/l1_l2_{optimizer_name}_train_loss.png')
        

        # Plot validation loss for all optimizers and lambda combinations
        plt.figure(figsize=(12, 6))
        # Plot L1 results
        for lambda_key, val_losses in results_l1['val_losses'].items():
            plt.plot(val_losses, label=lambda_key)
        # Plot L2 results
        for lambda_key, val_losses in results_l2['val_losses'].items():
            plt.plot(val_losses, label=lambda_key, linestyle='--')
        plt.title(f'Validation Loss for {optimizer_name} - L1 and L2')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/regularization/l1_l2_{optimizer_name}_val_loss.png')
        

        # Plot training F1 score for all optimizers and lambda combinations
        plt.figure(figsize=(12, 6))
        # Plot L1 results
        for lambda_key, train_f1_scores in results_l1['train_f1_scores'].items():
            plt.plot(train_f1_scores, label=lambda_key)
        # Plot L2 results
        for lambda_key, train_f1_scores in results_l2['train_f1_scores'].items():
            plt.plot(train_f1_scores, label=lambda_key, linestyle='--')
        plt.title(f'Training F1 Score for {optimizer_name} - L1 and L2')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/regularization/l1_l2_{optimizer_name}_train_f1.png')
        

        # Plot validation F1 score for all optimizers and lambda combinations
        plt.figure(figsize=(12, 6))
        # Plot L1 results
        for lambda_key, val_f1_scores in results_l1['val_f1_scores'].items():
            plt.plot(val_f1_scores, label=lambda_key)
        # Plot L2 results
        for lambda_key, val_f1_scores in results_l2['val_f1_scores'].items():
            plt.plot(val_f1_scores, label=lambda_key, linestyle='--')
        plt.title(f'Validation F1 Score for {optimizer_name} - L1 and L2')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/regularization/l1_l2_{optimizer_name}_val_f1.png')
        

def final_test(X_train, y_train, X_test, y_test, optimizers, num_epochs=100, patience=5, optimizer_hyperparam=None):
    """
    Trains models with different optimizers and plots their performance on training loss, validation loss, training F1, and validation F1.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        y_test: Test data labels.
        optimizers (list): A list of tuples containing optimizer names and classes.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before early stopping.
        optimizer_hyperparam (dict): A dictionary containing hyperparameters for each optimizer.
    """
    
    # Ensure results directory exists
    os.makedirs('results/plots/final_test/', exist_ok=True)
    
    # Load results from JSON
    with open('results/plots/final_test/optimizer_results.json', 'r') as f:
        optimizer_results = json.load(f)

    for optimizer_name, optimizer_class in tqdm(optimizers, desc='Final test'):
        optimizer_results[optimizer_name] = {}
        model = LinearModel(X_train.shape[1], **optimizer_hyperparam[optimizer_name])  # Initialize model
        
        # Initialize optimizer with specific hyperparameters
        optimizer = optimizer_class(**optimizer_hyperparam.get(optimizer_name, {}))
        
        train_losses, test_losses, train_f1_scores, test_f1_scores, epoch_time, test_predictions = train_model_with_early_stopping(
            model, optimizer, X_train, y_train, X_test, y_test, num_epochs=num_epochs, patience=patience, return_predictions=True
        )
        
        optimizer_results[optimizer_name]['train_losses'] = train_losses
        optimizer_results[optimizer_name]['test_losses'] = test_losses
        optimizer_results[optimizer_name]['train_f1_scores'] = train_f1_scores
        optimizer_results[optimizer_name]['test_f1_scores'] = test_f1_scores
        optimizer_results[optimizer_name]['epoch_times'] = epoch_time
        optimizer_results[optimizer_name]['test_predictions'] = test_predictions
        
    # Save results to JSON
    with open('results/plots/final_test/optimizer_results.json', 'w') as f:
        json.dump(optimizer_results, f)

    # Load results from JSON
    with open('results/plots/final_test/optimizer_results.json', 'r') as f:
        optimizer_results = json.load(f)

    # Plot training loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['train_losses'], label=optimizer_name)
    plt.title('Training Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/final_test/optimizer_train_loss.png')

    # Plot test loss for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['test_losses'], label=optimizer_name)
    plt.title('Test Loss for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/final_test/optimizer_test_loss.png')

    # Plot training F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['train_f1_scores'], label=optimizer_name)
    plt.title('Training F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/final_test/optimizer_train_f1.png')

    # Plot test F1 score for all optimizers
    plt.figure(figsize=(12, 6))
    for optimizer_name, value in optimizer_results.items():
        plt.plot(value['test_f1_scores'], label=optimizer_name)
    plt.title('Test F1 Score for Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/final_test/optimizer_test_f1.png')

    # Create a dataframe of accuracy, precision, recall, and f1 scores
    results_df = pd.DataFrame(columns=['Optimizer', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    X_train, y_train, X_test, y_test = X_train.get(), y_train.get(), X_test.get(), y_test.get() 
    for optimizer_name, value in optimizer_results.items():
        accuracy = accuracy_score(y_test, value['test_predictions'])
        precision = precision_score(y_test, value['test_predictions'])
        recall = recall_score(y_test, value['test_predictions'])
        f1 = f1_score(y_test, value['test_predictions'])
        results_df = pd.concat([results_df, pd.DataFrame({'Optimizer': [optimizer_name], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]})], ignore_index=True)
    
    
    # Train model using sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)
    test_predictions_sklearn = (model_sklearn.predict(X_test) > 0.5).astype(int)

    # Calculate evaluation metrics
    accuracy_sklearn = accuracy_score(y_test, test_predictions_sklearn)
    precision_sklearn = precision_score(y_test, test_predictions_sklearn)
    recall_sklearn = recall_score(y_test, test_predictions_sklearn)
    f1_score_sklearn = f1_score(y_test, test_predictions_sklearn)

    # Append results to dataframe
    results_df = pd.concat([results_df, pd.DataFrame({'Optimizer': ['sklearn'], 'Accuracy': [accuracy_sklearn], 'Precision': [precision_sklearn], 'Recall': [recall_sklearn], 'F1 Score': [f1_score_sklearn]})], ignore_index=True)
    # Save dataframe to CSV
    results_df.to_csv('results/plots/final_test/optimizer_results.csv', index=False)
