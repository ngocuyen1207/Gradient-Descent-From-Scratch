import sys
sys.path.append('.')
sys.path.append('/home/uyen/smoke_detection_training/')

# Load and preprocess data
from src.optimizers import (
    GradientDescentOptimizer, MiniBatchOptimizer, MomentumOptimizer, 
    RMSpropOptimizer, AdamOptimizer, AdaGradOptimizer, AdamWOptimizer,
    NadamOptimizer, AdadeltaOptimizer, NesterovOptimizer
)
from src.data_loader import load_and_preprocess_data
from src.visualize import optimizers_with_lrs_plots, plot_optimizer_performance, visualize_minibatch_optimizer, compare_time, final_test, evaluate_with_regularization
from sklearn.linear_model import LogisticRegression

file_path = 'data/stackoverflow_full.csv'
target_column = 'Employed'  # Replace with the actual target column name

# Load the data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(file_path, target_column)

# Different learning rates for each optimizer
optimizers = [
    # ("GradientDescent", GradientDescentOptimizer),
    # ("Mini-Batch Optimizer", MiniBatchOptimizer),
    ("Momentum", MomentumOptimizer),
    # ("RMSprop", RMSpropOptimizer),
    # ("Adam", AdamOptimizer),
    # ("AdaGrad", AdaGradOptimizer),
    # ("AdamW", AdamWOptimizer),
    # ("Nadam", NadamOptimizer),
    # ("Adadelta", AdadeltaOptimizer),
    # ("Nesterov", NesterovOptimizer),
]
optimizer_hyperparam = {
    "GradientDescent": {"learning_rate": 0.1, "l1_lambda": 0, "l2_lambda": 0},
    "Mini-Batch Optimizer": {"learning_rate": 0.1, "batch_size": 64, "l1_lambda": 0, "l2_lambda": 0},
    "Momentum": {"learning_rate": 0.1, "l1_lambda": 0.0, "l2_lambda": 0.001},
    "RMSprop": {"learning_rate": 0.1, "l1_lambda": 0.0, "l2_lambda": 0.0},
    "Adam": {"learning_rate": 0.1, "l1_lambda": 0, "l2_lambda": 0},
    "AdaGrad": {"learning_rate": 0.1, "l1_lambda": 0, "l2_lambda": 0},
    "AdamW": {"learning_rate": 0.1, "l1_lambda": 0, "l2_lambda": 0},
    "Nadam": {"learning_rate": 0.1, "l1_lambda": 0.0, "l2_lambda": 0.0},
    "Adadelta": {"learning_rate": 0.1, "l1_lambda": 0, "l2_lambda": 0.001},
    "Nesterov": {"learning_rate": 0.1, "l1_lambda": 0.0, "l2_lambda": 0.0},
}
learning_rates = [1, 2, 5]
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
l1_lambdas = [0.0, 0.001, 0.01, 0.1, 1, 2, 5, 10]
l2_lambdas = [0.001, 0.01, 0.1, 1, 2, 5, 10]

optimizers_with_lrs_plots(X_train, y_train, X_val, y_val, optimizers, learning_rates)
# plot_optimizer_performance(X_train, y_train, X_val, y_val, optimizers, num_epochs=10000, patience=100)
# visualize_minibatch_optimizer(X_train, y_train, X_val, y_val, MiniBatchOptimizer, batch_sizes, learning_rates)
# compare_time('results/plots/compare_optimizers/optimizer_results.json')
# evaluate_with_regularization(X_train, y_train, X_val, y_val, optimizers, num_epochs=10000, patience=100, l1_lambdas=l1_lambdas, l2_lambdas=l2_lambdas)
# final_test(X_train, y_train, X_test, y_test, optimizers, num_epochs=10000, patience=100, optimizer_hyperparam=optimizer_hyperparam)
