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

file_path = 'data/stackoverflow_full.csv'
target_column = 'Employed'  # Replace with the actual target column name

# Load the data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(file_path, target_column)

# Different learning rates for each optimizer
optimizers = [
    ("GradientDescent", GradientDescentOptimizer),
    ("Mini-Batch Optimizer", MiniBatchOptimizer),
    ("Momentum", MomentumOptimizer),
    ("RMSprop", RMSpropOptimizer),
    ("Adam", AdamOptimizer),
    ("AdaGrad", AdaGradOptimizer),
    ("AdamW", AdamWOptimizer),
    ("Nadam", NadamOptimizer),
    ("Adadelta", AdadeltaOptimizer),
    ("Nesterov", NesterovOptimizer),
]

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [1, 4, 8, 16, 32, 64]
l1_lambdas = [0.0, 0.001, 0.01, 0.1, 1, 2, 5]
l2_lambdas = [0.0, 0.001, 0.01, 1, 2, 5]

# optimizers_with_lrs_plots(X_train, y_train, X_val, y_val, optimizers, learning_rates)
# plot_optimizer_performance(X_train, y_train, X_val, y_val, optimizers, num_epochs=10000, patience=100)
# visualize_minibatch_optimizer(X_train, y_train, X_val, y_val, MiniBatchOptimizer, batch_sizes, learning_rates)
compare_time('results/plots/compare_optimizers/optimizer_results.json')
# final_test(X_train, y_train, X_test, y_test, optimizers, num_epochs=10000, patience=100)
# evaluate_with_regularization(X_train, y_train, X_val, y_val, optimizers, num_epochs=10000, patience=100, l1_lambdas=l1_lambdas, l2_lambdas=l2_lambdas)