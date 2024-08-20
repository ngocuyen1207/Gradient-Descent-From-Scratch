import cupy as cp

class LinearModel:
    def __init__(self, input_size):
        self.weights = cp.random.randn(input_size)
        self.bias = cp.random.randn()

    def forward(self, X):
        linear_output = cp.dot(X, self.weights) + self.bias
        return 1 / (1 + cp.exp(-linear_output))  # Sigmoid function for probabilities

    def compute_loss(self, predictions, y):
        # Binary Cross-Entropy Loss
        loss = -cp.mean(y * cp.log(predictions) + (1 - y) * cp.log(1 - predictions))
        return loss

    def backward(self, X, y, predictions, learning_rate):
        # Gradient calculation for BCE loss
        n = X.shape[0]
        dW = cp.dot(X.T, (predictions - y)) / n
        db = cp.sum(predictions - y) / n
        
        # Update weights and bias
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
