import cupy as cp
import pickle


class LinearModel:
    def __init__(self, input_size, l1_lambda=0.0, l2_lambda=0.0):
        self.weights = cp.random.randn(input_size)
        self.bias = cp.random.randn()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, X):
        linear_output = cp.dot(X, self.weights) + self.bias
        return 1 / (1 + cp.exp(-linear_output))  # Sigmoid function for probabilities

    def compute_loss(self, predictions, y):
        # Binary Cross-Entropy Loss
        loss = -cp.mean(y * cp.log(predictions) + (1 - y) * cp.log(1 - predictions))
        
        # Add L1 and L2 regularization terms
        l1_term = self.l1_lambda * cp.sum(cp.abs(self.weights))
        l2_term = self.l2_lambda * cp.sum(cp.square(self.weights))
        
        total_loss = loss + l1_term + l2_term
        return total_loss

    def backward(self, X, y, predictions, learning_rate):
        # Gradient calculation for BCE loss
        n = X.shape[0]
        dW = cp.dot(X.T, (predictions - y)) / n
        db = cp.sum(predictions - y) / n
        
        # Add gradients for L1 and L2 regularization
        if self.l1_lambda > 0:
            dW += self.l1_lambda * cp.sign(self.weights)
        if self.l2_lambda > 0:
            dW += 2 * self.l2_lambda * self.weights

        # Update weights and bias
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db