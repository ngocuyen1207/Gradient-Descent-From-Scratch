import cupy as cp

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, model, X, y, predictions):
        # Calculate the gradients
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        # Update weights and biases
        model.weights -= self.learning_rate * dW
        model.bias -= self.learning_rate * db


class MiniBatchOptimizer:
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def update(self, model, X, y, predictions):
        # Randomly shuffle the data
        indices = cp.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        X_batch = X_shuffled[:self.batch_size]
        y_batch = y_shuffled[:self.batch_size]
        
        # Forward pass: compute predictions
        predictions = model.forward(X_batch)
        
        # Calculate the gradients
        dW = (cp.dot(X_batch.T, (predictions - y_batch)) / X_batch.shape[0])
        db = cp.sum(predictions - y_batch) / X_batch.shape[0]
        
        # Update weights and biases
        model.weights -= self.learning_rate * dW
        model.bias -= self.learning_rate * db


class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = 0
        self.velocity_b = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]
        
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db
        
        model.weights += self.velocity_w
        model.bias += self.velocity_b


class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = 0
        self.s_b = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]
        
        self.s_w = self.beta * self.s_w + (1 - self.beta) * (dW ** 2)
        self.s_b = self.beta * self.s_b + (1 - self.beta) * (db ** 2)
        
        model.weights -= self.learning_rate * dW / (cp.sqrt(self.s_w) + self.epsilon)
        model.bias -= self.learning_rate * db / (cp.sqrt(self.s_b) + self.epsilon)


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        m_w_hat = self.m_w / (1 - cp.power(self.beta1, self.t))
        v_w_hat = self.v_w / (1 - cp.power(self.beta2, self.t))
        
        m_b_hat = self.m_b / (1 - cp.power(self.beta1, self.t))
        v_b_hat = self.v_b / (1 - cp.power(self.beta2, self.t))
        
        model.weights -= self.learning_rate * m_w_hat / (cp.sqrt(v_w_hat) + self.epsilon)
        model.bias -= self.learning_rate * m_b_hat / (cp.sqrt(v_b_hat) + self.epsilon)


class AdaGradOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G_w = 0
        self.G_b = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        self.G_w += dW ** 2
        self.G_b += db ** 2

        model.weights -= (self.learning_rate / (cp.sqrt(self.G_w) + self.epsilon)) * dW
        model.bias -= (self.learning_rate / (cp.sqrt(self.G_b) + self.epsilon)) * db


class AdamWOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        m_w_hat = self.m_w / (1 - cp.power(self.beta1, self.t))
        v_w_hat = self.v_w / (1 - cp.power(self.beta2, self.t))
        
        m_b_hat = self.m_b / (1 - cp.power(self.beta1, self.t))
        v_b_hat = self.v_b / (1 - cp.power(self.beta2, self.t))
        
        # Weight decay step
        model.weights -= self.weight_decay * model.weights

        # AdamW update
        model.weights -= self.learning_rate * m_w_hat / (cp.sqrt(v_w_hat) + self.epsilon)
        model.bias -= self.learning_rate * m_b_hat / (cp.sqrt(v_b_hat) + self.epsilon)


class AdadeltaOptimizer:
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-8, **kwargs):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.E_w = 0
        self.E_b = 0
        self.delta_w = 0
        self.delta_b = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        self.E_w = self.rho * self.E_w + (1 - self.rho) * (dW ** 2)
        self.E_b = self.rho * self.E_b + (1 - self.rho) * (db ** 2)

        delta_w = - (cp.sqrt(self.delta_w + self.epsilon) / cp.sqrt(self.E_w + self.epsilon)) * dW
        delta_b = - (cp.sqrt(self.delta_b + self.epsilon) / cp.sqrt(self.E_b + self.epsilon)) * db

        model.weights += delta_w
        model.bias += delta_b
        self.delta_w = self.rho * self.delta_w + (1 - self.rho) * (delta_w ** 2)
        self.delta_b = self.rho * self.delta_b + (1 - self.rho) * (delta_b ** 2)


class NesterovOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, **kwargs):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = 0
        self.velocity_b = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]
        
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db
        
        model.weights += self.momentum * self.velocity_w - self.learning_rate * dW
        model.bias += self.momentum * self.velocity_b - self.learning_rate * db

class NadamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def update(self, model, X, y, predictions):
        dW = (cp.dot(X.T, (predictions - y)) / X.shape[0])
        db = cp.sum(predictions - y) / X.shape[0]

        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        m_w_hat = self.m_w / (1 - cp.power(self.beta1, self.t))
        v_w_hat = self.v_w / (1 - cp.power(self.beta2, self.t))
        
        m_b_hat = self.m_b / (1 - cp.power(self.beta1, self.t))
        v_b_hat = self.v_b / (1 - cp.power(self.beta2, self.t))
        
        model.weights -= self.learning_rate * (self.beta1 * m_w_hat + (1 - self.beta1) * dW) / (cp.sqrt(v_w_hat) + self.epsilon)
        model.bias -= self.learning_rate * (self.beta1 * m_b_hat + (1 - self.beta1) * db) / (cp.sqrt(v_b_hat) + self.epsilon)