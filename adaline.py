import numpy as np

class Adaline:
    def __init__(
        self,
        initial_w1=None,
        initial_w2=None,
        initial_w3=None,
        learning_rate=0.05,
        epochs=100,
        initial_bias=0,
    ):
        # Store the training configuration.
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Generate missing initial values randomly.
        self.initial_w1 = np.random.uniform(-1, 1) if initial_w1 is None else initial_w1
        self.initial_w2 = np.random.uniform(-1, 1) if initial_w2 is None else initial_w2
        self.initial_w3 = np.random.uniform(-1, 1) if initial_w3 is None else initial_w3
        self.initial_bias = np.random.uniform(-1, 1) if initial_bias is None else initial_bias

        self.w1 = self.initial_w1
        self.w2 = self.initial_w2
        self.w3 = self.initial_w3
        self.bias = self.initial_bias

        # Store parameter updates and one MSE value per epoch.
        self.history = []

    def reset_parameters(self):
        # Restore the initial parameters before a new training process.
        self.w1 = self.initial_w1
        self.w2 = self.initial_w2
        self.w3 = self.initial_w3
        self.bias = self.initial_bias
        self.history = []

    def weighted_sum(self, x1, x2, x3):
        # Calculate the linear output of the neuron.
        return self.w1 * x1 + self.w2 * x2 + self.w3 * x3 + self.bias

    def predict_single(self, x1, x2, x3):
        # Return the continuous ADALINE output.
        return self.weighted_sum(x1, x2, x3)

    def predict(self, X):
        # Calculate the output for every input pattern.
        return np.array([
            self.weighted_sum(point[0], point[1], point[2])
            for point in X
        ])

    def update_parameters(self, x1, x2, x3, target, output):
        # Apply the delta rule using the continuous error.
        error = target - output
        adjustment = self.learning_rate * error

        self.w1 += adjustment * x1
        self.w2 += adjustment * x2
        self.w3 += adjustment * x3
        self.bias += adjustment

    def train(self, X, y):
        self.reset_parameters()

        for epoch in range(1, self.epochs + 1):
            for point_index in range(len(X)):
                x1, x2, x3 = X[point_index]
                target = y[point_index]
                output = self.predict_single(x1, x2, x3)
                error = target - output

                self.update_parameters(x1, x2, x3, target, output)
                self.history.append({
                    "epoch": epoch,
                    "point": point_index + 1,
                    "target": target,
                    "output": output,
                    "error": error,
                    "w1": self.w1,
                    "w2": self.w2,
                    "w3": self.w3,
                    "bias": self.bias,
                    "mse": None,
                })

            # Store MSE only once per epoch for the error plot.
            errors = y - self.predict(X)
            self.history[-1]["mse"] = float(np.mean(errors ** 2))

        return self.history
