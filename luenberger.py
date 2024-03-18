import numpy as np

class LuenbergerObserver:
    def __init__(self, A, B, C):
        self.A = A  # System matrix
        self.B = B  # Input matrix
        self.C = C  # Output matrix

        # Check system controllability
        if np.linalg.matrix_rank(np.hstack([self.A, self.B])) != self.A.shape[0]:
            raise ValueError("System is not controllable")

        # Check system observability
        if np.linalg.matrix_rank(np.vstack([self.C, np.dot(self.C, self.A)])) != self.A.shape[0]:
            raise ValueError("System is not observable")

        # Observer gain matrix
        self.L = self.compute_observer_gain()

        # Initial state estimate
        self.x_hat = np.zeros((self.A.shape[0], 1))

    def compute_observer_gain(self):
        # Solve the discrete-time algebraic Riccati equation (DARE)
        P = np.linalg.solve(
            np.dot(np.dot(self.A.T, self.A), -self.C.T) - np.dot(self.C, self.C.T),
            np.dot(self.A.T, self.B)
        )
        return np.dot(P, self.C.T)

    def update_estimate(self, u, y):
        # Prediction step
        self.x_hat = np.dot(self.A, self.x_hat) + np.dot(self.B, u)

        # Correction step (measurement update)
        y_hat = np.dot(self.C, self.x_hat)
        self.x_hat += np.dot(self.L, (y - y_hat))

        return self.x_hat

# Define the system matrices
A = np.array([[0.8, 0.2], [0.1, 0.9]])
B = np.array([[1], [0]])
C = np.array([[1, 0]])

# Create the Luenberger observer
observer = LuenbergerObserver(A, B, C)

# Simulate the system and observer
for t in range(10):
    u = np.random.rand(1, 1)  # Input
    y = np.dot(C, np.dot(A, observer.x_hat)) + np.dot(C, np.dot(B, u)) + np.random.randn(1, 1) * 0.1  # Output measurement with noise
    x_estimate = observer.update_estimate(u, y)
    print(f"Time step {t}: Estimated state = {x_estimate.flatten()}")
