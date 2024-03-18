import numpy as np
from scipy.linalg import inv

class EKFRS_Smoother:
    def __init__(self, f, h, Q, R, dt):
        self.f = f  # System dynamics function
        self.h = h  # Measurement function
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Measurement noise covariance matrix
        self.dt = dt  # Time step for numerical integration

        self.x_hat = None  # Initial state estimate
        self.P = None  # Initial error covariance matrix

        self.smoothed_states = []  # List to store smoothed states

    def dynamics(self, t, x):
        return self.f(t, x)

    def measurement(self, t, x):
        return self.h(t, x)

    def update_estimate(self, u, y):
        if self.x_hat is None:
            self.x_hat = np.zeros((len(y), 1))
            self.P = np.eye(len(y))

        # Extended Kalman Filter prediction step
        F = self.compute_jacobian(self.f, self.x_hat)
        self.x_hat += self.dt * (np.dot(F, self.x_hat) + self.f(0, self.x_hat, u)) + np.sqrt(self.dt) * np.random.multivariate_normal(np.zeros(len(self.x_hat)), self.Q).reshape(-1, 1)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        # Extended Kalman Filter correction step
        H = self.compute_jacobian(self.h, self.x_hat)
        K = np.dot(np.dot(self.P, H.T), inv(np.dot(np.dot(H, self.P), H.T) + self.R))
        self.x_hat += np.dot(K, (y - self.h(0, self.x_hat)))
        self.P = np.dot((np.eye(len(self.x_hat)) - np.dot(K, H)), self.P)

        return self.x_hat

    def compute_jacobian(self, func, x):
        n = len(x)
        m = len(func(0, x))
        J = np.zeros((m, n))

        for i in range(n):
            eps = 1e-6
            x_plus = np.copy(x)
            x_plus[i] += eps
            J[:, i] = (func(0, x_plus) - func(0, x)) / eps

        return J

    def rts_smoother(self):
        # Initialize RTS smoother
        self.smoothed_states = [self.x_hat]
        V = [self.P]

        # Backward pass
        n_steps = len(self.smoothed_states)
        for k in range(n_steps - 1, 0, -1):
            F = self.compute_jacobian(self.f, self.smoothed_states[k])
            Q_inv = inv(self.Q)
            V_prev = V[-1]
            V_pred = np.dot(np.dot(F.T, Q_inv), F) + inv(V_prev)
            L = np.dot(np.dot(V_prev, F.T), Q_inv) / V_pred

            self.smoothed_states.append(self.smoothed_states[k] + np.dot(L, (self.smoothed_states[k - 1] - np.dot(F, self.smoothed_states[k]))))
            V.append(self.P + np.dot(np.dot(L, V_prev - V_pred), L.T))

        # Reverse the list of smoothed states
        self.smoothed_states.reverse()

        return self.smoothed_states


# Example usage:
def dynamics(t, x, u):
    # Nonlinear system dynamics
    return np.array([[x[1]], [-0.1 * x[1] - np.sin(x[0])]])

def measurement(t, x):
    # Nonlinear measurement function
    return np.array([[x[0]]])

dt = 0.1  # Time step for numerical integration
Q = np.eye(2) * 1e-4  # Process noise covariance matrix
R = np.eye(1) * 1e-3  # Measurement noise covariance matrix

# Create the smoother object
smoother = EKFRS_Smoother(dynamics, measurement, Q, R, dt)

# Simulate the system and perform estimation
np.random.seed(0)  # For reproducibility
true_states = []
measurements = []
estimates = []

x = np.array([[0.5], [0.5]])  # Initial true state
for t in range(100):
    u = np.array([[0]])  # Input (not used in this example)
    y = measurement(0, x) + np.random.randn(1, 1) * np.sqrt(R[0, 0])  # True measurement with noise
    x = dynamics(0, x, u)  # True dynamics

    true_states.append(x)
    measurements.append(y)

    x_estimate = smoother.update_estimate(u, y)
    estimates.append(x_estimate)

# Perform RTS smoothing
smoothed_states = smoother.rts_smoother()

# Print results
print("True States:")
print(np.array(true_states).reshape(-1, 2))
print("\nMeasurements:")
print(np.array(measurements).reshape(-1, 1))
print("\nSmoothed Estimates:")
print(np.array(smoothed_states).reshape(-1, 2))
