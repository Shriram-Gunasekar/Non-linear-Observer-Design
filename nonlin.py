import numpy as np
from scipy.integrate import solve_ivp

class NonlinearObserver:
    def __init__(self, f, h, dt, x0, P0, Q, R):
        self.f = f  # System dynamics function
        self.h = h  # Measurement function
        self.dt = dt  # Time step for numerical integration
        self.x_hat = x0  # Initial state estimate
        self.P = P0  # Initial error covariance matrix
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Measurement noise covariance matrix

    def dynamics(self, t, x):
        return self.f(t, x)

    def measurement(self, t, x):
        return self.h(t, x)

    def update_estimate(self, u, y):
        # Numerically integrate the dynamics using solve_ivp
        t_span = (0, self.dt)
        sol = solve_ivp(self.dynamics, t_span, self.x_hat, method='RK45')
        x_next = sol.y[:, -1].reshape(-1, 1)

        # Predict step (EKF prediction)
        F = self.compute_jacobian(self.f, x_next)
        self.x_hat = x_next
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        # Correct step (EKF update)
        H = self.compute_jacobian(self.h, x_next)
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(np.dot(np.dot(H, self.P), H.T) + self.R))
        self.x_hat += np.dot(K, (y - self.measurement(0, self.x_hat)))
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

# Example usage:
def dynamics(t, x):
    return np.array([[x[1]], [-0.1 * x[1] - np.sin(x[0])]])

def measurement(t, x):
    return np.array([[x[0]]])

dt = 0.01  # Time step for numerical integration
x0 = np.array([[0.5], [0.5]])  # Initial state estimate
P0 = np.eye(2) * 0.1  # Initial error covariance matrix
Q = np.eye(2) * 1e-4  # Process noise covariance matrix
R = np.eye(1) * 1e-3  # Measurement noise covariance matrix

observer = NonlinearObserver(dynamics, measurement, dt, x0, P0, Q, R)

# Simulate the system and observer
for t in range(1000):
    u = np.array([[0]])  # Input (not used in this example)
    y = np.array([[np.sin(0.1 * t) + np.random.randn() * 0.1]])  # Measurement with noise
    x_estimate = observer.update_estimate(u, y)
    print(f"Time step {t}: Estimated state = {x_estimate.flatten()}")

