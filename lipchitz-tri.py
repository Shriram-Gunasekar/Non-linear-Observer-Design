class LipschitzTriangular:
    def __init__(self, L, c, d):
        self.L = L
        self.c = c
        self.d = d

    def evaluate(self, x):
        if x < self.c - self.d:
            return 0.0
        elif x <= self.c:
            return (self.L / self.d) * (x - self.c + self.d)
        elif x <= self.c + self.d:
            return (-self.L / self.d) * (x - self.c + self.d)
        else:
            return 0.0

# Example usage:
L = 2  # Lipschitz constant
c = 3  # Center of the triangular form
d = 1  # Half-width of the triangular form
triangular_form = LipschitzTriangular(L, c, d)

# Evaluate the function at different points
print(triangular_form.evaluate(2.5))  # Output: 1.0
print(triangular_form.evaluate(3.5))  # Output: -1.0
print(triangular_form.evaluate(4.5))  # Output: 0.0
