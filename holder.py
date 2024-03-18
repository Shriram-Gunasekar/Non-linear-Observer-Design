class HolderTriangular:
    def __init__(self, C, alpha, c, d):
        self.C = C  # Holder constant
        self.alpha = alpha  # Holder exponent
        self.c = c  # Center of the triangular form
        self.d = d  # Half-width of the triangular form

    def evaluate(self, x):
        if abs(x - self.c) <= self.d:
            return self.C * abs(x - self.c)**self.alpha
        else:
            return 0.0

# Example usage:
C = 1.0  # Holder constant
alpha = 0.5  # Holder exponent
c = 3  # Center of the triangular form
d = 1  # Half-width of the triangular form
holder_triangular = HolderTriangular(C, alpha, c, d)

# Evaluate the function at different points
x_values = [2.5, 3.5, 4.5]
for x in x_values:
    result = holder_triangular.evaluate(x)
    print(f"Value at {x}: {result}")
