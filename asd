import numpy as np
import matplotlib.pyplot as plt
import math

class Curve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
    
    def evaluate(self, t):
        pass

class BezierCurve(Curve):
    def evaluate(self, t):
        n = len(self.control_points) - 1
        result = np.zeros(self.control_points[0].shape)
        for i in range(n + 1):
            result += math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * self.control_points[i]
        return result

class BSplineCurve(Curve):
    def evaluate(self, t):
        n = len(self.control_points) - 1
        degree = min(3, n)
        result = np.zeros(self.control_points[0].shape)
        for i in range(n - degree + 1):
            result += self.control_points[i + degree] * self.bspline_basis(i, degree, t)
        return result

    def bspline_basis(self, i, k, t):
        if k == 0:
            return 1 if self.control_points[i] <= t < self.control_points[i + 1] else 0
        else:
            denominator1 = self.control_points[i + k] - self.control_points[i]
            term1 = 0 if denominator1 == 0 else ((t - self.control_points[i]) / denominator1) * self.bspline_basis(i, k - 1, t)

            denominator2 = self.control_points[i + k + 1] - self.control_points[i + 1]
            term2 = 0 if denominator2 == 0 else ((self.control_points[i + k + 1] - t) / denominator2) * self.bspline_basis(i + 1, k - 1, t)
            
            return term1 + term2

# Puntos de control para la curva de Bezier (máximo 4 puntos)
bezier_control_points = [[0, 0], [1, 3], [2, 0]]

# Puntos de control para la curva B-spline (máximo 4 puntos)
bspline_control_points = [[0, 0], [1, 2], [2, 2], [3, 0]]

# Valores de t para evaluar la curva
t_values = np.linspace(0, 1, 100)

# Evaluar y graficar la curva de Bezier
bezier_curve = BezierCurve(bezier_control_points)
bezier_points = np.array([bezier_curve.evaluate(t) for t in t_values])

# Evaluar y graficar la curva B-spline
bspline_curve = BSplineCurve(bspline_control_points)
bspline_points = np.array([bspline_curve.evaluate(t) for t in t_values])

# Graficar las curvas
plt.figure(figsize=(10, 5))
plt.plot(bezier_points[:, 0], bezier_points[:, 1], label="Curva de Bezier", color="blue")
plt.plot(bspline_points[:, 0], bspline_points[:, 1], label="Curva B-spline", color="red")
plt.scatter(np.array(bezier_control_points)[:, 0], np.array(bezier_control_points)[:, 1], label="Puntos de control (Bezier)", color="blue", marker="o")
plt.scatter(np.array(bspline_control_points)[:, 0], np.array(bspline_control_points)[:, 1], label="Puntos de control (B-spline)", color="red", marker="o")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Curvas de Bézier y B-spline")
plt.grid(True)
plt.show()
