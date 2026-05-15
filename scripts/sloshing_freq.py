import numpy as np


def get_freq(n: int, h: float = 0.09, L: float = 0.6) -> float:
    return np.sqrt(9.81 * (np.pi * n) / L * np.tanh(np.pi * n * h / L)) / (2 * np.pi)


print(f"n = 1: {get_freq(1)}")
print(f"n = 2: {get_freq(2)}")
print(f"n = 3: {get_freq(3)}")
print(f"n = 4: {get_freq(4)}")
print(f"n = 5: {get_freq(5)}")
