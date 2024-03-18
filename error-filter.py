import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Generate some noisy data
np.random.seed(0)
time = np.linspace(0, 10, 100)
signal = np.sin(time) + np.random.normal(0, 0.2, 100)

# Apply the Savitzky-Golay filter
smoothed_signal = savgol_filter(signal, window_length=11, polyorder=2)

# Plot the original signal and the smoothed signal
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time, smoothed_signal, label='Smoothed Signal (Savitzky-Golay)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('Savitzky-Golay Filter Example')
plt.show()
