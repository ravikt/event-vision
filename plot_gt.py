import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('ev_20251011_105115.csv')

# Ensure the 'ch' column is treated as integer
df['ch'] = df['ch'].astype(int)

# Convert RPM to blade-pass frequency in Hz (2 blades → 2 pulses per revolution)
df['freq_hz'] = df['rpm'] / 60.0 * 2

# Get unique channels (should be 0, 1, 2, 3)
channels = sorted(df['ch'].unique())

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, ch in enumerate(channels):
    freq_vals = df[df['ch'] == ch]['freq_hz'].reset_index(drop=True)
    axes[i].plot(freq_vals, label=f'Channel {ch} Frequency')
    axes[i].set_title(f'Channel {ch}')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].grid(True)

plt.tight_layout()
plt.show()