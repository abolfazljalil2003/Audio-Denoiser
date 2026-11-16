import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sr = 44100


def load(name):
    return wavfile.read(name)[1].astype(np.float64) / 32768


c, n, d = load("clean.wav"), load("noisy.wav"), load("denoised.wav")


def snr(ref, x):
    ref, x = ref[sr:], x[sr:]
    return 10 * np.log10(np.sum(ref**2) / np.sum((x - ref) ** 2))


print(f"SNR before: {snr(c, n):.2f} dB")
print(f"SNR after : {snr(c, d):.2f} dB")

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, x, title in zip(ax, [c, n, d], ["Clean", "Noisy", "Denoised"]):
    a.specgram(x[: 10 * sr] + 1e-10, NFFT=1024, Fs=sr, noverlap=512, vmin=-140, vmax=-40)
    a.set_ylim(0, 4000)
    a.set_title(title)
    a.set_xlabel("Time (s)")
ax[0].set_ylabel("Frequency (Hz)")
plt.savefig("spectrogram.png", dpi=150, bbox_inches="tight")

