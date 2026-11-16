import numpy as np
from scipy.io import wavfile

sr, dur = 44100, 60
t = np.arange(sr * dur) / sr
rng = np.random.default_rng(0)

tone = 0.3 * np.sin(2 * np.pi * 440 * t)
chirp = 0.2 * np.sin(2 * np.pi * (200 * t + 150 * t**2 / dur))
clean = (tone + chirp) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
clean[:sr] = 0.0

noisy = clean + 0.05 * rng.standard_normal(len(t))


def save(name, x):
    wavfile.write(name, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


save("clean.wav", clean)
save("noisy.wav", noisy)
