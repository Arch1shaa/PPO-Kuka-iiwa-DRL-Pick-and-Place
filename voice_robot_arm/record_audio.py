import sounddevice as sd
import soundfile as sf
import numpy as np

duration = 4
samplerate = 16000

print("Speak now...")

audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype="float32",
    device=14
)

sd.wait()

print("Max amplitude:", np.max(np.abs(audio)))

sf.write(
    "command.wav",
    audio,
    samplerate
)

print("Saved.")