from faster_whisper import WhisperModel
from voice_robot_arm.parser import parse_command

import sounddevice as sd
import soundfile as sf

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

def get_voice_command():

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

    sf.write(
        "command.wav",
        audio,
        samplerate
    )

    segments, info = model.transcribe("command.wav")

    text = ""

    for segment in segments:
        text += segment.text

    print("Speech:", text)

    color, shape = parse_command(text)

    return color