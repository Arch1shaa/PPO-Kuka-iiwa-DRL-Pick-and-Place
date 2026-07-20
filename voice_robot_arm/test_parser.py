from faster_whisper import WhisperModel
from parser import parse_command

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("command.wav")

text = ""

for segment in segments:
    text += segment.text

print("Speech:", text)

color, shape = parse_command(text)

print("Color:", color)
print("Shape:", shape)