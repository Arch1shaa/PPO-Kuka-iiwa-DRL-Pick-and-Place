from faster_whisper import WhisperModel

print("Loading model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Transcribing...")

segments, info = model.transcribe("command.wav")

for segment in segments:
    print(segment.text)