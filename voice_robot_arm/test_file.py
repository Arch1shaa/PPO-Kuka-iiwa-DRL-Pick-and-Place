from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",
    device="cuda",
    compute_type="float16"
)

print("Loaded!")