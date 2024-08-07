import sounddevice as sd
import numpy as np
import whisper
import keyboard

# Load the Whisper model
model = whisper.load_model("base")

# Define parameters
sampling_rate = 16000  # Sample rate for recording
chunk_duration = 5  # Duration of each audio chunk in seconds
chunk_size = int(sampling_rate * chunk_duration)  # Number of samples in each chunk

# Buffer to store recorded audio
audio_buffer = []

# Callback function to process audio chunks
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.append(indata[:, 0].copy())

# Function to stop recording on key press
def stop_recording():
    print("Press 'q' to stop the recording...")
    while True:
        if keyboard.is_pressed('q'):
            print("Recording stopped.")
            return

# Start recording audio
print("Recording...")
with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32', callback=callback, blocksize=chunk_size):
    stop_recording()

# Convert list to numpy array
audio_data = np.concatenate(audio_buffer)
print(f"Recorded audio length: {len(audio_data)} samples")

# Prepare audio for Whisper
print("Processing audio...")
audio_data = whisper.pad_or_trim(audio_data)  # Pad or trim the audio to fit model requirements
mel = whisper.log_mel_spectrogram(audio_data).to(model.device)  # Convert audio to Mel spectrogram

# Detect language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode audio
print("Transcribing audio...")
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# Print the recognized text
print("Transcribed Text:", result.text)
