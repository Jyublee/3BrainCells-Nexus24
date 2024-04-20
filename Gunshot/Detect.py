import pyaudio
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('best_weights.hdf5')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for gunshots...")

try:
    while True:
        data = stream.read(CHUNK)
        data_np = np.frombuffer(data, dtype=np.int16)
        
        # Preprocess the audio data (e.g., extract features)
        # This step depends on how the model was trained
        
        # Make a prediction using the model
        prediction = model.predict(preprocessed_data)
        
        if prediction > 0.5:
            print("Gunshot detected!")

except KeyboardInterrupt:
    pass

print("Stopping...")
stream.stop_stream()
stream.close()
p.terminate()