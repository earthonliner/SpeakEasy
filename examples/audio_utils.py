import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
import time

recording_started = False
recording_stopped = False
tick_count = 0
recording_buffer = []
start_time = 0

def on_press(key):
    global recording_started, recording_stopped, recording_buffer, start_time, tick_count

    if key == keyboard.Key.space and tick_count == 0:
        recording_started = True
        start_time = time.time()
        tick_count += 1
    elif key == keyboard.Key.space and tick_count == 1:
        recording_stopped = True
        return False

def record_and_save_audio(freq=16000, max_recording_time=120, save_path='tmp.wav', channels=1):
    global recording_started, recording_stopped, recording_buffer, start_time, tick_count
    # Start listener for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        print("Press 'Space' to start recording and press it again to stop.")
        while not recording_started:
            time.sleep(0.1)
        with sd.InputStream(samplerate=freq, channels=channels) as stream:
            while not recording_stopped:
                audio_chunk = stream.read(int(freq))[0]
                if recording_started:
                    recording_buffer.extend(audio_chunk)
                if time.time() - start_time >= max_recording_time:
                    recording_stopped = True
                    recording_buffer = []

            recording_array = np.array(recording_buffer, dtype=np.float32)
            recording_array = (recording_array * 32767).astype(np.int16)
            write(save_path, freq, recording_array)

            recording_started, recording_stopped = False, False
            audio_chunk, recording_buffer = [], []
            tick_count = 0
            listener.stop()
            stream.abort()
            stream.close()


if __name__ == "__main__":
    record_and_save_audio()

