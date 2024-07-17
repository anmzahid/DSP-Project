import pyaudio
import wave
from pydub import AudioSegment

def record_audio(output_file, record_seconds=30, sample_rate=44100, chunk_size=2048):
    audio_format = pyaudio.paInt16  # 16-bit resolution
    channels = 2  # Stereo

    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=audio_format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)
    print(f"Recording for {record_seconds} seconds...")

    frames = []

    for _ in range(int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    # Convert to mono
    sound = AudioSegment.from_wav(output_file)
    sound = sound.set_channels(1)
    sound.export(output_file, format="wav")
