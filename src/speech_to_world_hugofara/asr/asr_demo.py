"""
Demo file for an Automatic Speech Recognition system.
"""
import wave

from datasets import load_dataset
import pyaudio

from .speech_to_text import do_audio_transcription


def register_audio():
    """Register audio from the user's microphone."""
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    rate = 44100
    record_seconds = 10
    output_filename = "output.mp3"

    p = pyaudio.PyAudio()

    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("* recording")

    frames = []

    for _ in range(int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()
    return output_filename


def main_demo():
    """
    Print the user audio or a default sample.

    If the user chooses to enter their own audio, it calls the `register_audio` function to record
    audio and then uses the `sample_to_text` function to convert the audio to text.

    If the user chooses not to enter their own audio,
    it uses a default sample from the 'distil-whisper/librispeech_long' dataset.
    """
    if input("Would you like to enter your own audio (y/[N])? ") == "y":
        print("Please describe what you would like to see.")
        sample = register_audio()
    else:
        print("Using default sample")
        dataset = load_dataset(
            "distil-whisper/librispeech_long", "clean", split="validation"
        )
        sample = dataset[0]["audio"]

    result = do_audio_transcription(sample)
    print(result["text"])


if __name__ == "__main__":
    main_demo()
