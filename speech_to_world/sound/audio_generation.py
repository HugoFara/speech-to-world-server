"""
Generates an ambient audio from a text prompt.

* For ambient audio:
  * https://huggingface.co/declare-lab/tango2 : apparently a good model but difficult to integrate
  * https://huggingface.co/facebook/audiogen-medium : less good but sufficient model
* Music : https://huggingface.co/facebook/musicgen-small
* Text-to-speech : https://huggingface.co/suno/bark
"""

from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


def ambient_audio(descriptions, duration=10):
    """
    Generate audio samples based on descriptions provided.

    :param list[str] descriptions: Description of the audio.
    :param int duration: The duration of the audio.
    :return tuple[torch.Tensor, int]: WAVE audio samples and sample rate.
    """
    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(duration=duration)
    wav = model.generate(descriptions)

    return wav, model.sample_rate


def ambient_music(descriptions, duration=30):
    """
    Generate musics based on the descriptions provided.

    :param list[str] descriptions: Description of the audio.
    :param int duration: The duration of the audio.
    :return tuple[torch.Tensor, int]: WAVE audio samples and sample rate.
    """
    model = AudioGen.get_pretrained("facebook/musicgen-medium")
    model.set_generation_params(duration=duration)
    wav = model.generate(descriptions)

    return wav, model.sample_rate


def generate_audio(descriptions, duration=10):
    """
    Generates audio samples based on descriptions provided and saves them as .wav files.

    :param list[str] descriptions: Description of the audio.
    :param int duration: The duration of the audio in seconds. Default is 10 seconds.
    """
    wav_data, sample_rate = ambient_audio(descriptions, duration)
    for idx, one_wav in enumerate(wav_data):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(
            f"outputs/audio_{idx}.wav",
            one_wav.cpu(),
            sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )


def generate_music(descriptions, duration=30):
    """
    Generate music based on the descriptions provided.

    :param list[str] descriptions: Description of the audio.
    :param int duration: The duration of the audio. Default is 30 seconds.
    """
    wav_data, sample_rate = ambient_music(descriptions, duration)
    for idx, one_wav in enumerate(wav_data):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(
            f"outputs/music_{idx}.wav",
            one_wav.cpu(),
            sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )


if __name__ == "__main__":
    generate_audio(["Seagulls crying", "Waves crashing", "Water lapping at the shore"])
    # generate_music(["Calm and relaxing music"])
