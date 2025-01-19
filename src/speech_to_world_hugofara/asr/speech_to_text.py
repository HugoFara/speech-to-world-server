"""
A simple Speech-to-Text module.

It uses whisper by OpenAI, source https://huggingface.co/openai/whisper-large-v3
"""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

ASR_MODEL_ID = "openai/whisper-large-v3"


def get_asr_model():
    """Load the model from Hugging Face."""
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )


def do_audio_transcription(audio):
    """
    Return the text from an audio file.

    :param audio: Input audio, either a file path or bytes
    :type audio: str | bytes[]
    :return str: Text in the audio
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = get_asr_model().to(device)

    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"task": "translate"}
    )
    return pipe(audio)
