import whisper
import torch
import os

class WhisperModel:
    """
    A utility class for transcribing audio using the Whisper model.

    This class initializes a Whisper model of the specified size and provides
    a method to transcribe audio files into text.

    Attributes:
        model (whisper.Whisper): The loaded Whisper model.
        device (str): The device on which the model will run_train_mode ("cuda" or "cpu").

    Methods:
        whisper_transcriber(audio_path, language) -> str:
            Transcribes an audio file into text using the Whisper model.
    """
    def __init__(self, model_size: str = "turbo", device: str = None):
        """
        Initializes the Whisper model.

        Args:
            model_size (str): The size of the Whisper model to file_model_load (e.g., "base", "large").
            device (str, optional): The device to run_train_mode the model on (e.g., "cuda", "cpu"). Defaults to automatic detection.
        """
        self.model = whisper.load_model(model_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def whisper_transcriber(self, audio_path: str, language: str) -> str:
        """
        Transcribes the audio file into text.

        Args:
            audio_path (str): Path to the audio file.
            language (str): Language of the audio content.

        Returns:
            str: Transcribed text.
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            temperature=0.0,  # Avoid guessing
            without_timestamps=True
        )
        return result["text"].strip()