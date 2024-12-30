from difflib import SequenceMatcher
from melo.api import TTS

from System.models.whisper_model import WhisperModel
from System.models.correction_model import CorrectionModel
from System.handlers.text_handler import TextHandler
from System.handlers.audio_handler import AudioHandler
from System.handlers.file_handler import FileHandler

class System:
    """
    A Comprehensive System for Automatic Speech Recognition and Forced Alignment (ASR-FA).
    This class serves as the main interface for aligning spoken audio with expected textual content. It utilizes advanced
    speech-to-text transcription methods and text normalization techniques to compare transcribed and expected text,
    identifying discrepancies and storing results for further analysis.

    Features:
    - Audio transcription using state-of-the-art models like Whisper.
    - Text normalization and comparison for identifying mismatched words.
    - Error logging in CSV format for efficient debugging and review.
    - Interactive console-based interface for recording or loading audio.
    - Correction of transcribed text using predefined rules and models.
    - Synthesis of corrected text into speech using a TTS system.

    Dependencies:
    - WhisperModel: Handles speech-to-text transcription.
    - TextHandler: Provides text normalization and comparison utilities.
    - AudioHandler: Manages audio recording and selection.
    - FileHandler: Manages file operations, such as logging errors.
    - CorrectionModel: Applies error correction to transcribed text.
    - TTS: Converts corrected text into audio.

    Methods:
    - __init__: Initializes default configuration for the Speech Aligner system.
    - select_model: Selects and invokes the transcription model for audio processing.
    - process_audio: Normalizes and compares transcribed text against the expected text.
    - run_train_mode: Executes the main loop of the Speech Aligner system.
    - run_use_mode: Processes real-time audio input, applies corrections, and synthesizes speech output.
    - run_use_model_test: Allows testing of text correction rules and synthesis of corrected text.

    Example:
        speech_aligner = System()
        speech_aligner.run_train_mode(model_name="whisper", model_size="base", language="en", csv_path="errors.csv")
    """
    def __init__(self):
        """
        Initializes the Speech Aligner system (ASR-FA: Automatic Speech Recognition - Forced Alignment).
        """
        self.model_name = None
        self.model_size = None
        self.language = "en"
        self.csv_path = "errors_en.csv"
        self.expected_text = ""

    def select_model(self, audio_path: str):
        """
        Selects the model to be used for transcribing audio to text.

        Args:
            audio_path (str): Path to the audio file.
        """
        try:
            if self.model_name.lower() == "whisper":
                model = WhisperModel(self.model_size)
                transcribed_text = model.whisper_transcriber(audio_path, self.language)
                return self.process_audio(transcribed_text)
            else:
                raise ValueError("Unsupported model. Select 'Whisper...'")
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None

    def process_audio(self, transcribed_text: str):
        """
        Processes transcribed text comparing it to expected text.

        Args:
            transcribed_text (str): Transcribed text.

        Returns:
            str: Normalized transcribed text.
        """
        normalized_transcribed = TextHandler.text_normalize(transcribed_text)
        normalized_expected = TextHandler.text_normalize(self.expected_text)

        incorrect, correct = TextHandler.text_compare(normalized_transcribed, normalized_expected)
        FileHandler.file_errors_update(self.csv_path, zip(incorrect, correct))

        similarity = SequenceMatcher(None, normalized_expected, normalized_transcribed).ratio()
        print(f"Similarity of text: {similarity * 100:.2f}%")

        return normalized_transcribed

    def run_train_mode(self, model_name: str, model_size: str, language: str, csv_path: str):
        """
        Runs the main loop for the Speech Aligner system (ASR-FA: Automatic Speech Recognition - Forced Alignment).

        Args:
            model_name (str): The name of the model to use (e.g., "whisper").
            model_size (str): The size of the model to use (e.g., "turbo", "base").
            language (str): The language of the model to use (e.g., "en", "es", "ru").
            csv_path (str): CSV file path.
        """
        self.language = language
        self.model_name = model_name
        self.model_size = model_size
        self.csv_path = csv_path

        print("\nWelcome to the Speech Aligner System!")
        while True:
            print("1. Record audio from microphone.")
            print("2. Load existing audio file.")
            print("3. Exit.")
            choice = input("Choose an option: ").strip()

            if choice == "1":
                self.expected_text = input("Enter the expected text: ").strip()
                duration = int(input("Enter recording duration (seconds): "))
                audio_path = AudioHandler.audio_record(duration)
                try:
                    transcribed_text = self.select_model(audio_path)
                    print(f"Transcribed: {transcribed_text}")
                finally:
                    AudioHandler.audio_remove(audio_path)

            elif choice == "2":
                self.expected_text = input("Enter the expected text: ").strip()
                audio_path = AudioHandler.audio_select()

                if audio_path:
                    transcribed_text = self.select_model(audio_path)
                    print(f"Transcribed: {transcribed_text}")

            elif choice == "3":
                print("Saving data and exiting from Speech Aligner System... Goodbye!")
                break

            else:
                print("Invalid choice. Try again...")

    def run_use_mode(self, model_name: str, model_size: str, language: str, model_path: str, method: str):
        """
        Processes real-time audio input, applies corrections, and synthesizes speech output.

        Args:
            model_name (str): Name of the transcription model to use.
            model_size (str): Size of the transcription model.
            language (str): Language of the transcription model.
            model_path (str): Path to the correction model.
            method (str): Correction method to use.
        """
        self.language = language
        self.model_name = model_name
        self.model_size = model_size

        replacement_rules = FileHandler.file_model_load(model_path)
        if replacement_rules is None:
            print("Model is missing...")
            exit()

        print("\nWelcome to the Speech Aligner System!")
        while True:
            duration = int(input("Enter recording duration (seconds) or \"exit\" to exit from system: "))
            if duration == "exit":
                break

            audio_path = AudioHandler.audio_record(duration)
            try:
                transcribed_text = self.select_model(audio_path)
                print(f"Transcribed text: {transcribed_text}")

                corrected_sentence = CorrectionModel.correction_start(transcribed_text, replacement_rules, method, language)
                print("Corrected text:", corrected_sentence)

                model = TTS(language='EN', device='cuda')  # Use 'cuda', if you have GPU
                model.tts_to_file(corrected_sentence, speaker_id=5, output_path="output.wav")
            finally:
                AudioHandler.audio_remove(audio_path)

    def run_use_model_test(self, language: str, model_path: str, method: str):
        """
        Allows testing of text correction rules and synthesis of corrected text.

        Args:
            language (str): Language for the correction model.
            model_path (str): Path to the correction model.
            method (str): Correction method to use.
        """
        self.language = language

        replacement_rules = FileHandler.file_model_load(model_path)
        if replacement_rules is None:
            print("Model is missing...")
            exit()

        print("\nWelcome to the Speech Aligner System!")
        while True:
            test_sentence = input("Enter a error suggestion for correction or \"exit\" to exit from system: ").strip()
            if test_sentence == "exit":
                break

            corrected_sentence = CorrectionModel.correction_start(test_sentence, replacement_rules, method, language)
            print("Corrected sentence: ", corrected_sentence)

            model = TTS(language='ES', device='cuda')  # Use 'cuda', if you have GPU
            model.tts_to_file(corrected_sentence, speaker_id=5, output_path="output.wav")

