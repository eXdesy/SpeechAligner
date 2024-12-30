import sounddevice as sd
import wave
import tempfile
import os
from tkinter import Tk, filedialog

class AudioHandler:
    """
    A utility class for handling basic audio operations such as recording,
    deleting, and selecting audio files.

    This class provides static methods to:
    - Record audio using the system's microphone.
    - Delete a specified audio file.
    - Open a file dialog to select an audio file.

    Methods:
        audio_record(duration: int = 5, sample_rate: int = 16000) -> str:
            Records audio from the microphone and saves it to a temporary file.
        audio_remove(file_path: str):
            Deletes the specified audio file if it exists.
        audio_select() -> str:
            Opens a file dialog to select an audio file and returns the file path.
    """
    @staticmethod
    def audio_record(duration: int = 10, sample_rate: int = 16000) -> str:
        """
        Records audio from the microphone.

        Args:
            duration (int): Duration of the recording in seconds.
            sample_rate (int): Sample rate of the audio.

        Returns:
            str: Path to the saved audio file.
        """
        print(f"Recording {duration} seconds of audio...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"Audio recorded...")
        return temp_file.name

    @staticmethod
    def audio_remove(file_path: str):
        """
        Deletes the specified audio file.

        Args:
            file_path (str): Path to the audio file.
        """
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def audio_select() -> str:
        """
        Opens a file dialog to select an audio file.

        Returns:
            str: Path to the selected audio file or an error message if no file is selected.
        """
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        audio_path = filedialog.askopenfilename(title="Select an audio file",
                                                filetypes=[("Audio files", "*.wav;*.mp3")])

        if audio_path:
            return audio_path
        else:
            return "Error: No file selected or this files doesnt exist"
