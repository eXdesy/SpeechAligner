import string
import unicodedata

class TextHandler:
    """
    A utility class for handling text processing tasks, including normalization and comparison.

    This class provides methods to:
    - Normalize text by removing punctuation, converting to lowercase, and standardizing characters.
    - Compare transcribed text with expected text to identify mismatches.

    Methods:
        text_normalize(text: str) -> str:
            Normalizes input text for consistent processing.
        text_compare(transcribed: str, expected: str) -> tuple:
            Compares transcribed and expected text, identifying mismatched words.
    """
    @staticmethod
    def text_normalize(text: str) -> str:
        """
        Normalizes text by removing punctuation, converting to lowercase,
        and standardizing characters.

        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        translator = str.maketrans('', '', string.punctuation)
        normalized = text.translate(translator).lower()
        return unicodedata.normalize('NFD', normalized).encode('ascii', 'ignore').decode('utf-8')

    @staticmethod
    def text_compare(transcribed: str, expected: str):
        """
        Compares transcribed text with expected text to identify mismatches or errors.

        Args:
            transcribed (str): Transcribed text.
            expected (str): Expected text.

        Returns:
            tuple: Incorrect and correct word lists.
        """
        transcribed_words = transcribed.split()
        expected_words = expected.split()
        incorrect = [t for t, e in zip(transcribed_words, expected_words) if t != e]
        correct = [e for t, e in zip(transcribed_words, expected_words) if t != e]
        return incorrect, correct
