from openai import OpenAI
from itertools import combinations, product
from metaphone import doublemetaphone
from collections import Counter
import nltk
from nltk.corpus import brown, cess_esp, udhr # words
from nltk.metrics.distance import edit_distance
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class CorrectionModel:
    """
    This class provides advanced spelling and grammar correction functionalities using various algorithms and external APIs.
    It includes methods for correction based on substitution rules, Levenshtein distance, phonetic similarity, and
    context-aware approaches. The class leverages resources such as the NLTK library, the Brown corpus, and OpenAI's GPT API
    to deliver accurate and context-sensitive corrections.

    Key Features:
    - Substitution rule-based corrections.
    - Integration with OpenAI GPT API for grammar and spelling enhancements.
    - Use of Levenshtein distance and phonetic similarity for word corrections.
    - Handling of common typographical errors (e.g., missing or extra letters, adjacent letter swaps).
    - Dictionary-based validation using the NLTK Brown corpus.

    Attributes:
        word_freqs (Counter): A frequency counter of words in the Brown corpus, used for word frequency analysis.
        dictionaries (set): A set of valid dictionary words derived from the Brown corpus with frequency > 1 and length > 1.

    Methods:
        correction_start(sentence, rules, model):
            Corrects a given sentence using substitution rules and an optional model.

        correction_gpt(sentence, api_key):
            Corrects the grammar and spelling of a given sentence using OpenAI's GPT API.

        correction_levenshtein(word):
            Finds the most probable correction using Levenshtein distance.

        correction_phonetic(word):
            Finds the most probable correction using phonetic similarity.

        correction_combined(word):
            Combines Levenshtein and phonetic corrections to find the best match.

        correction_matches(possible_corrections, word):
            Selects the best match from possible corrections based on dictionary validation and word frequency.

        correction_insert_missing_letter(word):
            Generates possible corrections by inserting missing letters into a word.

        correction_remove_extra_letter(word):
            Generates possible corrections by removing extra letters from a word.

        correction_swap_adjacent_letter(word):
            Generates possible corrections by swapping adjacent letters in a word.

        correction_pronoun_or_possessive(word):
            Determines if a word is a pronoun or possessive pronoun.

        correction_generate(word, word_length, rules, possible_corrections):
            Generates all possible correction variants of a word based on given rules.

        correction_sentence(word, rules):
            Corrects a given word based on a set of transformation rules and dictionary validation.
    """
    try:
        nltk.data.find('corpora/words.zip')
        nltk.data.find('corpora/brown.zip')
        nltk.data.find('corpora/udhr.zip')
        nltk.data.find('corpora/cess_esp.zip')
        nltk.data.find('tokenizers/punkt_tab.zip')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng.zip')
    except LookupError:
        nltk.download('cess_esp')
        nltk.download('udhr')
        nltk.download('words')
        nltk.download('brown')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

    word_freqs = {
        "en": Counter(w.lower() for w in brown.words() if w.isalpha()), # word_freq = Counter(brown.words())
        "es": Counter(w.lower() for w in cess_esp.words() if w.isalpha()),
        "ru": Counter(w.lower() for w in udhr.words('Russian-Cyrillic') if w.isalpha()),
    }

    dictionaries = {
        "en": {word for word, freq in word_freqs["en"].items() if freq > 1 and len(word) > 1}, # dictionary = set(words.words())
        "es": {word for word, freq in word_freqs["es"].items() if freq > 1 and len(word) > 1},
        "ru": {word for word, freq in word_freqs["ru"].items() if freq > 1 and len(word) > 1},
    }

    @staticmethod
    def correction_start(sentence, rules, model, language):
        """
        Corrects a given sentence using substitution rules and an optional model.

        Args:
            sentence (str): The sentence to be corrected.
            rules (dict): A dictionary of substitution rules for character corrections.
            model (str): The correction model to use. Can be "gpt" or any other for default correction.
            language (str): Language for correction (en, es, ru).

        Returns:
            str: The corrected sentence.
        """
        if language not in CorrectionModel.dictionaries:
            raise ValueError(f"Language '{language}' is not supported...")

        word_freq = CorrectionModel.word_freqs[language]
        dictionary = CorrectionModel.dictionaries[language]

        print(f"Start of sentence correction: {sentence}")
        word = word_tokenize(sentence)
        tagged = pos_tag(word)

        corrected_words = []
        for word, tag in tagged:
            corrected_word = CorrectionModel.correction_sentence(word, rules, dictionary, word_freq)
            corrected_words.append(corrected_word)

        correct_sentence = ' '.join(corrected_words)

        if model == "gpt":
            key = "sk-proj-sV-TBJnmbemdbeKqbjYeqpXqCQOqsKJC99qPEJfTR2FjLjMaZunqYe7YY-FXo_xBsod5-PinbT3BlbkFJIyJixWGUikAatS4teAgNc_o4xnqLQb-JHLboPRDFAAGRMSZ4EppfkqL-mi1FyfdyHDOOfI4k8A"
            return CorrectionModel.correction_gpt(correct_sentence, key)
        else:
            return correct_sentence

    @staticmethod
    def correction_gpt(sentence, api_key):
        """
        Corrects the grammar and spelling of a given sentence using OpenAI's GPT API. This method uses OpenAI's
        `text-davinci-003` model to perform grammar and spelling correction.

        Args:
            sentence (str): The input sentence to be corrected.
            api_key (str): The API key required to authenticate with OpenAI's API.

        Returns:
            str: The corrected sentence, as returned by the OpenAI API. If an error occurs,
            the original sentence is returned.
        """
        client = OpenAI(api_key=api_key)
        try:
            response = client.completions.create(
                model="text-davinci-003",
                prompt=f"Correct the grammar and spelling of this sentence: '{sentence}'",
                max_tokens=100,
                temperature=0.7
            )
            corrected_sentence = response.choices[0].text.strip()
            print(f"Corrected proposal via GPT: {corrected_sentence}")
            return corrected_sentence
        except Exception as e:
            print(f"Error while accessing OpenAI API: {e}")
            return sentence

    @staticmethod
    def correction_levenshtein(word, dictionary, word_freq):
        """
        Finds the most probable correction using Levenshtein distance.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.
            word_freq (dict of str, int): A dictionary where keys are words and
                values are their corresponding frequency scores.

        Returns:
            str or None: The corrected word, or None if no correction is found.
        """
        candidates_levenshtein = [
            (w, edit_distance(word, w), word_freq[w])
            for w in dictionary if abs(len(w) - len(word)) <= 2
        ]
        candidates_levenshtein = sorted(candidates_levenshtein, key=lambda x: (x[1], -x[2]))
        if candidates_levenshtein:
            return candidates_levenshtein[0][0]
        return None

    @staticmethod
    def correction_phonetic(word, dictionary):
        """
        Finds the most probable correction using phonetic similarity.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.

        Returns:
            str or None: The corrected word, or None if no correction is found.
        """
        metaphone_word = doublemetaphone(word)[0]
        phonetic_candidates = [
            w for w in dictionary if doublemetaphone(w)[0] == metaphone_word
        ]
        if phonetic_candidates:
            return phonetic_candidates[0]
        return None

    @staticmethod
    def correction_combined(word, dictionary, word_freq):
        """
        Combines Levenshtein and phonetic corrections to find the best match.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.
            word_freq (dict of str, int): A dictionary where keys are words and
                values are their corresponding frequency scores.

        Returns:
            str or None: The best corrected word, or None if no correction is found.
        """
        candidates = []
        corrected_by_levenshtein = CorrectionModel.correction_levenshtein(word, dictionary, word_freq)
        corrected_by_phonetics = CorrectionModel.correction_phonetic(word, dictionary)

        if corrected_by_levenshtein:
            candidates.append((
                corrected_by_levenshtein,
                edit_distance(word, corrected_by_levenshtein),
                word_freq[corrected_by_levenshtein]
            ))

        if corrected_by_phonetics:
            candidates.append((
                corrected_by_phonetics,
                edit_distance(word, corrected_by_phonetics),
                word_freq[corrected_by_phonetics]
            ))

        if candidates:
            best_candidate = sorted(candidates, key=lambda x: x[1])[0][0]
            return best_candidate

        return None

    @staticmethod
    def correction_matches(possible_corrections, word, dictionary, word_freq):
        """
        Identifies the best correction for a given word from possible candidates.

        This method evaluates a list of possible corrections, filters them by
        checking against a provided dictionary, and selects the best match based
        on word frequency and context.

        Args:
            possible_corrections (list of str): List of potential word corrections.
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.
            word_freq (dict of str, int): A dictionary where keys are words and
                values are their corresponding frequency scores.

        Returns:
            str: The best-corrected word based on dictionary matches, frequency,
            and contextual relevance. Returns None if no correction can be found.
        """
        matches = [w for w in possible_corrections if w in dictionary]
        if matches:
            print(f"All matches in the dictionary: {matches}")
            best_match = sorted(matches, key=lambda w: (word_freq[w], -len(w)), reverse=True)[0]
            corrected_word = CorrectionModel.correction_combined(best_match,dictionary, word_freq)
            if corrected_word:
                print(f"The best candidate given the context: {corrected_word}")
                return corrected_word
            return best_match
        else:
            print(f"No matches found, only combined algorithm used...")
            corrected_word = CorrectionModel.correction_combined(word, dictionary, word_freq)
            if corrected_word:
                print(f"Best candidate by combined algorithm: {corrected_word}")
                return corrected_word

    @staticmethod
    def correction_insert_missing_letter(word, dictionary):
        """
        Generates possible corrections by inserting missing letters into a word.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.

        Returns:
            set: A set of corrected words found in the dictionary.
        """
        corrections = set()
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for i in range(len(word) + 1):
            for char in alphabet:
                temp = word[:i] + char + word[i:]
                if temp in dictionary:
                    corrections.add(temp)
        return corrections

    @staticmethod
    def correction_remove_extra_letter(word, dictionary):
        """
        Generates possible corrections by removing extra letters from a word.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.

        Returns:
            set: A set of corrected words found in the dictionary.
        """
        corrections = set()
        for i in range(len(word)):
            temp = word[:i] + word[i + 1:]
            if temp in dictionary:
                corrections.add(temp)
        return corrections

    @staticmethod
    def correction_swap_adjacent_letter(word, dictionary):
        """
        Generates possible corrections by swapping adjacent letters in a word.

        Args:
            word (str): The word to correct.
            dictionary (set of str): A set of valid words forming the dictionary.

        Returns:
            set: A set of corrected words found in the dictionary.
        """
        corrections = set()
        for i in range(len(word) - 1):
            temp = list(word)
            temp[i], temp[i + 1] = temp[i + 1], temp[i]
            candidate = ''.join(temp)
            if candidate in dictionary:
                corrections.add(candidate)
        return corrections

    @staticmethod
    def correction_pronoun_or_possessive(word):
        """
        Determines if a word is a pronoun or possessive pronoun.

        Args:
            word (str): The word to analyze.

        Returns:
            bool: True if the word is a pronoun or possessive pronoun, False otherwise.
        """
        pos = pos_tag(word_tokenize(word))
        return pos[0][1] in {'PRP', 'PRP$', 'POS'}

    @staticmethod
    def correction_generate(word, word_length, rules, possible_corrections, dictionary):
        """
        Generates all possible correction variants of a word based on given rules.

        Args:
            word (str): The original word to modify.
            word_length (int): The maximum number of changes allowed.
            rules (dict): A dictionary of replacement rules.
            possible_corrections (set): A set to store the generated corrections.
            dictionary (set of str): A set of valid words forming the dictionary.

        Returns:
            None: Modifies the possible_corrections set in place.
        """
        for num_changes in range(1, word_length + 1):
            for indices in combinations(range(len(word)), num_changes):
                replacement_options = [rules.get(word[i], [word[i]]) for i in indices]
                for replacements in product(*replacement_options):
                    temp = list(word)
                    for idx, replacement in zip(indices, replacements):
                        temp[idx] = replacement
                    candidate = ''.join(temp)
                    if candidate in dictionary:
                        possible_corrections.add(candidate)

                    if not CorrectionModel.correction_pronoun_or_possessive(word):
                        possible_corrections.update(CorrectionModel.correction_insert_missing_letter(candidate, dictionary))
                        possible_corrections.update(CorrectionModel.correction_remove_extra_letter(candidate, dictionary))
                        possible_corrections.update(CorrectionModel.correction_swap_adjacent_letter(candidate, dictionary))

    @staticmethod
    def correction_sentence(word, rules, dictionary, word_freq):
        """
        Attempts to correct a given word based on a set of transformation rules and dictionary validation.
        Generates potential corrections for the input word using the provided rules. Adds the original word to the
        candidate list if it exists in the dictionary. Enhances correction candidates by:
            - Adding missing letters.
            - Removing extra letters.
            - Swapping adjacent letters.

        Filters candidates to retain only valid dictionary words. Prioritizes corrections based on:
            - Word frequency from the Brown corpus.
            - Word length (prefers longer matches).

        Optionally applies a combined correction algorithm for improved context-awareness.

        Args:
            word (str): The input word that needs correction.
            rules (dict): A set of transformation rules specifying possible letter replacements, additions, or deletions.
            dictionary (set of str): A set of valid words forming the dictionary.
            word_freq (dict of str, int): A dictionary where keys are words and
                values are their corresponding frequency scores.

        Returns:
            str: The corrected word or the original word if no better match is determined.
        """
        print(f"Исправление слова: {word}")
        possible_corrections = set()
        CorrectionModel.correction_generate(word, len(word), rules, possible_corrections, dictionary)

        if word in dictionary:
            possible_corrections.add(word)

        if not CorrectionModel.correction_pronoun_or_possessive(word):
            possible_corrections.update(CorrectionModel.correction_insert_missing_letter(word, dictionary))
            possible_corrections.update(CorrectionModel.correction_remove_extra_letter(word, dictionary))
            possible_corrections.update(CorrectionModel.correction_swap_adjacent_letter(word, dictionary))

        matches = CorrectionModel.correction_matches(possible_corrections, word, dictionary, word_freq)
        if matches:
            return matches

        print(f"The word could not be corrected: {word}")
        return word
