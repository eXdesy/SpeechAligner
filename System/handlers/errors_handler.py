from collections import Counter, defaultdict

class ErrorsHandler:
    """
    A utility class for analyzing and handling character-level errors in text.

    This class provides methods to:
    - Analyze character-level errors between lists of incorrect and correct words.
    - Generate substitution rules based on the frequency of character-level errors.

    Methods:
        errors_analyze(incorrect_words, correct_words) -> dict:
            Analyzes character-level mismatches and creates a frequency map of substitutions.
        errors_rules_generate(errors, threshold=0.7) -> dict:
            Generates substitution rules based on the frequency of errors, applying a minimum threshold.
    """
    @staticmethod
    def errors_analyze(incorrect_words, correct_words):
        """
        Analyzes character-level errors between incorrect and correct word pairs.

        Args:
            incorrect_words (list of str): A list of misspelled words.
            correct_words (list of str): A list of corresponding correctly spelled words.

        Returns:
            errors: A dictionary where keys are correct characters, and values are counters of incorrect characters.
        """
        print("Started analysis of character-level errors...")
        errors = defaultdict(Counter)
        for incorrect, correct in zip(incorrect_words, correct_words):
            for i in range(min(len(incorrect), len(correct))):
                i_char = incorrect[i]
                c_char = correct[i]
                if i_char != c_char:
                    errors[c_char][i_char] += 1
        print("Error analysis completed...")
        return errors

    @staticmethod
    def errors_rules_generate(errors, threshold=0.7):
        """
        Generates substitution rules based on character-level error analysis.

        Args:
            errors (dict): A dictionary of character substitution frequencies.
            threshold (float): The minimum frequency threshold for generating a substitution rule.

        Returns:
            rules: A dictionary of substitution rules where keys are incorrect characters,
                  and values are their corrected counterparts.
        """
        print(f"Creating Replacement Rules with a Threshold: {threshold}")
        rules = {}
        for correct_letter, incorrect_counts in errors.items():
            total_errors = sum(incorrect_counts.values())
            for incorrect_letter, count in incorrect_counts.items():
                weight = count / total_errors
                if weight > threshold:
                    rules[incorrect_letter] = correct_letter
        print(f"Substitution rules successfully created... Number of rules: {len(rules)}")
        return rules
