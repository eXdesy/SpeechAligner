# © 2025 eXdesy — All rights reserved.
# This code is for educational use only.
# Do not reuse, copy, modify, or redistribute.

import os
import threading
import warnings

from System.main import System
from System.handlers.file_handler import FileHandler
from System.handlers.errors_handler import ErrorsHandler

# Suppress specific future warnings to avoid unnecessary clutter in the console
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(model_name, model_size, csv_path, model_path, language):
    """
    Trains a speech alignment model with specified parameters.

    Args:
        model_name (str): Name of the model to be trained.
        model_size (str): Size of the model (e.g., 'small', 'medium', 'large').
        csv_path (str): Path to the CSV file for logging errors.
        model_path (str): Path to save the trained model.
        language (str): Language of the data for model training.

    Workflow:
        1. Initializes a System instance and starts the training process.
        2. Loads error data from the CSV file.
        3. Analyzes errors and generates replacement rules.
        4. Updates the model with the generated rules.
    """
    # Run the training mode of the aligner system
    aligner = System()
    aligner.run_train_mode(model_name=model_name, model_size=model_size, language=language, csv_path=csv_path)

    # Load errors identified during training from the specified CSV file
    incorrect, correct = FileHandler.file_errors_load(csv_path)

    # Analyze the errors and generate replacement rules for corrections
    letter_errors = ErrorsHandler.errors_analyze(incorrect, correct)
    replacement_rules = ErrorsHandler.errors_rules_generate(letter_errors)

    # Update the correction model with the newly generated rules
    FileHandler.file_model_update(model_path, replacement_rules)

def use_model(model_name, model_size, model_path, language, method):
    """
    Uses a trained speech alignment model to process input audio and apply corrections.

    Args:
        model_name (str): Name of the model to be used.
        model_size (str): Size of the model (e.g., 'small', 'medium', 'large').
        model_path (str): Path to the model to be used.
        language (str): Language of the data to be processed.
        method (str): Method to apply additional checks or fixes.

    Workflow:
        1. Initializes a System instance and sets it to use mode.
        2. Processes input using the specified model and method.
    """
    # Run the aligner in use mode to process input audio and apply corrections
    aligner = System()
    aligner.run_use_mode(model_name=model_name, model_size=model_size, language=language, model_path=model_path, method=method)

def use_model_test(model_path, language, method):
    """
    Tests the model using predefined test cases to validate correction capabilities.

    Args:
        model_path (str): Path to the model to be tested.
        language (str): Language of the data for testing.
        method (str): Method to apply additional checks or fixes.

    Workflow:
        1. Initializes a System instance and sets it to test mode.
        2. Processes predefined test inputs using the specified model and method.
    """
    # Test the aligner's correction capabilities using predefined input
    aligner = System()
    aligner.run_use_model_test(language=language, model_path=model_path, method=method)

if __name__ == "__main__":
    supported_languages = {
        "en",
        "es",
        "ru",
    }

    # Gather user details and preferences for the session
    user_name = input("Enter your name: ").strip()
    language = input("Enter the language ('en', 'es', 'ru'): ").strip()
    while language not in supported_languages:
        print(f"Language '{language}' is not supported. Please try again.")
        language = input("Enter the language ('en', 'es', 'ru'): ").strip()

    method = input("Enter \"gpt\" fix if you want to enable additional checks or just click \"Enter\": ").strip().lower()

    # Set up working directories and paths for files
    my_dir = os.path.join(os.getcwd())
    csv_path = FileHandler.file_errors_create(language, user_name, my_dir)
    model_path = FileHandler.file_model_create(language, user_name, my_dir)

    # Default model parameters for training and usage
    model_name = "whisper"
    model_size = "turbo"

    while True:
        print(f"\nWelcome to the System! Dear {user_name}, selected language is: {language}")
        print("1. Train model.")
        print("2. Use model.")
        print("3. Test use model.")
        print("4. Exit.")
        action = input("Choose an option: ").strip()

        if action == '1':
            # Start a separate thread for training to avoid blocking the main program
            print("Starting training thread...")
            training_thread = threading.Thread(target=train_model, args=(model_name, model_size, csv_path, model_path, language))
            training_thread.start()

            # Wait for the training thread to complete
            training_thread.join()
            print("Training thread completed...")

        elif action == '2':
            # Start a separate thread for using the model in mode
            print("Starting usage thread...")
            training_thread = threading.Thread(target=use_model, args=(model_name, model_size, model_path, language, method))
            training_thread.start()

            # Wait for the usage thread to complete
            training_thread.join()
            print("Usage thread completed...")

        elif action == '3':
            # Start a separate thread for testing the model with specific input
            print("Starting testing thread...")
            training_thread = threading.Thread(target=use_model_test, args=(model_path, language, method))
            training_thread.start()

            # Wait for the testing thread to complete
            training_thread.join()
            print("Testing thread completed...")

        elif action == "4":
            print("Exiting the program...")
            break

        else:
            # Handle invalid actions
            print("\nInvalid action...\n")
