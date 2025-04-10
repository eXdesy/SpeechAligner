# © 2025 eXdesy — All rights reserved.
# This code is for educational use only.
# Do not reuse, copy, modify, or redistribute.

import csv
import os
import pickle
import pandas as pd

class FileHandler:
    """
    A utility class for handling file operations related to error logging and model management.

    This class provides methods for:
    - Creating and managing directories and files for error logs.
    - Appending data to error log files.
    - Loading data from CSV files into pandas structures.
    - Saving and loading models using pickle serialization.

    Methods:
        file_errors_create(language, user_name, user_dir) -> str:
            Creates a directory and CSV file for error logging if it doesn't exist.
        file_errors_update(file_path, rows):
            Appends error data to an existing CSV file.
        file_errors_load(file_path) -> tuple:
            Loads data from a CSV file into pandas Series.
        file_model_create(language, user_name, user_dir) -> str:
            Creates a directory for model files and returns the path for the model file.
        file_model_update(file_path, data):
            Saves data to a file using pickle serialization.
        file_model_load(file_path) -> object:
            Loads data from a pickle file.
    """
    @staticmethod
    def file_errors_create(language: str, user_name: str, user_dir: str) -> str:
        """
        Creates the UserErrorsFiles directory with CSV errors file for each user with logging errors if it does not exist.

        If the errors file already exists, it loads the file using file_errors_load.

        Args:
            language (str): Language code (e.g., "en").
            user_name (str): Name of the user.
            user_dir (str): Directory path for user files.

        Returns:
            str: Path to the created or existing CSV file.
        """
        path = os.path.join(user_dir, "UserFiles", user_name, "errors")
        if not os.path.exists(path):
            os.makedirs(path)

        csv_path = os.path.join(path, f"errors_{language}.csv")

        if os.path.exists(csv_path):
            FileHandler.file_errors_load(csv_path)
        else:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Incorrect", "Correct"])
            print(f"Errors data created successfully...")

        return csv_path

    @staticmethod
    def file_errors_update(file_path: str, rows: list):
        """
        Appends error rows to a CSV errors file.

        Args:
            file_path (str): Path to the CSV file.
            rows (list): Rows to append.
        """
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"Errors data updated successfully...")

    @staticmethod
    def file_errors_load(file_path):
        """
        Loads incorrect and correct data from a CSV file. The CSV file is expected to have at least two columns:
        'incorrect' and 'correct'. These columns will be returned as separate data structures.

        Args:
            file_path (str): The path to the CSV file containing the data.

        Returns:
            tuple: A tuple containing two pandas Series: (incorrect, correct).
        """
        if not os.path.exists(file_path):
            print("File not found. Please create it first..")
            return None

        data = pd.read_csv(file_path)

        print(f"Errors data loaded successfully. Number of records: {len(data)}")
        return data['Incorrect'], data['Correct']

    @staticmethod
    def file_model_create(language: str, user_name: str, user_dir: str) -> str:
        """
        Creates a directory for model files and defines the path for the model file.

        If the model file already exists, it loads the model using file_model_load.

        Args:
            language (str): Language code (e.g., "en").
            user_name (str): Name of the user.
            user_dir (str): Directory path for user files.

        Returns:
            str: Path to the model file.
        """
        path = os.path.join(user_dir, "UserFiles", user_name, "rules")
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = os.path.join(path, f"rules_model_{language}.pkl")

        if os.path.exists(model_path):
            FileHandler.file_model_load(model_path)
        else:
            print("Model file was created successfully...")
        return model_path

    @staticmethod
    def file_model_update(file_path: str, data):
        """
        Saves the provided data to a file using pickle serialization.

        Args:
            file_path (str): The path where the data will be saved.
            data (object): The data to be serialized and saved.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model file was saved successfully...")

    @staticmethod
    def file_model_load(file_path):
        """
        Loads data from a file saved using pickle serialization.

        Args:
            file_path (str): The path to the file to be loaded.

        Returns:
            object: The deserialized data from the file. Returns None if the file does not exist.
        """
        if not os.path.exists(file_path):
            print("Model file not found. Please create it first...")
            return None

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("Model file was uploaded successfully...")
        return data
