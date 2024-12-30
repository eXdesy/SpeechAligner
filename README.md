# Speech Aligner System

## Overview
This project is designed to correct errors in written text by leveraging various algorithms such as:
- Levenshtein distance
- Phonetic correction (Double Metaphone)
- Markov Models for context-based corrections
- Machine learning integration using OpenAI GPT

The system can be adapted for multiple languages, including English, Spanish, and Russian, by changing the dictionary and frequency data sources.

## Install Required Libraries
1. Python 3.8+
2. Install dependencies via pip:
   ```bash
   pip install nltk pandas metaphone openai pymorphy2
   ```

## Setup
1. Clone this repository.
2. Download the **MeloTTS** repository from [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS) and place it in the root directory of this project.

## Support
For issues and feature requests, please open an issue on the GitHub repository.

## License
This project is licensed under the MIT License.

