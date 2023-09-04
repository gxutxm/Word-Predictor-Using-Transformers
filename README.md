# Word-Predictor-Using-Transformers

This is a language model that can predict the next word in a sequence. It is based on the Transformer architecture, which is a type of neural network that has been shown to be very effective for natural language processing tasks.This project provides a Python script to make predictions using several transformer-based language models. It includes models like BERT, XLNet, XLM-Roberta, BART, Electra, and Roberta. The script allows you to input a text sentence with a masked token ("<mask<mask>>") and generates predictions for the most likely tokens to fill in the mask using each model.

## Features:

1. **Multi-Model Compatibility:** This project is designed to work with multiple transformer-based language models, including BERT, XLNet, XLM-Roberta, BART, Electra, and Roberta. It allows users to seamlessly switch between these models for masked token predictions.
2. **Predictive Functionality:** The core functionality of this project is to generate predictions for a given input text sentence with a masked token ("<mask<mask>>"). It provides predictions for the most likely tokens to fill in the mask using each selected model.
3. **Customizable Prediction Count:** Users can customize the number of top predictions to consider for each model by adjusting the top_k variable. This flexibility allows for fine-tuning the prediction results according to specific requirements.
4. **Ease of Use:** The code includes functions for encoding, decoding, and generating predictions. Users can easily integrate this functionality into their own projects or use it for experimentation and analysis.
5. **Pre-Trained Models:** The project leverages pre-trained transformer models from the Hugging Face Transformers library, which are known for their high performance in natural language understanding tasks. This ensures that the predictions are based on state-of-the-art language models.
6. **Tokenization Support:** The code includes tokenization for input sentences, ensuring that the input text is correctly processed for each model. It also handles special cases where the masked token is the last token in the sentence.

## Features of Transformer models:

1. **BERT (Bidirectional Encoder Representations from Transformers):**
    - **Bidirectional Context:** BERT is pre-trained as a bidirectional model, meaning it considers both left and right context when predicting words in a sentence.
    - **Masked Language Modeling:** BERT is pre-trained using a masked language modeling objective, where it learns to predict masked words within a sentence.
    - **Fine-Tuning:** BERT can be fine-tuned for various downstream NLP tasks, such as text classification, named entity recognition, and question answering.
    - **Variants:** BERT has multiple variants, including BERT-base and BERT-large, with varying model sizes and capacities.
2. **XLNet (eXtreme Learning with a Transformer):**
    - **Permutation-Based Training:** XLNet uses a permutation-based training approach, allowing it to capture dependencies among all words in a sentence.
    - **Bidirectional Context:** Similar to BERT, XLNet considers bidirectional context.
    - **Fine-Tuning:** XLNet can also be fine-tuned for a wide range of NLP tasks.
    - **Large Model Capacity:** XLNet models are known for their large model sizes and high performance on various tasks.
3. **XLM-Roberta (Cross-lingual Language Model Roberta):**
    - **Cross-Lingual Capabilities:** XLM-Roberta is designed to handle multiple languages and is pre-trained on a wide range of languages.
    - **RoBERTa Base:** It is built upon the RoBERTa architecture, which is an optimized version of BERT with improved training techniques.
    - **Fine-Tuning:** Like other models, XLM-Roberta can be fine-tuned for cross-lingual tasks and multilingual applications.
4. **BART (BART: Denoising Sequence-to-Sequence Pre-training):**
    - **Seq2Seq Architecture:** BART is a sequence-to-sequence model designed for tasks like text summarization and text generation.
    - **Denoising Objective:** It is pre-trained with a denoising objective, where it learns to reconstruct noisy or partially masked sentences.
    - **Abstractive Summarization:** BART is particularly well-suited for abstractive summarization tasks, where it generates concise summaries of longer text.
5. **Electra (Efficiently Learning an Encoder that Classifies Token Replacements Accurately):**
    - **Discriminative Training:** Electra introduces a novel approach where a discriminator is trained to distinguish between original and replaced tokens.
    - **Efficiency:** Electra is designed for efficiency and competitive performance with smaller model sizes.
    - **Masked Language Modeling:** Similar to BERT, it uses a masked language modeling objective but with a different training paradigm.
6. **RoBERTa (A Robustly Optimized BERT Pretraining Approach):**
    - **Optimized Pretraining:** RoBERTa is an optimized version of BERT with improved training strategies, including larger batch sizes and more data.
    - **Masked Language Modeling:** It follows the same masked language modeling objective as BERT.
    - **Large-Scale Training:** RoBERTa is trained on a vast amount of text data and achieves state-of-the-art results on many NLP benchmarks.

## Installation:

1. Clone the repository:
    
    ```
    git clone https://github.com/gxutxm/Word-Predictor-Using-Transformers
    ```
    
2. Install the required dependencies:
    
    `pip install -r requirements.txt`
    

## Usage:

1. Open the terminal and navigate to the project directory.
2. Run the functionality that you would like to perform. For example,
    
    ```
    python3 app.py
    ```
    
3. The script will initialise the Flask-based API for generating predictions with transformer-based language models and the API will respond with JSON data containing predictions generated by transformer-based models. 
4. Click on the IP address from the terminal and try forming sentences to see how different transformers suggests new words

## Result:

## Project Structure:

```
**Word-Predictor-Using-Transformers**/
│
├── README.md                  # Project documentation
│
├── requirements.txt           # List of project dependencies
│
├── .gitignore                 # Specify files and directories to be ignored by version control
│
├── main.py                    # Main Python script for running the application
│
├── app.py                     # Python script for your application logic
│
├── static/                    # Static files (CSS and JavaScript)
│   ├── css/                   # CSS files
│   │   ├── app.css            # Your application's custom CSS
│   │   ├── bootstrap-suggest.css  # Custom CSS (if needed)
│   │   └── bootstrap.min.css  # Bootstrap CSS
│   │
│   ├── js/                    # JavaScript files
│   │   ├── app.js             # Your application's custom JavaScript
│   │   ├── bootstrap-suggest.js   # Custom JavaScript (if needed)
│   │   └── bootstrap.min.js   # Bootstrap JavaScript
│
├── templates/                 # HTML templates
│   ├── index.html             # Main HTML template for your web app
│
├── .env                       # Environment variables (for sensitive data)
│
├── venv/                      # Virtual environment (create using virtualenv or similar)
```
