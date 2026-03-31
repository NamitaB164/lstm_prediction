# Character Prediction using LSTM

This project implements a character-level language model using a Long Short-Term Memory (LSTM) network. It was developed as a micro-project for the Deep Learning course to demonstrate the capabilities of Recurrent Neural Networks (RNNs) in sequence modeling and text generation.

## Project Goal

The primary objective is to train a deep learning model that can predict the next character in a sequence of text. By training on the "Tiny Shakespeare" dataset, the model learns the structural patterns, vocabulary, and stylistic nuances of Shakespearean English, allowing it to generate or complete text in a similar fashion.

## Core Idea

The model treats text as a sequence of discrete characters. Unlike word-level models, character-level models have a small, fixed vocabulary (all possible characters), making them robust to out-of-vocabulary words and capable of learning everything from punctuation to capitalization and morphological structures.

## Implementation Logic

The project follows a structured pipeline:

1.  **Data Acquisition**: Loading the Tiny Shakespeare dataset (approx. 1.1 million characters).
2.  **Preprocessing**:
    *   Creating character-to-index and index-to-character mappings.
    *   Encoding the text into integer sequences.
    *   Using a subsequence of 200,000 characters for efficient training.
3.  **Dataset Pipeline**:
    *   Implementing a custom `CharDataset` in PyTorch.
    *   Input: A sequence of length `SEQ_LEN` (150).
    *   Target: The same sequence shifted by one character.
4.  **Model Architecture**:
    *   **Embedding Layer**: Maps character indices to a 128-dimensional dense space.
    *   **LSTM Backbone**: A 3-layer LSTM with 512 hidden units per layer and 30% dropout to prevent overfitting.
    *   **Output Layer**: A fully connected (Linear) layer that maps the 512-dimensional hidden state back to the vocabulary size (65 characters).
5.  **Training**:
    *   Optimization via the Adam optimizer with a learning rate of 0.003.
    *   Loss function: Cross-Entropy Loss to minimize the error between predicted and actual character distributions.
    *   Gradient clipping (magnitude 5) to ensure stable training.

## Mathematical Foundation

The core of the model lies in the LSTM architecture, which manages temporal dependencies through three main gates:

*   **Forget Gate**: Decides what information from the previous cell state to discard.
    $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
*   **Input Gate**: Determines which new information to store in the cell state.
    $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
*   **Output Gate**: Controls what part of the cell state is output as the hidden state.
    $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

The final layer uses a **Softmax** function to produce a probability distribution over the characters:
$P(x_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$

During inference, a **Temperature (T)** parameter is used to scale the logits before softmax to control the "creativity" of the model:
$P(x_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$

## Results

The model was trained for 15 epochs, achieving a steady reduction in training loss from initial high values to approximately **0.2721**.

### Sample Predictions

Given a seed string, the model successfully predicts contextually relevant characters:

*   **Seed**: `ROMEO` -> **Predicted**: `' '` (Space), showing understanding of word boundaries.
*   **Seed**: `To be or not to` -> **Predicted**: `' '`, anticipating the next word in the iconic phrase.
*   **Seed**: `JULIET:\nO ` -> **Predicted**: `'p'`, forming the beginning of a potential word following the vocative 'O'.

## Requirements

*   Python 3.x
*   PyTorch
*   NumPy
*   Tensorflow (for dataset utilities if applicable)