# Music Genre Virtuoso

This project is a music genre classification model that aims to predict the genre of a given music clip using deep learning techniques. The model is trained on the GTZAN dataset, which contains various music genres.

## Table of Contents

1. [Project Description](#project-description)
2. [Project Demo](#demo)
3. [Dataset](#dataset)
4. [Code](#code)
5. [Frontend](#frontend)
6. [Installation and Steps-To-Run](#installation)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

## Project Description

The goal of this project is to create a music genre classification model that can identify the genre of a given music clip accurately. The model is built using a combination of machine learning and deep learning techniques, specifically an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN).

## Project Demo

![demo](demo.mp4)

### frontend

![The Frontend](<Output Screenshots/Screenshot 1.png>)

![The Output](<Output Screenshots/Screenshot 2.png>)

## Dataset

The dataset used for training the model is the GTZAN dataset, which consists of audio clips from various music genres. This dataset is helpful in understanding sound and differentiating one song from another. The music clips are labeled with their respective genres, making it suitable for supervised learning tasks.

Dataset: [GTZAN Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## Code

The provided code contains the implementation of the music genre classification model using Keras with TensorFlow backend. It consists of the following parts:

1. Loading and preprocessing the dataset
2. Building a simple ANN model
3. Training the ANN model
4. Managing overfitting with regularization
5. Implementing a Convolutional Neural Network (CNN) for genre classification
6. Evaluating the CNN model on the test set
7. Predicting the genre of new songs using the trained model

# Music Genre Classification Frontend

This part of the project contains the frontend code that allows users to interact with the trained music genre classification model. The frontend is built using Flask for the web server and HTML/CSS/JavaScript for the user interface.

### `app.py`

This file contains the Flask application that serves as the web server and handles user requests. It also includes functions for processing the audio file and making predictions using the trained model.

### Endpoints:

1. `/`: The index route that renders the main page where users can upload an audio file for prediction.
2. `/predict`: The route that receives the uploaded audio file, processes it, and returns the predicted music genre.

### `index.html`

This HTML file contains the user interface for the music genre classifier. It allows users to upload an audio file (in `.mp3` or `.wav` format) and see the predicted genre after processing.

### Elements:

1. **Upload Button**: Allows users to click and select an audio file to upload.
2. **Drop Area**: Users can drag and drop an audio file to upload it.
3. **Progress Bar**: Shows the progress of file upload and processing.
4. **Genre Result**: Displays the predicted genre after processing.
5. **Audio Player**: An HTML5 audio element to play the uploaded audio file.

### How to Use

1. Make sure you have the necessary dependencies installed to run the Flask application and serve the frontend.(listed all the dependencies in requirements.txt)
2. Place the `Music_Genrel_CNN_Model.h5` file in the same directory as `app.py` to load the trained model. (NOTE: I have already created the model so you can skip this if you want to)
3. Ensure that the `allowed_file` function in `app.py` supports the audio file formats you want to allow for prediction (e.g., `.mp3` and `.wav`). (NOTE: Attached two audio files, reggae_genre.wav and country_genre.wav)
4. Run the Flask application using the `app.py` script.
5. Access the application through a web browser and upload an audio file to see the predicted genre.

## Installation and Steps to Run

To directly run the code, follow these steps:

1. Install all the required dependencies mentioned in requirements.txt using pip.
   example - `pip install tensorflow==2.16.1`.
2. Run `python app.py`.
3. simply provide the given example audio files(country_genre.wav or reggae_genre.wav to the MUSIC GENRE VIRTUOSO and witness the intelligence of this Neural Network model)

_Note: If you want to generate new model. please install the required dependencies like `numpy`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `librosa` and run the code cells in the provided Python script or notebook._

## Usage

The code contains detailed explanations of each step, and you can use it to:

1. Load and preprocess the GTZAN dataset.
2. Build and train a simple ANN model for music genre classification.
3. Implement a regularized ANN model to manage overfitting.
4. Create a CNN model for genre classification.
5. Evaluate the trained CNN model on the test set.
6. Predict the genre of new songs using the trained model.

## Acknowledgments

We would like to thank Kaggle and Andrada Olteanu for providing the GTZAN dataset, which makes this project possible.

---

_Note: This README file provides an overview of the project and how to use the code. Make sure to execute the code in a Python environment with all the required dependencies installed._
