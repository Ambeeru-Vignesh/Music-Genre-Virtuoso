# Music Genre Virtuoso

This project is a music genre classification model that aims to predict the genre of a given music clip using deep learning techniques. The model is trained on the GTZAN dataset, which contains various music genres. This is extension of our midterm project.

In the Final Project, we have enhanced the model architecture from custom CNN model to VGG16 CNN model with additional layers and lot's of fine tuning methods to improve performance

In this enhanced version, we also evaluate the trustworthiness of the AI system based on its fairness, transparency, robustness, and explainability.

## Table of Contents

1. [Project Objective](#project-description)
2. [Project Demo](#demo)
3. [Dataset](#dataset)
4. [Implementation Workflow](#Implementation-workflow)
5. [Code Segments](#code-segments)
6. [Music Genre Classification Frontend](#frontend)
7. [Installation and Steps-To-Run](#installation)
8. [Process Steps](#steps-to-run)
9. [Output Screenshots](#output-screenshots)
10. [Acknowledgments](#acknowledgments)

## 1. Project Objective

Following the midterm project, the application has been evaluated on the aspect of Fairness and Bias, ensuring that the model's predictions are equitable across all music genres.

## 2. Project Demo

[Watch the video](demo.mp4)

## 3. Dataset

The dataset used for training the model is the GTZAN dataset, which consists of audio clips from various music genres. This dataset is helpful in understanding sound and differentiating one song from another. The music clips are labeled with their respective genres, making it suitable for supervised learning tasks.

Dataset: [GTZAN Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

Enhanced the dataset by applying data augmentation techniques: Pitch Shifting, Time Stretching, Noise Injection.

## 4. Implementation Workflow

The provided code contains the implementation of the music genre classification model using Keras with TensorFlow backend. It consists of the following parts:

Here are concise headings with one-line summaries for each section:

1. **Data Preprocessing**  
   Enhanced the dataset using data augmentation techniques like pitch shifting, time stretching, and noise injection.

2. **Model Development**  
   Introduced transfer learning by replacing the custom CNN with advanced architectures like VGG16 and ResNet.

3. **Evaluation**  
   Developed a fairness evaluation methodology using metrics like per-class accuracy, confusion matrix, and fairness index.

4. **Explainability**  
   Leveraged SHAP to compute feature importance for improved model transparency.

5. **Fine-tuning**  
   Reduced overfitting with dropout layers, L2 regularization, and dynamic learning rate adjustment using ReduceLROnPlateau.

## 5. Code Segments

_Data Augmentation_

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,        # Randomly rotate images
    width_shift_range=0.2,    # Randomly shift images horizontally
    height_shift_range=0.2,   # Randomly shift images vertically
    zoom_range=0.2,           # Randomly zoom images
    horizontal_flip=True      # Randomly flip images horizontally
)

# Fit data augmentation generator on training data
datagen.fit(X_train_resized)

```

_VGG16 MODEL_

```
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
# Load pre-trained VGG16 without the top classification layer
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (initially)
base_model.trainable = False

# Add custom layers for music genre classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce feature maps to a single vector
x = Dense(256, activation='relu')(x)  # Larger dense layer for better learning
x = Dropout(0.5)(x)  # Add dropout for regularization
output = Dense(10, activation='softmax')(x)  # Output layer for 10 genres

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

```

_Learning Rate Scheduler_

```
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Use ReduceLROnPlateau to decrease learning rate on plateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50, callbacks=[lr_scheduler])

```

_Range of hyperparameter Values that's been used_

```
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 50, 100],
    'learning_rate': [0.0001, 0.001, 0.01],
}
```

## Music Genre Classification Frontend

This part of the project contains the frontend code that allows users to interact with the trained music genre classification model. The frontend is built using Flask for the web server and HTML/CSS/JavaScript for the user interface.

### `app.py`

This file contains the Flask application that serves as the web server and handles user requests. It also includes functions for processing the audio file and making predictions using the trained model.

The application would be running on http://127.0.0.1:5000

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
2. Place the `MUSIC_GENRE_VGG16_CNN.h5` file in the same directory as `app.py` to load the trained model. (NOTE: I have already created the model so you can skip this if you want to)
3. Ensure that the `allowed_file` function in `app.py` supports the audio file formats you want to allow for prediction (e.g., `.mp3` and `.wav`). (NOTE: Attached two audio files, reggae_genre.wav and country_genre.wav)
4. Run the Flask application using the `app.py` script.
5. Access the application through a web browser and upload an audio file to see the predicted genre.

## 7. Installation and Steps to Run

To directly run the code, follow these steps:

1. Install all the required dependencies mentioned in requirements.txt using pip.
   _example - `pip install tensorflow==2.16.1`._
2. Run `python app.py` and wait for server to start on http://127.0.0.1:5000.
3. simply provide the given example audio files(country_genre.wav or reggae_genre.wav to the MUSIC GENRE VIRTUOSO and witness the intelligence of this Neural Network model)

_Note: If you want to generate new model. please install the required dependencies like `numpy`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `librosa` and run the code cells in the provided Python script or notebook._
[This is the Notebook, please use google colab as you won't require to manually install any dependency](MUSIC_GENRE_CLASSIFICATION_PROJECT.ipynb)

## 8. Process Steps

The code contains detailed explanations of each step, and you can use it to:

1. Load and preprocess the GTZAN dataset.
2. Build VGG16 CNN Model.
3. Implement a regularized and customized CNN layer on to manage overfitting.
4. Perform Fine Tuning.
5. Evaluate through Fairness Evaluation Methodology.
   Metrics used:
   1. Per-Class Accuracy
   2. Confusion Matrix
   3. Fairness Index
6. Predict the genre of new songs using the built trained model.

## 9. Output Screenshots

![The Frontend](<Output Screenshots/Screenshot Frontend.png>)

![The Output predicting Reggae genre correctly](<Output Screenshots/Screenshot Predicting Reggae Genre.png>)

![The Output predicting Country genre correctly](<Output Screenshots/Screenshot Predicting Country Genre.png>)

## 10. Acknowledgments

We would like to thank Kaggle and Andrada Olteanu for providing the GTZAN dataset, which makes this project possible.

---

_Note: This README file provides an overview of the project and how to use the code. Make sure to execute the code in a Python environment with all the required dependencies installed._
