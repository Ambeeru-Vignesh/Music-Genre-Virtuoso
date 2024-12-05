import math
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Loaded the new trained VGG16CNN architecture model
model = tf.keras.models.load_model('MUSIC_GENRE_VGG16_CNN.h5')


# Audio files pre-processing
def process_input(audio_file, track_duration):
    sample_rate = 22050
    num_mfcc = 13
    n_fft = 2048
    hop_length = 512
    track_duration = track_duration  # measured in seconds
    samples_per_track = sample_rate * track_duration
    num_segments = 10

    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(audio_file, sr=sample_rate)

    mfcc_vectors = []  # List to store the MFCC vectors for each segment

    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(
            y=signal[start:finish],
            sr=sample_rate,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfcc = mfcc.T

        mfcc_vectors.append(mfcc)  # Append the extracted MFCC vectors to the list

    return mfcc_vectors


genre_dict = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']

    if uploaded_file and allowed_file(uploaded_file.filename):
        try:
            new_input_mfcc = process_input(uploaded_file, 30)
            X_to_predict = np.array(new_input_mfcc)
            # Reshape X_to_predict to match the expected input shape of the model
            X_to_predict = X_to_predict.reshape(X_to_predict.shape[0], X_to_predict.shape[1], X_to_predict.shape[2], 1)
            prediction = model.predict(X_to_predict)
            # Get the most common predicted index
            predicted_index = np.argmax(np.bincount(prediction.argmax(axis=1)))
            genre = genre_dict[predicted_index]
            return genre
        except Exception as e:
            return "Error processing audio: " + str(e)
    else:
        return "Invalid file format. Please upload an audio file which is .mp3 or .wav."


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
