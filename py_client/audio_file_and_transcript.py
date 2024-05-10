# import moviepy.editor as mp
# import speech_recognition as sr
# import pandas as pd
# import os
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# import re

# # Specify the path to the video file
# video_file_path = "F:/GP/Apps/backend/py_client/video.mp4"

# # Specify the paths for audio and CSV files
# audio_file_path = "F:/GP/Apps/backend/py_client/audio.wav"
# csv_file_path = "F:/GP/Apps/backend/py_client/extracted_text.csv"

# # Load the video
# video = mp.VideoFileClip(video_file_path)

# # Extract the audio from the video
# audio_file = video.audio
# audio_file.write_audiofile(audio_file_path)


# def clean_text(text):
#     text = re.sub(r"[^A-Za-z]", " ", text)  # Keep only alphabetical characters
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
#     return text


# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()
# ngram_range = (1, 2)  # This example considers unigrams and bigrams
# vectorizer = CountVectorizer(ngram_range=ngram_range)

# # Initialize the recognizer
# r = sr.Recognizer()

# # Load the audio file
# with sr.AudioFile(audio_file_path) as source:
#     data = r.record(source)

# # Convert speech to text
# recognized_text = r.recognize_google(data)

# text = pd.DataFrame({"Text": [recognized_text]})
# text["cleaned_text"] = text["Text"].apply(clean_text)
# text["tokenized_text"] = text["cleaned_text"].apply(word_tokenize)
# text["lowercased_text"] = text["tokenized_text"].apply(
#     lambda tokens: [token.lower() for token in tokens]
# )
# text["filtered_text"] = text["lowercased_text"].apply(
#     lambda tokens: [token for token in tokens if token not in stop_words]
# )
# text["lemmatized_text"] = text["filtered_text"].apply(
#     lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
# )
# X_vectorized = vectorizer.fit_transform(
#     text["lemmatized_text"].apply(lambda tokens: " ".join(tokens))
# )
# # Get the duration of the audio (in seconds)
# duration = audio_file.duration


# # Define a function to calculate start and end times
# def get_time_intervals(text, duration):
#     sentences = text.split(
#         "."
#     )  # Split the text into sentences (you can adjust this based on your use case)
#     interval_length = duration / len(sentences)
#     intervals = []
#     start_time = 0
#     for sentence in sentences:
#         end_time = start_time + interval_length
#         intervals.append((start_time, end_time))
#         start_time = end_time
#     return intervals


# # Get the time intervals
# time_intervals = get_time_intervals(recognized_text, duration)

# # Create a DataFrame with columns: "Raw_Text", "Preprocessed_Text", "Start_Time", "End_Time"
# df = pd.DataFrame(
#     {
#         "Start_Time": [interval[0] for interval in time_intervals],
#         "End_Time": [interval[1] for interval in time_intervals],
#         "Text": [recognized_text],
#         "lemmatized_text": [text["lemmatized_text"].loc[0]],
#     }
# )

# # Save the DataFrame to CSV
# df.to_csv(csv_file_path, index=False)

# print("Transcript with start and end times saved to:", csv_file_path)
# print(X_vectorized)


import moviepy.editor as mp
import speech_recognition as sr
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re


def process_video(video_path):
    def clean_text(text):
        text = re.sub(r"[^A-Za-z]", " ", text)  # Keep only alphabetical characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
        return text

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    ngram_range = (1, 2)  # This example considers unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=ngram_range)

    # Initialize the recognizer
    r = sr.Recognizer()

    # Specify the paths for audio and CSV files
    audio_file_path = os.path.splitext(video_path)[0] + ".wav"
    csv_file_path = os.path.splitext(video_path)[0] + "_extracted_text.csv"

    # Load the video
    video = mp.VideoFileClip(video_path)

    # Extract the audio from the video
    audio_file = video.audio
    audio_file.write_audiofile(audio_file_path)

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        data = r.record(source)

    # Convert speech to text
    recognized_text = r.recognize_google(data)

    text = pd.DataFrame({"Text": [recognized_text]})
    text["cleaned_text"] = text["Text"].apply(clean_text)
    text["tokenized_text"] = text["cleaned_text"].apply(word_tokenize)
    text["lowercased_text"] = text["tokenized_text"].apply(
        lambda tokens: [token.lower() for token in tokens]
    )
    text["filtered_text"] = text["lowercased_text"].apply(
        lambda tokens: [token for token in tokens if token not in stop_words]
    )
    text["lemmatized_text"] = text["filtered_text"].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )
    X_vectorized = vectorizer.fit_transform(
        text["lemmatized_text"].apply(lambda tokens: " ".join(tokens))
    )
    # Get the duration of the audio (in seconds)
    duration = audio_file.duration

    # Define a function to calculate start and end times
    def get_time_intervals(text, duration):
        sentences = text.split(
            "."
        )  # Split the text into sentences (you can adjust this based on your use case)
        interval_length = duration / len(sentences)
        intervals = []
        start_time = 0
        for sentence in sentences:
            end_time = start_time + interval_length
            intervals.append((start_time, end_time))
            start_time = end_time
        return intervals

    # Get the time intervals
    time_intervals = get_time_intervals(recognized_text, duration)

    # Create a DataFrame with columns: "Raw_Text", "Preprocessed_Text", "Start_Time", "End_Time"
    df = pd.DataFrame(
        {
            "Start_Time": [interval[0] for interval in time_intervals],
            "End_Time": [interval[1] for interval in time_intervals],
            "Text": [recognized_text],
            "lemmatized_text": [text["lemmatized_text"].loc[0]],
        }
    )

    # Save the DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    return csv_file_path, audio_file_path  # Return the path to the CSV file


# Now you can call process_video(video_path) to process a video file
