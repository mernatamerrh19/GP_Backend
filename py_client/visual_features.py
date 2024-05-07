import cv2
import dlib
import numpy as np
import math
import csv
from feat import Detector


def get_focal_length(landmarks):
    # Assuming distance between eyes represents focal length
    left_eye_center = np.mean(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0
    )
    right_eye_center = np.mean(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0
    )
    distance = np.linalg.norm(left_eye_center - right_eye_center)
    return distance


def pixel_to_world(x, y, z, focal_length):
    # Convert pixel coordinates to world coordinates
    x_world = x * z / focal_length
    y_world = y * z / focal_length
    return x_world, y_world, z


# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create a Detector instance
detector_feat = Detector()

# Open the video file
cap = cv2.VideoCapture("video.mp4")

# Initialize variables for calculating average face size and distance
total_face_sizes = 0
num_frames = 0

# Initialize variables to store pose, gaze direction, and AU results
pose_results = []
gaze_results = []
au_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform AU detection
    detected_faces = detector_feat.detect_faces(frame)
    detected_landmarks = detector_feat.detect_landmarks(frame, detected_faces)
    detected_aus = detector_feat.detect_aus(frame, detected_landmarks)
    au_results.append(detected_aus)

    # Convert frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Calculate the focal length for this frame based on the distance between eye landmarks
        left_eye_center = np.mean(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0
        )
        right_eye_center = np.mean(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0
        )
        focal_length = np.linalg.norm(left_eye_center - right_eye_center)

        # Extract left and right eye landmarks
        left_eye = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        )
        right_eye = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        )

        # Calculate gaze direction vectors for each eye
        gaze_left = np.mean(left_eye, axis=0).astype("int")
        gaze_right = np.mean(right_eye, axis=0).astype("int")

        # Calculate depth for each eye independently based on their position
        gaze_0_z = np.linalg.norm(left_eye_center - right_eye_center) / 2
        gaze_1_z = np.linalg.norm(right_eye_center - left_eye_center) / 2

        # Convert gaze vectors to world coordinates
        gaze_0_x, gaze_0_y, _ = pixel_to_world(
            gaze_left[0], gaze_left[1], gaze_0_z, focal_length
        )
        gaze_1_x, gaze_1_y, _ = pixel_to_world(
            gaze_right[0], gaze_right[1], gaze_1_z, focal_length
        )

        # Calculate gaze angles (averaged for both eyes)
        gaze_angle_x = np.arctan2((gaze_0_x + gaze_1_x) / 2, focal_length)
        gaze_angle_y = np.arctan2((gaze_0_y + gaze_1_y) / 2, focal_length)

        # Store gaze results
        gaze_results.append(
            (
                gaze_0_x,
                gaze_0_y,
                gaze_0_z,
                gaze_1_x,
                gaze_1_y,
                gaze_1_z,
                gaze_angle_x,
                gaze_angle_y,
            )
        )

        # Calculate rotation around X-axis (pitch)
        Rx = math.atan2(
            landmarks.part(30).y - landmarks.part(8).y,
            landmarks.part(30).x - landmarks.part(8).x,
        )
        # Calculate rotation around Y-axis (yaw)
        Ry = math.atan2(
            landmarks.part(16).y - landmarks.part(0).y,
            landmarks.part(16).x - landmarks.part(0).x,
        )
        # Calculate rotation around Z-axis (roll)
        Rz = math.atan2(
            landmarks.part(54).y - landmarks.part(48).y,
            landmarks.part(54).x - landmarks.part(48).x,
        )

        # Calculate face size (Euclidean distance between two corners of the eye)
        eye1 = (landmarks.part(36).x, landmarks.part(36).y)
        eye2 = (landmarks.part(45).x, landmarks.part(45).y)
        face_size = np.linalg.norm(np.array(eye1) - np.array(eye2))

        # Update total face size and number of frames
        total_face_sizes += face_size
        num_frames += 1

        # Calculate average face size and estimate distance from the camera
        avg_face_size = total_face_sizes / num_frames
        # Assuming a constant reference face size and distance
        REF_FACE_SIZE = 100  # Assume reference face size as 100 pixels
        REF_DISTANCE = 1000  # Assume reference distance as 1000 mm
        Tz = -REF_DISTANCE * REF_FACE_SIZE / avg_face_size  # In millimeters

        # Calculate Tx and Ty based on the centroid of facial landmarks
        centroid_x = np.mean([landmarks.part(i).x for i in range(68)])
        centroid_y = np.mean([landmarks.part(i).y for i in range(68)])
        Tx = centroid_x - frame.shape[1] / 2  # Adjust for image center
        Ty = centroid_y - frame.shape[0] / 2  # Adjust for image center

        # Store pose results
        pose_results.append((Tz, Tx, Ty, Rx, Ry, Rz))


# Open a CSV file for writing
with open("facial_features.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write headers
    writer.writerow(
        [
            "pose_Tx",
            "pose_Ty",
            "pose_Tz",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
            "gaze_0_x",
            "gaze_0_y",
            "gaze_0_z",
            "gaze_1_x",
            "gaze_1_y",
            "gaze_1_z",
            "gaze_angle_x",
            "gaze_angle_y",
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU11_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU24_r",
            "AU25_r",
            "AU26_r",
            "AU28_r",
            "AU43_r",
        ]
    )

    # Write data rows
    for pose, gaze, au in zip(pose_results, gaze_results, au_results):
        row = [*pose, *gaze, *au]  # Merge pose, gaze, and AU data into a single row
        writer.writerow(row)

####### Normalization left to be done
####### Dropping not needed AUs not done yet