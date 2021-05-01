import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model


# load face detector
detector = MTCNN()

# load the model
emotion_model = load_model('../models/my-emotion-model-final-3.hdf5')


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def detect_face(img):
    mt_res = detector.detect_faces(img)
    return_res = []

    for face in mt_res:
        x, y, width, height = face['box']
        center = [x + (width / 2), y + (height / 2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0] - (max_border / 2)), 0)
        right = max(int(center[0] + (max_border / 2)), 0)
        top = max(int(center[1] - (max_border / 2)), 0)
        bottom = max(int(center[1] + (max_border / 2)), 0)

        # crop the face
        center_img_k = img[top:top + max_border,
                       left:left + max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([224, 224]))

        # create predictions

        # convert to grey scale then predict using the emotion model
        grey_img = np.array(Image.fromarray(center_img_k).resize([48, 48]))
        emotion_preds = emotion_model.predict(rgb2gray(grey_img).reshape(1, 48, 48, 1))

        # output to the cv2
        return_res.append([top, right, bottom, left, emotion_preds])

    return return_res


# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

emotion_dict = {
    0: 'Surprise',
    1: 'Happy',
    2: 'Disgust',
    3: 'Anger',
    4: 'Sadness',
    5: 'Fear',
    6: 'Contempt'
}

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = detect_face(rgb_frame)

    # Display the results
    for top, right, bottom, left, emotion_preds in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.putText(frame, 'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)], np.max(emotion_preds)),
                    (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()