import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from tkinter import Tk, Label, Button
import tkinter.font as font
import time
import pandas as pd
import matplotlib.pyplot as plt

# load face detector
detector = MTCNN()

# load the model
emotion_model = load_model('../models/my-emotion-model-4.hdf5')


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def detect_face(image):
    deteced_faces = detector.detect_faces(image)
    results = []

    for face in deteced_faces:
        x, y, width, height = face['box']
        center = [x + (width / 2), y + (height / 2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0] - (max_border / 2)), 0)
        right = max(int(center[0] + (max_border / 2)), 0)
        top = max(int(center[1] - (max_border / 2)), 0)
        bottom = max(int(center[1] + (max_border / 2)), 0)

        # crop the face
        center_img_k = image[top:top + max_border,
                       left:left + max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([224, 224]))

        # create predictions

        # convert to grayscale then predict using the emotion model
        grey_img = np.array(Image.fromarray(center_img_k).resize([48, 48]))
        emotion_preds = emotion_model.predict(rgb2gray(grey_img).reshape(1, 48, 48, 1))

        # output to the cv2
        results.append([top, right, bottom, left, emotion_preds])

    return results


def mainvideo():
    # Parameters
    emotion_dict = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happiness',
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }
    emotion_score = {
        0: -3,
        1: -3,
        2: -2,
        3: 2,
        4: -1,
        5: 1,
        6: 0
    }

    # Get a reference to webcam
    video_capture = cv2.VideoCapture(0)

    emotion = ""
    data = pd.read_csv(r"data.csv")
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

            cv2.putText(frame,
                        'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)], np.max(emotion_preds)),
                        (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        if emotion == "":
            emotion = np.argmax(emotion_preds)
        if emotion != np.argmax(emotion_preds):
            df1 = pd.DataFrame({'Emotion': [emotion_dict[np.argmax(emotion_preds)]],
                                'Time': [time.time()],
                                'Score': [emotion_score[np.argmax(emotion_preds)]]
                                })
            data = pd.concat([data, df1])
            emotion = np.argmax(emotion_preds)

            data.to_csv(r"data.csv")

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def graph():
    data = pd.read_csv(r"data.csv")
    plt.plot(data["Time"], data["Score"])
    plt.xlabel("Time")
    plt.ylabel("Emotion Score")
    plt.show()


def faces():
    im = Image.open("picemot.png")
    im.show()


if __name__ == "__main__":
    root = Tk()
    root.configure(background="DeepSkyBlue2")
    root.title("Zoomer")
    root.geometry("300x300")

    Label(root, text="", bg="DeepSkyBlue2").grid(row=2, column=0)

    Label(root, text="", bg="DeepSkyBlue2").grid(row=3, column=0)
    Label(root, text="", bg="DeepSkyBlue2").grid(row=3, column=1)
    Label(root, text="", bg="DeepSkyBlue2").grid(row=3, column=2)
    Label(root, text="", bg="DeepSkyBlue2").grid(row=3, column=3)

    myFont = font.Font(family='Helvetica')
    Label(root, text="Welcome to Zoomer!", font=myFont, bg='DeepSkyBlue2', fg='black').grid(row=4, column=3)
    Label(root, text="", bg="DeepSkyBlue2").grid(row=5, column=3)
    btn = Button(root,
                 text="  Sentiment Analyser !! ",
                 fg="white",
                 bg="DeepSkyBlue3",
                 command=mainvideo,
                 padx=20, pady=20,
                 activeforeground='DeepSkyBlue2',
                 relief='solid',
                 font=myFont)
    btn.grid(row=6, column=3)
    btn1 = Button(root,
                  text="  Graph ",
                  fg="white",
                  bg="DeepSkyBlue3",
                  command=graph,
                  padx=10, pady=10,
                  activeforeground='DeepSkyBlue2',
                  relief='solid',
                  )
    Label(root, text="", bg="DeepSkyBlue2").grid(row=7, column=3)
    btn1.grid(row=8, column=3)
    btn2 = Button(root,
                  text="  Emotions ",
                  fg="white",
                  bg='DeepSkyBlue3',
                  command=faces,
                  padx=2, pady=2,
                  activeforeground='DeepSkyBlue2',
                  relief='solid',
                  )
    Label(root, text="", bg="DeepSkyBlue2", fg='black').grid(row=9, column=3)
    btn2.grid(row=10, column=3)

    root.mainloop()