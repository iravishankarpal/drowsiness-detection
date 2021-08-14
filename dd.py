import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import imutils
import datetime
import dlib
import requests
from PIL import ImageTk
import PIL.Image
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog


def lip_distance(shape):  # For Acquiring Mouth Landmarks
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


def f1():  # Main Function
    mixer.init()
    sound = mixer.Sound('sleepalert.wav')  # Loading Alarm

    face = cv2.CascadeClassifier(
        'haar cascade files\haarcascade_frontalface_alt.xml')  # Face cascade
    leye = cv2.CascadeClassifier(
        'haar cascade files\haarcascade_lefteye_2splits.xml')  # Left eye cascade
    reye = cv2.CascadeClassifier(
        'haar cascade files\haarcascade_righteye_2splits.xml')  # Right eye cascade

    lbl = ['Close ', 'Open']

    model = load_model('models/cnnCat2.h5')  # loading model
    path = os.getcwd()
    cap = cv2.VideoCapture(0)  # Capturing Video feed

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]
    YAWN_THRESH = 30  # Yawning threshold
    alarm_status = False

    detector = dlib.get_frontal_face_detector()  # shape predictor for yawning
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    root.withdraw()

    while(True):

        rect, frame = cap.read()  # processing video stream
        height, width = frame.shape[:2]

        # converting video to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        faces = face.detectMultiScale(
            gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))  # Detecting Face
        left_eye = leye.detectMultiScale(gray)  # Detecting left eye
        right_eye = reye.detectMultiScale(gray)  # Detecting right eye

        cv2.rectangle(frame, (0, height-50), (200, height),
                      (0, 0, 0), thickness=cv2.FILLED)  # building frame

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        # Analyse the right eye
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y+h, x:x+w]
            count = count+1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0] == 1):
                lbl = 'Open'
            if(rpred[0] == 0):
                lbl = 'Closed'
            break

        # Analyse the left eye
        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y+h, x:x+w]
            count = count+1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0] == 1):
                lbl = 'Open'
            if(lpred[0] == 0):
                lbl = 'Closed'
            break

        # Analyse the mouth for yawn
        for rect in rects:
            # for (x, y, w, h) in rects:
            # rect = edlib.rctangle(int(x), int(y), int(x + w),int(y + h))     #for embedded system

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            distance = lip_distance(shape)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if (distance > YAWN_THRESH):
                # cv2.putText(frame, "Yawn Alert", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status == False:
                    alarm_status = True
                    sound.play()  # Alarm is triggered
                    Current_time = datetime.datetime.now().strftime('%d-%b-%Y-%H-%M-%S')
                    cv2.imwrite(
                        r'C:\Users\Akshay\DrowsinessDetection\Snaps\YawnDetection_' + str(Current_time) + '.jpg', frame)  # Snapshot of detection
                    cv2.rectangle(frame, (0, 0), (width, height),
                                  (0, 0, 255), thicc)
            else:
                alarm_status = False

            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (200, height-20),
                        font, 1, (255, 255, 255), 1, cv2.LINE_AA)  # Yawn score on frame
            break

        if(rpred[0] == 0 and lpred[0] == 0):
            score = score+1
            cv2.putText(frame, "Closed", (10, height-20), font,
                        1, (255, 255, 255), 1, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score = score-1
            cv2.putText(frame, "Open", (10, height-20), font,
                        1, (255, 255, 255), 1, cv2.LINE_AA)

        if(score < 0):
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height-20),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if(score > 15):
            # person is feeling sleepy so we beep the alarm
            Current_time = datetime.datetime.now().strftime('%d-%b-%Y-%H.%M.%S')
            cv2.imwrite(
                r'C:\Users\Akshay\DrowsinessDetection\Snaps\SleepDetection' + str(Current_time) + '.jpg', frame)  # Snapshot of detection
            try:
                sound.play()  # Alarm is triggered

            except:  # isplaying = False
                pass
            if(thicc < 16):
                thicc = thicc+2
            else:
                thicc = thicc-2
                if(thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        cv2.imshow('Drowsiness detector', frame)  # Show video feed on display
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit key i.e. 'Q'
            break
    cap.release()
    cv2.destroyAllWindows()  # All windows are closed
    root.deiconify()


def f2():  # Function to save snapshot of detection in 'Snaps' directory
    filename = filedialog.askopenfilename(initialdir=r"C:\Users\Akshay\DrowsinessDetection\Snaps",
                                          title="Select a File",
                                          filetypes=(("Images", "*.jpg*"),
                                                     ("all files", "*.*")))
    img = PIL.Image.open(filename)
    img.thumbnail((350, 350))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)  # Show the selected image in Tkinter window
    lbl.image = img


# GUI Tkinter
root = Tk()
root.title("Drowsiness Detection")
root.geometry("800x800")
root.iconbitmap("drive.ico")
root.configure(bg='aquamarine')
bg = PhotoImage(file="test.png")  # importing background image
img = ImageTk.PhotoImage(PIL.Image.open("logo.png"))  # importing college logo


mycanvas = Canvas(root, width=800, height=800)
mycanvas.pack(fill="both", expand=True)


# Applying bg image in Tkinter window
mycanvas.create_image(0, 0, anchor=NW, image=bg)
# Applying College logo in Tkinter window
mycanvas.create_image(80, 30, image=img)


title = Label(mycanvas, text='Drowsiness Detection System \nUsing Computer Vision', font=(
    'Raleway', 30, 'italic', 'bold',), justify='center', fg="Maroon")  # Project Title label
title.configure(bg='thistle3')
title.pack()
mycanvas.create_window(425, 150, window=title)


btstart = Button(root, text='START', font=(
    "Raleway", 20, 'italic'), bg="thistle3", command=f1)  # Start Button Configuration

btbrowse = Button(root, text="PREVIOUS DETECTIONS", font=(
    "Raleway", 20, 'italic'), bg="thistle3", command=f2)  # Button to view detections made previously

btexit = Button(root, text="EXIT", font=(
    "Raleway", 20, 'italic'), bg="thistle3", command=root.destroy)  # Exit button configuration

btstart_window = mycanvas.create_window(250, 300, anchor=NW, window=btstart)
btbrowse_window = mycanvas.create_window(250, 380, anchor=NW, window=btbrowse)
btexit_window = mycanvas.create_window(510, 300, anchor=NW, window=btexit)

lbl = Label(mycanvas)  # Selected image displaying config.
lbl.pack()
mycanvas.create_window(425, 600, window=lbl)

root.mainloop()  # End of Tkinter window

