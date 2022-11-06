#Programmer  : JOHN B. LACEA
#Place       : Bilis, Burgos, La Union 2510
#Date        : July 7, 2022
#Description : Face Detection and Age Prediction on the Image with Text-to-Speech
#              You must install the following libraries:
#              1. python -m pip install -U pip  # Update the pip package manager
#              2. pip install numpy
#              3. pip install opencv-python
#              4. pip install pyttsx3
#              5. pip install filetype
#              
#  If there is a problem "pythoncom38.dll cannot be found".
#  Made a check for package pypiwin32, which is is actually installed.
#  Following the recommendation here I looked at this folder:
#  C:\Users\JOHN B. LACEA\AppData\Roaming\Python\Python38\site-packages\pywin32_system32
#  I then copied the files "pythoncom38.dll" and "pywintypes38.dll" over to:
#  C:\Users\JOHN B. LACEA\AppData\Roaming\Python\Python38\site-packages\win32\lib 
#              
#              
#              Run the Program: python FaceDetectionAgePrediction.py
#              You are hereby to use this code for free. God Bless Us!

import cv2
import pyttsx3
import threading
import time
import ctypes
import filetype
import numpy as np

# Global Variables
print("+++ Face Detection and Age Prediction on the Image with Text-to-Speech +++")
# Age Prototype
AGE_MODEL = 'weights/deploy_age.prototxt'
# The age model pre-trained weights
AGE_PROTO = 'weights/age_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illumination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['0 and 2', '4 and 6', '8 and 12', '15 and 20','25 and 32', '38 and 43', '48 and 53', '60 and 100']
# Fade Prototype
FACE_PROTO = "weights/deploy.prototxt.txt"
# The face model pre-trained weights
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Get the Desktop Screen Resolution
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print("Desktop Screen Resolution: {}x{}".format(w,h))

window_title = "Face Detection and Age Prediction on the Image with Text-to-Speech"
prev_frame_time = 0
new_frame_time = 0

running = True
speak = True
item = 'It is fun to learn Open Computer Vision using Python Programming Language.'
say = item
itemOld = ''
confidence = 0
engine = None
message = ''

def Speak():
    global speak, say, running, engine
    while running:
        if speak == True:
            engine.say(say)
            engine.runAndWait()
            speak=False

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("voice", "english+m1")    

# 0 => default or built-in Web Camera; 1 => USB Web Camera.
# Sometimes, you need to restart your computer upon connecting your USB Web Camera.
camera_id = 1
video_capture = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)

# My A4Tech USB Web Camera supports 320x240,640x480,1024x768,1280x720,1366x768 and
# 1920x1080 at 30 fps in Windows 7
# It will display a black window if the supplied frame width and frame height
# are not supported by your Web Camera.
# It slows down the frame rate when you supply high resolution!
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)  
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction Caffe model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    """Returns the box coordinates of all detected faces"""
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# Resize an image without distortion
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)

if video_capture.isOpened():
    try:
        # Create a window and place at the center or at the left top of the screen
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        if not(width>w and height>h):
          cv2.moveWindow(window_title,(w//2)-width+(width//2),(h//2)-height+(height//2))
        else:
          cv2.moveWindow(window_title,0,0)
        
        # Run the method Speak() in a separate thread
        x = threading.Thread(target=Speak, daemon=True)
        x.start()
                
        while True:
            # Get the start time in milliseconds for computing the fps
            new_frame_time = time.time()
            
            # Get the video frame image
            ret, frame = video_capture.read()
            
            # If video finished or no Video Input then stop the while loop
            if not ret:              
              break
            
            # Take a copy of the initial image and resize it
            frame = frame.copy()
            if frame.shape[1] > width:
                frame = image_resize(frame, width=width)
                
            faces = get_faces(frame)
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]
                # image --> Input image to preprocess before passing it through our dnn for classification.
                blob = cv2.dnn.blobFromImage(
                    image=face_img, scalefactor=1.0, size=(227, 227), 
                    mean=MODEL_MEAN_VALUES, swapRB=False
                )
                # Predict Age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                i = age_preds[0].argmax()
                age = AGE_INTERVALS[i]
                age_confidence_score = age_preds[0][i]
                # Draw the box
                label = f"Age between {age} - {age_confidence_score*100:.2f}%"                
                # get the position where to put the text
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                # write the text into the frame
                cv2.putText(frame, label, (start_x, yPos),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                # draw the rectangle around the face
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
               
                if speak==False:
                    total = len(faces)                
                    message = "There is {} face detected!".format(total)
                    if total > 0 and age_confidence_score > 0.70:
                        if total == 1:
                            item = "There is {} face detected! The predicted age is between {}.".format(total, age)
                            message = item
                        else:
                            item = "There are {} faces detected! The predicted age is between {}.".format(total, age)
                            message = item
                        if item != itemOld:                                                
                            say = item
                            speak=True
                if len(faces) == 0:                    
                    item=''
                itemOld=item
            
            if len(faces) == 0:                    
                item=''
                message = "There is 0 face detected!"
            itemOld=item
                                   
            # Calculate the frame per second (fps) and display it!
            font = cv2.FONT_HERSHEY_SIMPLEX            
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time         
            fps = int(fps)         
            fps = "fps: {}".format(fps)         
            cv2.putText(frame, fps, (0, height-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA) #BGR color format not RGB
            cv2.putText(frame, message, (0, 30), font, 0.58, (0, 255, 0), 2, cv2.LINE_AA) #BGR color format not RGB
            
            # Determine if the user click the window's close(X) button
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:                
                break
            else:
                cv2.imshow(window_title, frame) # Display the video image inside the window
            
            # Determine if the user press a keyboard key
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):                
                break
    finally:
        say="Thank you for watching. God Bless Us. Goodbye."
        speak=True
        time.sleep(5)
        # Terminate the thread
        speak = False
        running = False
        video_capture.release()
        cv2.destroyAllWindows()
else:
    print("Unable to open camera!")


engine.stop()   
exit()   