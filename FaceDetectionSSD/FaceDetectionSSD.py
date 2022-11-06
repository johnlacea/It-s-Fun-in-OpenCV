#Programmer  : JOHN B. LACEA
#Place       : Bilis, Burgos, La Union 2510
#Date        : July 7, 2022
#Description : SSD Face Detection on the Image with Text-to-Speech
#              You must install the following libraries:
#              1. python -m pip install -U pip  # Update the pip package manager
#              2. pip install numpy
#              3. pip install opencv-python
#              4. pip install pyttsx3
#              
#  If there is a problem "pythoncom38.dll cannot be found".
#  Made a check for package pypiwin32, which is is actually installed.
#  Following the recommendation here I looked at this folder:
#  C:\Users\JOHN B. LACEA\AppData\Roaming\Python\Python38\site-packages\pywin32_system32
#  I then copied the files "pythoncom38.dll" and "pywintypes38.dll" over to:
#  C:\Users\JOHN B. LACEA\AppData\Roaming\Python\Python38\site-packages\win32\lib 
#              
#              
#              Run the Program: python FaceDetectionSSD.py
#              You are hereby to use this code for free. God Bless Us!

import cv2
import pyttsx3
import threading
import time
import ctypes
import numpy as np

# Global Variables
print("+++ SSD Face Detection on the Image with Text-to-Speech +++")

# Load face prototypes
prototxt_path = "weights/deploy.prototxt.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# Load face Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Get the Desktop Screen Resolution
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print("Desktop Screen Resolution: {}x{}".format(w,h))

window_title = "SSD Face Detection on the Image with Text-to-Speech"
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
faces = 0
facesOld = 0

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
video_capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

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
            
            # preprocess the image: resize and performs mean subtraction
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            # set the image into the input of the neural network
            model.setInput(blob)
            # perform inference and get the result
            output = np.squeeze(model.forward())
            font_scale = 1.0
            faces = 0
            for i in range(0, output.shape[0]):
                # get the confidence
                confidence = output[i, 2]
                # if confidence is above 50%, then draw the surrounding box
                if confidence > 0.5:
                    # get the surrounding box cordinates and upscale them to original image
                    box = output[i, 3:7] * np.array([width, height, width, height])
                    # convert to integers
                    start_x, start_y, end_x, end_y = box.astype(np.int)
                    # draw the rectangle surrounding the face
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
                    # draw text as well
                    cv2.putText(frame, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
                    faces = faces + 1
                    if speak==False:
                        total = faces
                        message = "There is {} face detected!".format(total)
                        if total > 0  and faces >= facesOld:                        
                            if total == 1:
                                item = "There is {} face detected!".format(total)
                                message = item
                            else:
                                item = "There are {} faces detected!".format(total)
                                message = item
                            if item != itemOld:                                                
                                say = item
                                speak=True
                    if faces == 0:
                        item=''
                    itemOld=item                    
                    if(facesOld<faces):
                        facesOld = faces
            
            if faces == 0:                    
                item=''
                facesOld = 0                
                message = "There is 0 face detected!"
            itemOld=item
            if(facesOld>faces):
                facesOld = faces                        
            
                                   
            # Calculate the frame per second (fps) and display it!
            font = cv2.FONT_HERSHEY_SIMPLEX            
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time         
            fps = int(fps)         
            fps = "fps: {}".format(fps)         
            cv2.putText(frame, fps, (0, height-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA) #BGR color format not RGB
            cv2.putText(frame, message, (0, 30), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA) #BGR color format not RGB
            
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