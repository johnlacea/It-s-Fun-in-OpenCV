#Programmer  : JOHN B. LACEA
#Place       : Bilis, Burgos, La Union 2510
#Date        : July 7, 2022
#Description : Hand Detection using CVZONE library
#              You must install the following libraries:
#              1. python -m pip install -U pip  # Update the pip package manager
#              2. pip install numpy
#              3. pip install opencv-python
#              4. pip install mediapipe
#              5. pip install cvzone
#              
#              Run the Program: python HandDetection.py
#              You are hereby to use this code for free. God Bless Us!

import cv2
import time
import ctypes
import cv2
from cvzone.HandTrackingModule import HandDetector

# Global Variables
print("+++ Hand Detection +++")

# Get the Desktop Screen Resolution
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print("Desktop Screen Resolution: {}x{}".format(w,h))

window_title = "Hand Detection"
prev_frame_time = 0
new_frame_time = 0

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
        
        # Instantiate the HandDetector object to detect two(2) hands
        detector = HandDetector(maxHands=2)  
        
        while True:
            # Get the start time in milliseconds for computing the fps
            new_frame_time = time.time()
            
            # Get the video frame image
            ret, frame = video_capture.read()
            
            # If video finished or no Video Input then stop the while loop
            if not ret:              
              break            
            
            # Detect your hands and update the video frame image
            hands, frame = detector.findHands(frame)
                              
            # Calculate the frame per second (fps) and display it!
            font = cv2.FONT_HERSHEY_SIMPLEX            
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time         
            fps = int(fps)         
            fps = "fps: {}".format(fps)         
            cv2.putText(frame, fps, (0, height-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA) #BGR color format not RGB
            
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
        video_capture.release()
        cv2.destroyAllWindows()
else:
    print("Unable to open camera!")


exit()   