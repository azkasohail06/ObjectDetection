import cv2
import time
from datetime import datetime
import os
import numpy as np

# CLASSIFIER FOR DETECTING CARS--------------------------------------------------
carCascade = cv2.CascadeClassifier('files/HaarCascadeClassifier.xml')  # Ensure path is correct

# TAKE VIDEO---------------------------------------------------------------------
video = cv2.VideoCapture('files/videoTest.mp4')  # Ensure path is correct

WIDTH = 1280  # WIDTH OF VIDEO FRAME
HEIGHT = 720  # HEIGHT OF VIDEO FRAME
cropBegin = 240  # CROP VIDEO FRAME FROM THIS POINT
fpsFactor = 3  # TO COMPENSATE FOR SLOW PROCESSING
speedLimit = 20  # SPEED LIMIT
startTracker = {}  # STORE STARTING TIME OF CARS
carPositions = {}  # STORE CAR POSITIONS BETWEEN FRAMES

# MAKE DIRECTORY TO STORE OVER-SPEEDING CAR IMAGES
if not os.path.exists('overspeeding/cars/'):
    os.makedirs('overspeeding/cars/')

print('Speed Limit Set at 20 Kmph')

def saveCar(speed, image):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    link = 'overspeeding/cars/' + nameCurTime + '.jpeg'
    cv2.imwrite(link, image)

# FUNCTION TO CALCULATE SPEED----------------------------------------------------
def estimateSpeed(carID, displacement, timeDiff):
    speed = round(displacement / timeDiff * fpsFactor * 3.6, 2)  # Convert from meters per second to km/h
    return speed

# FUNCTION TO ESTIMATE SPEED BASED ON CAR MOVEMENT----------------------------
def speedEstimation():
    frameCounter = 0
    previousFrame = None
    carTracker = {}

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            print("End of video or video read error.")
            break

        frameTime = time.time()
        image = cv2.resize(image, (WIDTH, HEIGHT))[cropBegin:720, 0:1280]
        resultImage = image.copy()

        # Convert the image to grayscale for car detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect cars in the frame
        cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))  # DETECT CARS

        if len(cars) == 0:
            print("No cars detected in this frame.")
        
        for (_x, _y, _w, _h) in cars:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)

            # PUT BOUNDING BOXES-------------------------------------------------
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate speed based on position change between frames
            carID = str(x) + str(y)  # Simple car ID based on position (can be improved)

            # If this is the first frame for this car, initialize its position and time
            if carID not in carPositions:
                carPositions[carID] = {"last_position": (x + w / 2, y + h / 2), "last_time": frameTime}

            # Calculate displacement and speed if car was detected previously
            if carID in carPositions:
                last_position = carPositions[carID]["last_position"]
                last_time = carPositions[carID]["last_time"]
                
                # Calculate displacement (distance between old and new position)
                displacement = np.linalg.norm(np.array([x + w / 2, y + h / 2]) - np.array(last_position))
                
                # Calculate time difference between frames
                time_diff = frameTime - last_time
                # Calculate speed
                if time_diff > 0:
                    speed = estimateSpeed(carID, displacement, time_diff)
                    if speed > speedLimit:
                        print(f"CAR-ID: {carID} - {speed} km/h - OVERSPEED")
                        saveCar(speed, image[y:y + h, x:x + w])
                    else:
                        
                        print(f"CAR-ID: {carID} - {speed} km/h")
                    cv2.putText(resultImage, f"Speed: {speed} km/h", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


                # Update the car's last position and time
                carPositions[carID] = {"last_position": (x + w / 2, y + h / 2), "last_time": frameTime}

        # DISPLAY EACH FRAME
        cv2.imshow('result', resultImage)

        if cv2.waitKey(33) == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    speedEstimation()
