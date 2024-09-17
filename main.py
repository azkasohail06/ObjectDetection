# from ultralytics import YOLO
# import torch
# import numpy as np
# import cv2
# from time import time

# # model=YOLO('yolov8l-obb.pt')

# # results=model("highway_mini.mp4",show=True)

# # print(results)



# class ObjectDetection:

#     def __init__(self, capture_index):
       
#         self.capture_index = capture_index
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("Using Device: ", self.device)
        
#         self.model = self.load_model()
        
#         self.CLASS_NAMES_DICT = self.model.model.names
    

#     def load_model(self):
#         model=YOLO('yolov8l-obb.pt')
#         return model
        
#     def predict(self,frame):
#         results = self.model(frame)
#         return results
    
#     def __call__(self):
#         cap = cv2.VideoCapture("highway_mini.mp4")
#         assert cap.isOpened()
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
#         while True:
          
#             start_time = time()
            
#             ret, frame = cap.read()
#             assert ret
            
#             results = self.predict(frame)
#             end_time = time()
#             fps = 1/np.round(end_time - start_time, 2)
             
#             cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
#             if cv2.waitKey(5) & 0xFF == 27:
                
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()


# detector = ObjectDetection(capture_index=0)
# detector()


import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import random

class ObjectDetection:

    def load_model(self):
        model = YOLO("yolov8m.pt")  # Load a pretrained YOLOv8 model
        model.fuse()
        return model


    def __call__(self):
        cap = cv2.VideoCapture("2053100-uhd_3840_2160_30fps.mp4")
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_vid=640
        frame_hyt=480
        my_file=open("coco.txt")
        data=my_file.read()
        class_list=data.split("\n")
        my_file.close()
        detection_colors=[]
        for i in range(len(class_list)):
          r=random.randint(0,255)
          g=random.randint(0,255)
          b=random.randint(0,255)
          detection_colors.append((b,g,r))


        while True:
            start_time = time()
            
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (frame_vid, frame_hyt))

            cv2.imwrite("images/frame.png",frame)
            model=self.load_model()
            detect=model.predict(source=[frame],conf=0.45,save=False)

            Dnum=detect[0].numpy()

            if len(Dnum)!=0:
              for i in range(len(detect[0])):
                boxes = detect[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
                )
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                end_time = time()
                fps = 1 / (end_time - start_time)
                cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow('objectdetection', frame)
            if cv2.waitKey(1) == ord("q"):
              break
        
        cap.release()
        cv2.destroyAllWindows()
detector = ObjectDetection()
detector()
