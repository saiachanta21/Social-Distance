import cv2 # used to manipulate images and videos.
import imutils #to make basic image processing functions such as translation,rotation,resizing,displaying Matplotlib image ,detecting edges.
import datetime 
import numpy as np # is used to store arrays
from itertools import combinations #s a tool from the itertools module that allows you to generate all possible combinations of a given iterable, such as a list or a tuple.
import math #provides a set of mathematical functions and constants.
from scipy.spatial import distance as dist
from collections import OrderedDict
protopath = "MobileNetSSD_deploy.prototxt" #contain a list of the network layers in the model 
modelpath = "MobileNetSSD_deploy.caffemodel" #contains the weights of the model.
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)#we will pass these two files as arguments to the cv2.dnn.readNetFromCaffe module to create our model and load the model.


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# import the necessary packages

class Centroidcode:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
    def register(self, centroid, inputRect):
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.bbox
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])
        return self.bbox


tracker = Centroidcode(maxDisappeared=10, maxDistance=200)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0: # Check if the input array of bounding boxes is empty
            return []
  # Convert the data type of the bounding boxes to float if it's not already
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
# Initialize an empty list to hold the indices of the bounding boxes that will be kept
        pick = []
# Extract the x and y coordinates of the top-left and bottom-right corners of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
# Calculate the area of each bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)  # Sort the bounding boxes by their bottom coordinate (y2)
 # Continue selecting bounding boxes with high y2 coordinates and removing any overlapping boxes
        while len(idxs) > 0:  # Select the bounding box with the highest y2 coordinate
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
 # Calculate the overlap between the selected box and all remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
 # Remove any boxes with an overlap greater than the specified threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
 # Return the subset of bounding boxes that were selected
        return boxes[pick].astype("int")
    except Exception as e: # Return the subset of bounding boxes that were selected
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    cap = cv2.VideoCapture('videosai.mp4')
# setting up the frame rate and other fps-related parameters
    fps_start_time = datetime.datetime.now()
    fps = 0
    count=0
    count_pic=0
   # count_off=0
    total_frames = 0
# running a loop to read the frames of the video
    while cap.isOpened():
        ret, frame = cap.read() # reading the current frame of the video
        if not ret:
            break
        frame = imutils.resize(frame, width=1200) # resizing the frame using the imutils library
        total_frames = total_frames + 1 # incrementing the total_frames count
# getting the height and width of the frame
        (H, W) = frame.shape[:2]
# creating a blob from the current frame using the dnn module of cv2
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
# setting the input to the detector model and detecting persons in the frame
        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []   # initializing an empty list to store the bounding boxes
        count_ppl=0
        for i in np.arange(0, person_detections.shape[2]): # iterating through all the detections and storing the bounding boxes with confidence score greater than 0.5
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)
                count_ppl+=1
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        boundingboxes = np.array(rects) # converting the list of bounding boxes into numpy array and calling the non-maximum suppression function
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.5)
        centroid_dict = dict() # creating a dictionary to store the centroids of the detected persons
        objects = tracker.update(rects)  # updating the tracker with the latest set of bounding boxes and getting the objects
        for (objectId, bbox) in objects.items(): # iterating through all the objects and drawing their bounding boxes and ids
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

# adding the centroid coordinates to the centroid_dict
            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)
        # drawing the text containing the object id above the object's bounding box
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        close_person=""
        red_zone_list = [] # Create an empty list to store information about objects in the "red zone".
        off=0
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):# Loop through all possible pairs of objects using itertools.combinations.

            dx, dy = p1[0] - p2[0], p1[1] - p2[1] # Calculate the distance between the centroids.
            distance = math.sqrt(dx * dx + dy * dy) # Use the Equlidean theorem to calculate the distance.
            print("P",id1+1,"- P",id2+1,"=",distance)
            if distance < 90.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    off+=1
                    count+=1
                    close_person+="Person "+str(id1+1)+" and Person "+str(id2+1)+" "
                    close_person+="are not following social distancing"
                    print(close_person) 
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
        for id, box in centroid_dict.items():
            if id in red_zone_list:
                 cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) # red rectangle for objects outside the red zone
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # green rectangle for objects outside the red zon
# Draw rectangles around the detected objects based on whether they are in the red zone or not
        if count>=5:
            print("FRAME "+str(count_pic)+"    People Count : "+str(count_ppl)+"   RL : "+str(off))
            cv2.imwrite('dataset\\frame'+str(count_pic)+'.png',frame)   # Saving frames in Main Folder
            count_pic+=1 
            count=0
            off=0

#it calculates the FPS by dividing the total number of frames (total_frames) by the number of seconds in the time difference.
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1) # the FPS text on the frame at the specified location, using the specified font, font scale, color, and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame,"Social Distancing Monitoring System ", (300, 45),font, 1, (128, 0, 0), 2)#(50,45)
        cv2.putText(frame, "Batch Number 15", (900, 45),font, 1, (0, 255, 255), 2) #(150,85)
        cv2.putText(frame, "Bounding box shows the level of risk to the person. ", (45, 90),font, 1, (128, 0, 128), 2) #(45,120)      
        cv2.putText(frame, "RED: DANGER", (45, 180),font, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (975, 60), (20, 200), (0, 0, 0), 2)            
        cv2.putText(frame, "GREEN: CONGRATULATIONS YOU ARE SAFE", (45, 140),font, 1, (0, 255, 0), 2)
        cv2.imshow("Application", frame) # Show the frame in a window named "Application".
        if cv2.waitKey(1)& 0xFF == ord('q'): # If the "q" key is pressed, break out of the loop.
            break
    cap.release()
    cv2.destroyAllWindows() # Destroy all windows created by OpenCV.
    cv2.waitKey(1)

main() # Call the "main" function to start the program.
