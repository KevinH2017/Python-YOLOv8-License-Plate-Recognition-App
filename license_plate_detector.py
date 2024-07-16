from ultralytics import YOLO
import cv2
from sort.sort import *
from .util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load the models
model = YOLO("yolov8n.pt") 
license_plate_detector = YOLO("./models/license_plate_detector.pt")       # Custom model created for license plates

# Load video object
cap = cv2.VideoCapture("./videos/license_plate_test.mp4")

# Vehicle ids from yolov8n.pt
vehicles = [2, 3, 5, 7]

# Reads through frames of video object
frame_num = -1
ret = True          # Having ret be a boolean makes the loop, not-infinite
while ret:
    frame_num += 1
    ret, frame = cap.read()

    if ret:
        if frame_num > 10:
            # Loops breaks after processing 10 frames
            break
        results[frame_num] = {}

        # Detects vehicles and create boxes around them
        detections = model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Appends detected vehicles if they match vehicles ids from yolov8n.pt
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates and create boxes around them
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Crops license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Process the license plate
            # Applies a fixed-level threshold to an image, converts all pixels below a certain value to a max value
            # The new threshold image makes it black and white, so that it is easier to process
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # cv2.imshow('original_crop', license_plate_crop)         # Cropped image with license plate
            # cv2.imshow('threshold', license_plate_crop_thresh)      # License plate with modified threshold

            # cv2.waitKey(0)

            # Reads the license plate numbers
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)        # Exits with no error here

            # Only creates bounding boxes and their scores if license plate text is detected
            if license_plate_text is not None:
                results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},                # Bounding box coordinates for the car
                                                'license_plate': {'bbox': [x1, y1, x2, y2],                 # Bounding box coordinates for the license plate
                                                                'text': license_plate_text,                 # The text for the license plate
                                                                'bbox_score': score,                        # The score for the car bounding box
                                                                'text_score': license_plate_text_score}}    # The score for the license plate text bounding box
    
# Writes the results to csv file
write_csv(results, './test.csv')
print("Results written to test.csv")