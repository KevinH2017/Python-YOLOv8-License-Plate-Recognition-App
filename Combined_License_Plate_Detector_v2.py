# Detects and attempts to predict license plate text by creating a csv file to process
# information and create boundary boxes for both the license plate and their associated car
# and creates an image of the predicted text for the associated license plate.
# Higher quality videos create more accurate predictions.
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import ast, cv2, csv, os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.ttk import *


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """Draws a four corner border around detected object."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def interpolate_bounding_boxes(data):
    """Extract necessary data columns from input data and creates new bounding boxes for smoother output"""
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


# Creates folders inside current directory if they don't exist already
csv_directory = './csv'
video_output = "./video_output"
if not os.path.exists(csv_directory):
    os.mkdir(csv_directory)
    print("Folder %s created!" % csv_directory)
else:
    print("Folder %s already exists!" %csv_directory)

if not os.path.exists(video_output):
    os.mkdir(video_output)
    print("Folder %s created!" % video_output)
else:
    print("Folder %s already exists!" % video_output)

# # Supresses window popup
Tk().withdraw()
# Show an "Open" dialog box and returns the absolute path to the selected file
filename = askopenfilename() 

results = {}

mot_tracker = Sort()

# Load the models
model = YOLO("yolov8n.pt") 
license_plate_detector = YOLO("./models/license_plate_detector.pt")       # Custom model created for license plates

# Load video object
cap = cv2.VideoCapture(filename)

# Vehicle ids from yolov8n.pt; car, motorcycle, bus, truck
vehicles = [2, 3, 5, 7]

# Reads through frames of video object
frame_num = -1
ret = True          # Having ret be a boolean makes the loop, not-infinite
while ret:
    frame_num += 1
    ret, frame = cap.read()

    if ret:
            
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

            # Processes each frame until no more detected cars are found
            if car_id != -1:
                # Crops license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process the license plate
                # Applies a fixed-level threshold to an image, converts all pixels below a certain value to a max value
                # The new threshold image makes it black and white, so that it is easier to process
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Reads the license plate numbers
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Only creates bounding boxes and their scores if license plate text is detected
                if license_plate_text is not None:
                    results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},                # Bounding box coordinates for the car
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],                 # Bounding box coordinates for the license plate
                                                                    'text': license_plate_text,                 # The text for the license plate
                                                                    'bbox_score': score,                        # The score for the car bounding box
                                                                    'text_score': license_plate_text_score}}    # The score for the license plate text bounding box


# Creates new file names if the name already exists
filename_licensePlates = "licensePlates"
counter_1 = 0
while os.path.exists(f"./csv/{filename_licensePlates}_{counter_1}.csv"):
    counter_1 += 1

# Writes the results to csv file
write_csv(results, f'./csv/{filename_licensePlates}_{counter_1}.csv')
print(f"Created {filename_licensePlates}_{counter_1}.csv!")
print(f"Results written to {filename_licensePlates}_{counter_1}.csv!")
print(f"Processing {filename_licensePlates}_{counter_1}.csv...")

### Interpolates and creates Bounding Boxes ###

# Load the CSV file
with open(f'./csv/{filename_licensePlates}_{counter_1}.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Creates new file names if the name already exists
filename_interpolated = "licensePlates_interpolated"
counter_2 = 0
while os.path.exists(f"./csv/{filename_interpolated}_{counter_2}.csv"):
    counter_2 += 1

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open(f'./csv/{filename_interpolated}_{counter_2}.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

print(f"{filename_interpolated}_{counter_2}.csv created!")
print(f"Processing {filename_interpolated}_{counter_2}.csv...")
print(f"Creating {filename_interpolated}_{counter_2}.mp4...")

### Processes licensePlates_interpolated.csv to create output mp4 that has license plate text in the video

results = pd.read_csv(f'./csv/{filename_interpolated}_{counter_2}.csv')

# Loads video
cap = cv2.VideoCapture(filename)

# Creates new output mp4 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Creates new file names if the name already exists
interpolated_video = "license_plates_output"
counter_3 = 0
while os.path.exists(f"./video_output/{interpolated_video}_{counter_3}.mp4"):
    counter_3 += 1

out = cv2.VideoWriter(f'./video_output/{interpolated_video}_{counter_3}.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Reads frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Draws car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draws license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Crops license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()

print(f"{interpolated_video}_{counter_3}.mp4 created!")
print("Check video_output folder for video!")