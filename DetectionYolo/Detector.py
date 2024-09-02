# Detector.py
import json
import csv
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import ast

# Load the YOLO model
model1 = YOLO("Models/zm.pt")
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model1.to(device)

def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k.replace("-", " "): convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k.replace("-", " "): convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [convert_defaultdict_to_dict(i) for i in d]
    return d 

# Initialize YOLO models
def detect_track(video_path, regions_param, turning_pat_param):
    # print(video_path, regions_param, turning_pat_param)
    
    cap = cv2.VideoCapture(video_path)
    regions = {key: ast.literal_eval(value) for key, value in regions_param.items()}
    patterns = {ast.literal_eval(key): value for key, value in turning_pat_param.items()}

    class_names_model1 = {0: 'Bicycle', 1: 'Bus', 2: 'Car', 3: 'LCV', 4: 'Three-Wheeler', 5: 'Truck', 6: 'Two-Wheeler'}

    class_colors = {
        'Bicycle': (255, 0, 0),
        'Bus': (0, 255, 0),
        'Car': (0, 0, 255),
        'LCV': (255, 255, 0),
        'Three-Wheeler': (255, 0, 255),
        'Truck': (0, 255, 255),
        'Two-Wheeler': (128, 0, 128)
    }

    track_history = defaultdict(lambda: [])
    entry_count = defaultdict(lambda: defaultdict(int))
    vehicle_regions = defaultdict(lambda: defaultdict(int))
    turning_points = defaultdict(lambda: defaultdict(list))
    tracking_data = defaultdict(lambda: {"class_name": None, "start_region": None, "end_region": None})
    last_region = defaultdict(lambda: None)

    def point_in_region(point, region):
        x, y = point
        (x1, y1), (x2, y2) = region

        # Ensure that the coordinates are correctly ordered
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        return x_min <= x <= x_max and y_min <= y <= y_max

    def update_region_counts(track_id, class_name, tracking_trail):
        for center_point in tracking_trail:
            current_region = None
            for region_name, region_coords in regions.items():
                if point_in_region(center_point, region_coords):
                    current_region = region_name
                    if track_id not in vehicle_regions[current_region]:
                        vehicle_regions[current_region][track_id] = {'entry': True}
                        entry_count[current_region][class_name] += 1
                        if not tracking_data[track_id]["start_region"]:
                            tracking_data[track_id]["start_region"] = current_region
                        last_region[track_id] = current_region
                    else:
                        if vehicle_regions[current_region][track_id]['entry']:
                            vehicle_regions[current_region][track_id]['entry'] = False
        tracking_data[track_id]["end_region"] = last_region[track_id]

    class_id_counters = defaultdict(int)

    def generate_class_id(class_name):
        class_id_counters[class_name] += 1
        prefix_map = {
            'Car': 'C',
            'LCV': 'L',
            'Truck': 'T',
            'Three-Wheeler': 'Tr',
            'Two-Wheeler': 'Tw',
            'Bus': 'Bs',
            'Bicycle': 'Bi'
        }
        prefix = prefix_map.get(class_name, 'V')
        return f"{prefix}{class_id_counters[class_name]}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('annotated_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    csv_file = open('tracking_data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Track ID', 'Class ID', 'Class Name', 'Start Region', 'End Region'])

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results1 = model1.track(frame, persist=True, verbose=False,classes=[0, 1, 2, 3, 4, 5, 6], conf=0.7)

            detections = []

            if results1:
                for result in results1:
                    if result is not None:
                        boxes = result.boxes.xywh.cpu()
                        try:
                            ids = result.boxes.id.int().cpu().tolist()
                        except AttributeError:
                            ids = [None] * len(boxes)
                        classes = result.boxes.cls.int().cpu().tolist()
                        for box, track_id, cls in zip(boxes, ids, classes):
                            detections.append((box, track_id, cls))

            annotated_frame = frame.copy()
            for box, track_id, cls in detections:
                x, y, w, h = box
                class_name = class_names_model1.get(cls, 'Unknown')
                color = class_colors.get(class_name, (255, 255, 255))

                track = track_history[track_id]
                track.append((float(x + w / 2), float(y + h / 2)))
                if len(track) > 180:
                    track.pop(0)

                if not tracking_data[track_id]["class_name"]:
                    tracking_data[track_id]["class_name"] = class_name
                    tracking_data[track_id]["class_id"] = generate_class_id(class_name)

                class_id = tracking_data[track_id]["class_id"]

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)

                update_region_counts(track_id, class_name, track)

                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                cv2.putText(annotated_frame, f"{class_id}",
                            (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if len(track) > 1:
                    if "turning_points" not in tracking_data[track_id]:
                        tracking_data[track_id]["turning_points"] = []
                    turning_points[track_id][class_name].append((track[-2], track[-1]))
                    if len(turning_points[track_id][class_name]) > 5:
                        turning_points[track_id][class_name].pop(0)
                    tracking_data[track_id]["turning_points"].append((track[-2], track[-1]))

            for region_name, (start_point, end_point) in regions.items():
                cv2.rectangle(annotated_frame, start_point, end_point, (255, 255, 255), 2)
                cv2.putText(annotated_frame, region_name, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                for class_name, count in entry_count[region_name].items():
                    cv2.putText(annotated_frame, f"{class_name} Entry: {count}",
                                (start_point[0], start_point[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # out.write(annotated_frame)
        else:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    for track_id, data in tracking_data.items():
        class_id = data["class_id"]
        class_name = data["class_name"]
        start_region = data["start_region"]
        end_region = data["end_region"]
        csv_writer.writerow([track_id, class_id, class_name, start_region, end_region])

    csv_file.close()

    def process_entries(start_region, end_region):
        if start_region is None or end_region is None:
            return None
        
        transition_tuple = (start_region, end_region)
        return patterns.get(transition_tuple, f"{start_region}-{end_region}")

    input_csv_file = 'tracking_data.csv'
    output_csv_file = 'processed_tracking_data.csv'

    with open(input_csv_file, 'r') as infile, open(output_csv_file, 'w', newline='') as outfile:
        csv_reader = csv.DictReader(infile)
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Track ID', 'Class ID', 'Class Name', 'Processed Transition'])
        
        for row in csv_reader:
            track_id = row['Track ID']
            class_id = row['Class ID']
            class_name = row['Class Name']
            start_region = row['Start Region']
            end_region = row['End Region']
            processed_transition = process_entries(start_region, end_region)
            csv_writer.writerow([track_id, class_id, class_name, processed_transition])

    pattern_counts = defaultdict(lambda: defaultdict(int))
    with open(output_csv_file, 'r') as infile:
        csv_reader = csv.DictReader(infile)
        for row in csv_reader:
            pattern = row['Processed Transition']
            class_name = row['Class Name']
            if pattern in patterns.values() and class_name in class_names_model1.values():
                pattern_counts[pattern][class_name] += 1

    json_data = defaultdict(lambda: defaultdict(int))
    all_classes = set(class_names_model1.values())
    for pattern in patterns.values():
        json_data[pattern] = {class_name: 0 for class_name in all_classes}
    
    for pattern, class_count in pattern_counts.items():
        for class_name, count in class_count.items():
            json_data[pattern][class_name] += count

    # Ensure all patterns and classes are included in the output
    for pattern in patterns.values():
        if pattern not in json_data:
            json_data[pattern] = {class_name: 0 for class_name in all_classes}

    # Convert defaultdict to dict for JSON serialization
    return convert_defaultdict_to_dict(json_data)
