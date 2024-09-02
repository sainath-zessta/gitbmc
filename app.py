# app.py
from DetectionYolo.Detector import detect_track
from DetectionYolo.TimeTracker import detect_time
from DetectionYolo.Predictor import get_predicted_count

import sys
from collections import defaultdict
from utils.color import colorText, colorSingleLine
from utils.anim import load_animation, animate
import json
import os
import random
import math


def add_gaussian_noise(count,time_data):
    print(time_data,"*******")
    if count > 0:
        # Calculate variance
        var = count / 10
        # Generate Gaussian noise
        noise = random.gauss(0, var)
        # Ensure predicted count is non-negative
        return max(math.ceil(count + noise), 0)
    else:
        # For zero counts, ensure prediction remains zero
        return 0

def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k.replace("-", " "): convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k.replace("-", " "): convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [convert_defaultdict_to_dict(i) for i in d]
    return d 

def aggregate_vehicle_counts(data_list):
    
    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for camera_id, video_id, counts in data_list:
        for region, vehicles in counts.items():
            for vehicle_type, count in vehicles.items():
                aggregated_data[camera_id][region][vehicle_type] += count

    # Convert defaultdict to regular dict for the final output
    return convert_defaultdict_to_dict(dict(aggregated_data))

if __name__ == "__main__":
    colorText("utils/text_files/logo.txt")
    # load_animation(5)
   
    print("App Running " + sys.argv[0])
    ip_json = sys.argv[1]
    op_json = sys.argv[2]
        
    ip_dict={}
    with open(ip_json) as f:
        ip_dict = json.load(f)
        print(ip_dict)



    jsonList = []

# for camera_id, video_list in ip_dict.items():
#     camera_object_json_path = os.path.join('jsons', camera_id) + '.json'
#     with open(camera_object_json_path) as camera_json:
#         camera_json_data_object = json.load(camera_json)
#         print(camera_json_data_object)
#         regions = camera_json_data_object['regions']
#         turning_pat = camera_json_data_object['turning_logics']

#         for vid_count, vid_path in video_list.items():
#             print('Current File', camera_id, vid_path)
#             try:
#                 time_data = detect_time(vid_path)  # Calculate time-related data
#                 print(time_data)
#             except:
#                 time_data=1.1

#             current_data = (camera_id, vid_count, detect_track(vid_path, regions, turning_pat))

#             print("done", vid_path)
#             jsonList.append((current_data, time_data))



for camera_id, video_list in ip_dict.items():
    camera_object_json_path = os.path.join('jsons', camera_id) + '.json'
    with open(camera_object_json_path) as camera_json:
        camera_json_data_object = json.load(camera_json)
        print(camera_json_data_object)
        regions = camera_json_data_object['regions']
        turning_pat = camera_json_data_object['turning_logics']

        for vid_count, vid_path in video_list.items():
            print('Current File', camera_id, vid_path)
            time_data = None

            try:
                time_data = detect_time(vid_path)  # Calculate time-related data
            except Exception as e:
                print(f"Error detecting time for video: {vid_path}, error: {e}")

            if time_data is None or time_data == "00:00:00":
                try:
                    video_name = os.path.splitext(os.path.basename(vid_path))[0]
                    date_time = video_name.split("_2")[1]
                    time_data = date_time[10:18].replace("_", ":")
                except Exception as e:
                    print(f"Error parsing time from filename: {vid_path}, error: {e}")
                    time_data = "00:00:00"
            print(time_data)
            current_data = (camera_id, vid_count, detect_track(vid_path, regions, turning_pat))

            print("RAN COMPLETED", vid_path)
            jsonList.append((current_data, time_data))

data = aggregate_vehicle_counts([item[0] for item in jsonList])
final_data = {}
for (cam_id, regions), (_, time_data) in zip(data.items(), jsonList):
    cumulative_counts = {region: counts for region, counts in regions.items()}

    predicted_counts = {}
    for region, counts in regions.items():
        region_predicted_counts = {}
        for v_tp,vehicle_counts in counts.items():
            if time_data == "00:00:00" or time_data is None:
                print("******* GUASS TRIGGERED *******")
                pred_count = add_gaussian_noise(vehicle_counts, time_data)
            else:
                print("******* TIME TRIGGERED *******")

                pred_count,vehicle = get_predicted_count(time_data,v_tp,vehicle_counts)
        # predicted_counts[region] = tp_temp_dict_cont_prd              # Pass time_data here
            region_predicted_counts[v_tp] = pred_count
            predicted_counts[region] = region_predicted_counts



    final_data[cam_id] = {
        'Cumulative Counts': cumulative_counts,
        'Predicted Counts': predicted_counts
    }

with open(op_json, "w") as outfile:
    json.dump(convert_defaultdict_to_dict(final_data), outfile)

print(final_data)



