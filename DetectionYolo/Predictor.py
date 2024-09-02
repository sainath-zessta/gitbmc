# Predictor.py
from datetime import datetime
import math

def get_predicted_count(time, vehicle_class, count):
    scaling_factors = {
        "Bicycle": [2.1, 0.6, 1.3, 1.1, 1.1, 1.3, 0.7, 1.0, 1.4, 0.7, 1.0, 1.0],
        "Bus": [1.0, 0.7, 1.0, 1.2, 0.8, 0.9, 1.2, 0.8, 1.1, 1.2, 1.0, 1.0],
        "Car": [1.3, 1.1, 1.1, 1.1, 1.2, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
        "LCV": [1.0, 0.8, 1.3, 0.9, 1.1, 1.1, 1.0, 1.0, 1.1, 0.9, 1.1, 1.0],
        "Three-Wheeler": [1.4, 1.2, 1.0, 1.1, 1.3, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Truck": [0.8, 1.2, 1.1, 0.7, 1.3, 0.9, 1.2, 0.9, 1.1, 1.2, 0.5, 1.0],
        "Two-Wheeler": [1.4, 1.1, 1.2, 1.2, 1.2, 1.0, 1.0, 0.9, 1.1, 0.9, 1.0, 1.0]
    }

    # Convert the time string to integers, ignoring seconds
    hour, minute, _ = map(int, time.split(":"))

    time_intervals = [
        ("07:30", "07:45"),
        ("07:45", "08:00"),
        ("08:00", "08:15"),
        ("08:15", "08:30"),
        ("08:30", "08:45"),
        ("08:45", "09:00"),
        ("09:00", "09:15"),
        ("09:15", "09:30"),
        ("09:30", "09:45"),
        ("09:45", "10:00"),
        ("10:00", "10:15"),
        ("10:15", "10:30")
    ]

    # Slice off the seconds to match the time format
    input_time = datetime.strptime(time[:-3], "%H:%M")

    # Find the correct interval
    for index, (start, end) in enumerate(time_intervals):
        start_time = datetime.strptime(start, "%H:%M")
        end_time = datetime.strptime(end, "%H:%M")
        
        if start_time <= input_time < end_time:
            break

    # Apply the scaling factor based on the vehicle class and time interval
    for vehicle, scale in scaling_factors.items():
        if vehicle == vehicle_class:
            count = math.ceil(count * scale[index])

    return count, vehicle_class
