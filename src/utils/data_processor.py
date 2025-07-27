import numpy as np
import csv

def smooth_coordinates(coords, window_size=5):
    if len(coords) < window_size:
        return coords
    return np.convolve(coords, np.ones(window_size)/window_size, mode='valid')

def normalize_coordinates(coords, frame_shape):
    h, w = frame_shape[:2]
    return [(x/w, y/h) for x, y in coords]

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def export_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def filter_outliers(data, threshold=2.0):
    return data
