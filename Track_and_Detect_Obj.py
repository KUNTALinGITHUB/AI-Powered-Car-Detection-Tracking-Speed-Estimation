import cv2
import math
from ultralytics import YOLO
from typing import Dict, List, Tuple


def calculate_object_distances(object_paths: Dict[int, List[Tuple[int, int]]]) -> Dict[int, float]:
    object_distances = {}
    for obj_id, path in object_paths.items():
        total_distance = 0.0
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            total_distance += math.hypot(x2 - x1, y2 - y1)
        object_distances[obj_id] = total_distance
    return object_distances


def calculate_object_speeds(
    object_distances: Dict[int, float],
    object_paths: Dict[int, List[Tuple[int, int]]],
    fps: float,
    real_world_length_per_pixel: float
) -> Dict[int, float]:
    object_speeds = {}
    for obj_id, distance_pixels in object_distances.items():
        num_frames = len(object_paths[obj_id])
        time_seconds = num_frames / fps
        distance_real_world = distance_pixels * real_world_length_per_pixel  # in meters
        speed_mps = distance_real_world / time_seconds if time_seconds > 0 else 0
        object_speeds[obj_id] = speed_mps  # meters per second
    return object_speeds


def track_and_count_cars(
    model_path: str,
    video_path: str,
    output_path: str,
    target_class: int = 0,
    real_world_length_per_pixel: float = 0.01  # 1 pixel = 0.05 meters
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    line_x = int(width * 0.5)
    unique_ids = set()
    track_centroids = {}
    object_paths = {}
    car_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)

        results = model.track(frame, persist=True, classes=[target_class], tracker="botsort.yaml")

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == target_class:
                    track_id = int(box.id) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if track_id != -1:
                        prev_cx = track_centroids.get(track_id)
                        track_centroids[track_id] = cx

                        if prev_cx is not None and prev_cx < line_x <= cx and track_id not in unique_ids:
                            car_count += 1
                            unique_ids.add(track_id)

                        if track_id not in object_paths:
                            object_paths[track_id] = []
                        object_paths[track_id].append((cx, cy))

                        # Calculate speed in km/h
                        speed_kmph = 0.0
                        if len(object_paths[track_id]) > 1:
                            distance = calculate_object_distances({track_id: object_paths[track_id]})[track_id]
                            speed_mps = calculate_object_speeds(
                                {track_id: distance},
                                {track_id: object_paths[track_id]},
                                fps,
                                real_world_length_per_pixel
                            )[track_id]
                            speed_kmph = speed_mps * 3.6  # Convert m/s to km/h

                        # Choose color based on speed
                        speed_color = (0, 255, 0) if speed_kmph <= 40 else (0, 0, 255)

                        label = f"Car | ID: {track_id} | {speed_kmph:.1f} km/h"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), speed_color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        cv2.rectangle(frame, (0, 0), (250, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Final distance and speed report
    distances = calculate_object_distances(object_paths)
    speeds = calculate_object_speeds(distances, object_paths, fps, real_world_length_per_pixel)
    for obj_id in distances:
        speed_kmph = speeds[obj_id] * 3.6
        print(f"Object ID {obj_id} traveled {distances[obj_id]:.2f} pixels")
        print(f"Speed of Object ID {obj_id}: {speed_kmph:.1f} km/h")

    print(f"Total cars crossed the line: {car_count}")


# Example usage
model_path = r"D:\project_kuntal\Speed_Detection_main\runs\detect\train3\weights\best.pt"
video_path = r"D:\project_kuntal\Speed_Detection_main\cars.mp4"
output_path = "New_Model_botsort_Track_and_Detect_output_cars_only_part2_speed.mp4"

track_and_count_cars(
    model_path=model_path,
    video_path=video_path,
    output_path=output_path,
    real_world_length_per_pixel=0.01  # 1 pixel = 0.05 meters
)
