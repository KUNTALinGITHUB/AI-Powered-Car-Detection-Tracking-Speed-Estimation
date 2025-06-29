# AI-Powered-Car-Detection-Tracking-Speed-Estimation



# 🚗 Car Tracking, Counting & Speed Detection using YOLO and BoT-SORT

This project uses the YOLO object detection model and BoT-SORT tracking algorithm to detect, track, count, and estimate the speed of cars in a video. It draws bounding boxes around detected vehicles, calculates their real-world speed, and counts how many cross a virtual line.


---

## 📦 Features

- ✅ Detects and tracks cars using YOLO + BoT-SORT
- ✅ Counts cars crossing a virtual line
- ✅ Estimates real-world speed (in km/h) based on pixel movement
- ✅ Annotates video with car ID, speed, and bounding boxes
- ✅ Highlights speeding cars in red (speed > 40 km/h), others in green
- ✅ Outputs annotated video and prints summary stats

---

## 🧠 How It Works

1. Loads a trained YOLO model.
2. Processes each frame of the input video.
3. Tracks detected cars using BoT-SORT.
4. Calculates distance traveled in pixels and converts it to meters.
5. Estimates speed using frame rate and real-world scale.
6. Annotates each car with ID and speed.
7. Counts cars crossing a vertical line.

---

## 📁 Project Structure

```
├── track_and_detect.py        # Main script
├── botsort.yaml               # Tracker config
├── cars.mp4                   # Input video
├── best.pt                    # Trained YOLO model
├── output_video.mp4           # Annotated output
├── README.md                  # This file
```

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- tqdm (optional for progress bars)

Install dependencies:

```bash
pip install opencv-python ultralytics
```

---

## 🚀 Usage

### 1. Prepare Your Files

- Place your trained YOLO model (e.g., best.pt) in the project folder.
- Place your input video (e.g., cars.mp4) in the same folder.

### 2. Run the Script

```bash
python track_and_detect.py
```

Or customize parameters:

```python
track_and_count_cars(
    model_path="runs/detect/train3/weights/best.pt",
    video_path="cars.mp4",
    output_path="output_video.mp4",
    real_world_length_per_pixel=0.05  # Adjust based on calibration
)
```

---

## 📏 Calibrating Real-World Scale

To get accurate speed estimates, you must set the correct real-world length per pixel:

```python
real_world_length_per_pixel = real_world_distance_in_meters / pixel_distance
```

Example: If a lane is 3.5 meters wide and spans 70 pixels in the video:

```python
real_world_length_per_pixel = 3.5 / 70 = 0.05
```

---

## 🎨 Output Example

- Green box: Car speed ≤ 40 km/h
- Red box: Car speed > 40 km/h
- Label: `Car | ID: 3 | 38.2 km/h`

---

## 📊 Sample Output (Console)

```
Object ID 3 traveled 120.45 pixels
Speed of Object ID 3: 38.2 km/h
Object ID 5 traveled 210.12 pixels
Speed of Object ID 5: 52.7 km/h
Total cars crossed the line: 12
```

---

## 📌 Notes

- The accuracy of speed estimation depends on:
  - Camera angle and height
  - Frame rate (FPS)
  - Real-world calibration
- This script assumes a fixed vertical line for counting cars moving left to right.

---

## 📜 License

This project is open-source and available under the [MIT License].

https://github.com/user-attachments/assets/3111a643-0804-4576-9f42-e51cbed306d1



---

## 🙌 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [BoT-SORT Tracker](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- OpenCV for video processing

---

Let me know if you'd like a sample video, demo GIF, or badge icons added to the README!
