from ultralytics import YOLO
import cv2
import os
import argparse
import datetime
import yt_dlp as ydlp

def format_timestamp_for_filename(ts):
    """ Format a timedelta timestamp for use in filenames. """
    total_seconds = int(ts.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{seconds:02d}"

def process_frame(frame, frame_number, fps, model, log_file, detection_folder):
    """ Process each frame for pothole detection and handling. """
    results = model.track(frame, conf=0.3, persist=True, show=True, save_txt=True)

    # Check for pothole detections in the first result
    for i, detection in enumerate(results[0].boxes.xyxy):
        class_id = int(results[0].boxes.cls[i])  # Get class ID
        if model.names[class_id] == 'pothole':
            # Get timestamp
            timestamp = datetime.timedelta(seconds=frame_number / fps)
            formatted_timestamp = format_timestamp_for_filename(timestamp)
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection)
            # Log detection
            log_file.write(f"Time: {timestamp}, Location: ({x1}, {y1}), ({x2}, {y2})\n")

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Pothole: {model.names[class_id]}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the frame with pothole detection
            frame_filename = os.path.join(detection_folder, f"pothole_detected_{formatted_timestamp}.jpg")
            cv2.imwrite(frame_filename, frame)

    # Return the frame with the results plotted
    return results[0].plot()

def download_youtube_video(url):
    """ Retrieve a video from YouTube. """
    ydl_opts = {'format': 'best[ext=mp4]'}
    with ydlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        return info_dict['url']
    
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='last.pt', help='model weights path')
    parser.add_argument('--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--source', type=str, default='0', help='source for the video stream')
    args = parser.parse_args()

    # Check if the source is a YouTube URL and download it if it is
    if args.source.startswith('http'):
        # Download the YouTube video and use the local file as the video source
        print(f"Downloading YouTube video from {args.source}")
        video_source = download_youtube_video(args.source)
    else:
        # Convert source to int if it's a digit (webcam)
        video_source = int(args.source) if args.source.isdigit() else args.source

    # Load model (YOLOv8)
    model_path = os.path.join('.', 'runs', 'detect', 'train38', 'weights', args.weights)
    model = YOLO(model_path)

    # Open the video stream using the source argument
    cap = cv2.VideoCapture(video_source)

    # File to log pothole detections
    log_file = open("pothole_detections.txt", "a")

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Getting frames per second of the video
    detection_folder = "detected_frames"  # Folder to store detected frames
    os.makedirs(detection_folder, exist_ok=True)  # Create the folder if it doesn't exist

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        processed_frame = process_frame(frame, frame_number, fps, model, log_file, detection_folder)

        cv2.imshow('frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # changed waitKey to 1 for faster response
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
