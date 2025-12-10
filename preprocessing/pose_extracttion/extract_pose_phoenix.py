
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_ROOT = os.path.join(BASE_DIR, "..", "..", "data", "videos_phoenix", "train")
OUTPUT_ROOT = os.path.join(BASE_DIR, "phoenix_pose_features_train")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


#Extract landmarks
def extract_landmarks(results):
    def get_coords(landmarks, dim=3):
        if landmarks:
            return np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
        return np.zeros(21 * 3)  # fallback for hands

    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    lh = get_coords(results.left_hand_landmarks)
    rh = get_coords(results.right_hand_landmarks)
    
    return np.concatenate([pose, lh, rh])

# Process Video
def process_video(video_path):
    try:
        rel_path = os.path.relpath(video_path, VIDEO_ROOT)
        save_path = os.path.join(OUTPUT_ROOT, os.path.splitext(rel_path)[0] + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Skip if already processed
        if os.path.exists(save_path):
            return f"Skipped: {video_path}"

        cap = cv2.VideoCapture(video_path)
        frames_data = []

        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                frames_data.append(extract_landmarks(results))

        cap.release()
        np.save(save_path, np.array(frames_data))
        return f"Processed: {video_path}"

    except Exception as e:
        return f"Failed: {video_path}, Error: {str(e)}"


if __name__ == "__main__":
    all_videos = glob.glob(os.path.join(VIDEO_ROOT, "**/*.mp4"), recursive=True)
    print(f"Found {len(all_videos)} videos.")

    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_video, all_videos), total=len(all_videos)):
            print(result)
    
    print("Extraction complete, All poses stored in:", OUTPUT_ROOT)
