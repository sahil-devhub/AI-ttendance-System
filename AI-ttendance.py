import cv2
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
from scipy.spatial.distance import cosine
import warnings
from pathlib import Path
import threading    
import queue
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FACE_IMAGES_DIR = BASE_DIR / "Face_images"
EMBEDDINGS_FILE = BASE_DIR / "face_embeddings.pkl"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
HAARCASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Face recognition parameters
FACE_RECOGNITION_THRESHOLD = 0.35  # Lower is stricter
CONSECUTIVE_FRAMES_REQUIRED = 3    # Number of consecutive frames for positive ID
MIN_RECOGNITION_INTERVAL = 30      # Seconds between recognitions of same person
MIN_CONFIDENCE_FOR_ATTENDANCE = 0.6 # Minimum confidence to mark attendance

# Try to import FaceNet, fallback to a simpler face recognition method if not available
try:
    from keras_facenet import FaceNet
    embedder = FaceNet()
    USE_FACENET = True
    print("Using FaceNet for face recognition")
except ImportError:
    import face_recognition  # pip install face-recognition (this uses dlib)
    USE_FACENET = False
    print("FaceNet not available, using face_recognition library")
    
    # If face_recognition also not available, we'll use a simple fallback
    if 'face_recognition' not in sys.modules:
        USE_FACENET = False
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE)
        print("Using basic OpenCV face detection as fallback")

# Create necessary directories
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)

class FaceProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE)
        self.embeddings_data = {}
        self.load_embeddings()
        
        # For better performance
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.processing_thread = None

    def load_embeddings(self):
        """Load stored embeddings from pickle file"""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    self.embeddings_data = pickle.load(f)
                print(f"Loaded embeddings for {len(self.embeddings_data)} students")
                return True
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        else:
            print("No existing embeddings found.")
        return False

    def save_embeddings(self):
        """Save embeddings to pickle file"""
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.embeddings_data, f)
        print(f"Saved embeddings for {len(self.embeddings_data)} students")

    def detect_face(self, image):
        """Detect face in image using OpenCV for better performance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80)
        )
        
        if len(faces) == 0:
            return None
        
        # Take the largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
        x, y, w, h = faces[0]
        # Add some margin
        margin = min(20, min(x, y))
        x_new = max(0, x - margin)
        y_new = max(0, y - margin)
        w_new = min(image.shape[1] - x_new, w + 2*margin)
        h_new = min(image.shape[0] - y_new, h + 2*margin)
        
        return (x_new, y_new, w_new, h_new)

    def get_face_embedding(self, image, face_location=None):
        """Extract face embedding"""
        if face_location is None and not USE_FACENET:
            face_location = self.detect_face(image)
            if face_location is None:
                return None
                
        if USE_FACENET:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = embedder.extract(img_rgb, threshold=0.95)
            if faces:
                return faces[0]["embedding"], faces[0]["box"]
            return None, None
        else:
            try:
                x, y, w, h = face_location
                face_img = image[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(face_rgb)
                if not face_locations:
                    return None, None
                face_encoding = face_recognition.face_encodings(face_rgb, face_locations)[0]
                return face_encoding, (x, y, w, h)
            except Exception as e:
                print(f"Error in get_face_embedding: {e}")
                return None, None

    def register_student(self):
        """Register a new student by capturing their face"""
        student_name = input("Enter student name (format: name_rollNo): ").strip()
        if not student_name:
            print("Error: Name cannot be empty")
            return
            
        # Validate name format
        if not '_' in student_name or len(student_name.split('_')[1].strip()) == 0:
            print("Error: Name must be in format 'name_rollNo'")
            return
        
        print("\nPreparing camera for registration...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        # Wait for camera to initialize
        time.sleep(1)
        
        # Create directory for this student
        student_dir = FACE_IMAGES_DIR / student_name
        os.makedirs(student_dir, exist_ok=True)
        
        best_face = None
        best_quality = 0
        
        print("\nPlease look at the camera...")
        print("Press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
                
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect face
            face_location = self.detect_face(frame)
            
            if face_location:
                x, y, w, h = face_location
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate quality score (simple heuristic based on size)
                quality = w * h / (frame.shape[0] * frame.shape[1])
                quality_text = f"Quality: {int(quality * 100)}%"
                cv2.putText(display_frame, quality_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If this is the best face so far, keep it
                if quality > best_quality:
                    best_quality = quality
                    best_face = frame.copy()
            else:
                cv2.putText(display_frame, "No face detected", (30, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show instructions
            cv2.putText(display_frame, "SPACE: Capture   ESC: Cancel", 
                       (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Registration - AI-ttendance", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # SPACE
                if face_location:
                    break
                else:
                    print("No face detected. Please try again.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if best_face is None:
            print("Failed to capture a usable face image")
            return
            
        # Save the original image
        image_path = student_dir / f"{student_name}_original.jpg"
        cv2.imwrite(str(image_path), best_face)
        
        # Get embedding
        embedding, _ = self.get_face_embedding(best_face)
        if embedding is None:
            print("Failed to extract face embedding")
            return
            
        # Store the embedding
        if student_name in self.embeddings_data:
            # Update existing embeddings
            self.embeddings_data[student_name].append(embedding)
        else:
            # Create new entry
            self.embeddings_data[student_name] = [embedding]
            
        # Save embeddings to disk
        self.save_embeddings()
        
        print(f"\nSuccessfully registered {student_name}")
        print(f"Original image saved to {image_path}")
        
        # Generate synthetic images if possible
        try:
            import albumentations as A
            self.generate_synthetic_images(student_name, best_face)
        except ImportError:
            print("\nAlbumentations library not found. Skipping synthetic image generation.")
            print("If you want to generate synthetic images, please install albumentations:")
            print("pip install albumentations")
            
    def generate_synthetic_images(self, student_name, original_image, num_images=10):
        """Generate synthetic images for better recognition"""
        print(f"Generating {num_images} synthetic images...")
        
        import albumentations as A
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
        student_dir = FACE_IMAGES_DIR / student_name
        face_location = self.detect_face(original_image)
        
        if face_location is None:
            print("Cannot generate synthetic images: no face detected")
            return
            
        x, y, w, h = face_location
        face_img = original_image[y:y+h, x:x+w]
        
        # Generate augmented images
        for i in range(num_images):
            try:
                # Apply augmentation
                augmented = transform(image=face_img)['image']
                
                # Save augmented image
                output_path = student_dir / f"{student_name}_synthetic_{i}.jpg"
                cv2.imwrite(str(output_path), augmented)
                
                # Extract and store embedding
                embedding, _ = self.get_face_embedding(augmented)
                if embedding is not None:
                    self.embeddings_data[student_name].append(embedding)
                    print(f"Synthetic image {i+1}/{num_images} processed successfully")
                else:
                    print(f"Failed to extract embedding for synthetic image {i+1}")
            except Exception as e:
                print(f"Error generating synthetic image {i+1}: {e}")
        
        # Save updated embeddings
        self.save_embeddings()
        print(f"Added {num_images} synthetic variations. Total embeddings for {student_name}: {len(self.embeddings_data[student_name])}")

    def process_frames(self):
        """Process frames in a separate thread for better performance"""
        while not self.stop_event.is_set():
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = self.frame_queue.get(block=False)
                results = self.recognize_faces(frame)
                
                # Put results in queue if there's space, otherwise skip
                try:
                    self.result_queue.put(results, block=False)
                except queue.Full:
                    pass
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.1)

    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        results = []
        
        if USE_FACENET:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = embedder.extract(img_rgb, threshold=0.95)
            
            for face in faces:
                embedding = face["embedding"]
                box = face["box"]
                name, confidence = self.match_embedding(embedding)
                results.append({
                    "name": name,
                    "confidence": confidence,
                    "box": box
                })
        else:
            # Use OpenCV for face detection
            face_location = self.detect_face(frame)
            if face_location:
                embedding, box = self.get_face_embedding(frame, face_location)
                if embedding is not None:
                    name, confidence = self.match_embedding(embedding)
                    results.append({
                        "name": name,
                        "confidence": confidence,
                        "box": box
                    })
        
        return results

    def match_embedding(self, embedding):
        """Match a face embedding against stored embeddings"""
        best_match = "Unknown"
        best_confidence = 0
        
        for name, embeddings in self.embeddings_data.items():
            # Compare against all stored embeddings for this person
            similarities = []
            for stored_embedding in embeddings:
                if USE_FACENET:
                    similarity = 1 - cosine(embedding, stored_embedding)
                else:
                    # For face_recognition library
                    similarity = 1 - np.linalg.norm(np.array(embedding) - np.array(stored_embedding))
                similarities.append(similarity)
            
            # Take average of top 3 similarities (or all if less than 3)
            top_similarities = sorted(similarities, reverse=True)[:min(3, len(similarities))]
            avg_similarity = sum(top_similarities) / len(top_similarities)
            
            if avg_similarity > best_confidence and avg_similarity > FACE_RECOGNITION_THRESHOLD:
                best_confidence = avg_similarity
                best_match = name
        
        return best_match, best_confidence

    def take_attendance(self):
        """Run face recognition for attendance"""
        if not self.embeddings_data:
            print("No embeddings found. Please register students first.")
            return
            
        # Ensure attendance file exists with header
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w") as f:
                f.write("Name,DateTime,Confidence\n")
        
        print("\nStarting attendance system...")
        print(f"Loaded data for {len(self.embeddings_data)} students")
        print("Press 'q' to quit")
        
        # Start processing thread for better performance
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            self.stop_event.set()
            return
            
        # State for tracking consecutive recognitions
        consecutive_counts = {}
        last_recognition_time = {}
        recognized_names = set()
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Add frame to processing queue (non-blocking)
            try:
                if self.frame_queue.empty():
                    self.frame_queue.put(frame.copy(), block=False)
            except queue.Full:
                pass
                
            # Get results if available
            face_results = []
            try:
                if not self.result_queue.empty():
                    face_results = self.result_queue.get(block=False)
            except queue.Empty:
                pass
            
            # Display results
            display_frame = frame.copy()
            current_time = time.time()
            
            for result in face_results:
                name = result["name"]
                confidence = result["confidence"]
                box = result["box"]
                
                # Track consecutive recognitions
                if name not in consecutive_counts:
                    consecutive_counts[name] = 0
                if name not in last_recognition_time:
                    last_recognition_time[name] = 0
                    
                # Update counts
                consecutive_counts[name] += 1
                
                # Reset other counts
                for other_name in consecutive_counts:
                    if other_name != name:
                        consecutive_counts[other_name] = 0
                
                # Determine display color based on confidence
                if name != "Unknown":
                    if confidence > 0.7:
                        color = (0, 255, 0)  # Green for high confidence
                    else:
                        color = (0, 165, 255)  # Orange for medium confidence
                else:
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and name
                x, y, w, h = box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Display name and confidence
                confidence_text = f"{int(confidence * 100)}%" if name != "Unknown" else ""
                label = f"{name} {confidence_text}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Mark attendance if conditions are met
                if (name != "Unknown" and 
                    name not in recognized_names and
                    consecutive_counts[name] >= CONSECUTIVE_FRAMES_REQUIRED and
                    current_time - last_recognition_time[name] > MIN_RECOGNITION_INTERVAL and
                    confidence > MIN_CONFIDENCE_FOR_ATTENDANCE):
                    
                    recognized_names.add(name)
                    last_recognition_time[name] = current_time
                    
                    # Write to attendance file
                    with open(ATTENDANCE_FILE, "a") as f:
                        f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{confidence:.2f}\n")
                    
                    print(f"✓ Attendance marked for: {name} (Confidence: {confidence:.2f})")
            
            # Show FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show recognized names
            y_pos = 90
            cv2.putText(display_frame, "Attendance:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for name in recognized_names:
                y_pos += 30
                cv2.putText(display_frame, f"- {name}", (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("AI-ttendance - Face Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Clean up
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Attendance session ended. Results saved to {ATTENDANCE_FILE}")

    def view_attendance(self):
        """View attendance records"""
        if not os.path.exists(ATTENDANCE_FILE):
            print("No attendance records found.")
            return
            
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if df.empty:
                print("Attendance file exists but has no records.")
                return
                
            # Group by date
            df['Date'] = pd.to_datetime(df['DateTime']).dt.date
            dates = df['Date'].unique()
            
            while True:
                print("\n=== Attendance Records ===")
                print("Available dates:")
                for i, date in enumerate(dates, 1):
                    count = len(df[df['Date'] == date])
                    print(f"{i}. {date} ({count} records)")
                print("0. Return to main menu")
                
                choice = input("\nSelect a date (number): ")
                if choice == '0':
                    break
                    
                try:
                    date_idx = int(choice) - 1
                    if 0 <= date_idx < len(dates):
                        selected_date = dates[date_idx]
                        records = df[df['Date'] == selected_date]
                        
                        print(f"\nAttendance for {selected_date}:")
                        print("-" * 60)
                        print("Name                         Time           Confidence")
                        print("-" * 60)
                        
                        for _, row in records.iterrows():
                            time_str = pd.to_datetime(row['DateTime']).strftime('%H:%M:%S')
                            conf = float(row['Confidence']) if 'Confidence' in row else 0.0
                            print(f"{row['Name']:<30} {time_str:<15} {conf:.2f}")
                        
                        print("-" * 60)
                        input("Press Enter to continue...")
                    else:
                        print("Invalid selection!")
                except ValueError:
                    print("Please enter a valid number!")
        except Exception as e:
            print(f"Error reading attendance file: {e}")

def main():
    """Main function to control program flow"""
    face_processor = FaceProcessor()
    
    while True:
        print("\n" + "=" * 50)
        print("               AI-ttendance System")
        print("=" * 50)
        print("1. Register new student")
        print("2. Take attendance")
        print("3. View attendance records")
        print("4. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            face_processor.register_student()
        elif choice == "2":
            face_processor.take_attendance()
        elif choice == "3":
            face_processor.view_attendance()
        elif choice == "4":
            print("Exiting AI-ttendance. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()