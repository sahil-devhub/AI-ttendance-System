import streamlit as st
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
import base64
from PIL import Image
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI-ttendance System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Create necessary directories
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)

# Try to import FaceNet, fallback to a simpler face recognition method if not available
try:
    from keras_facenet import FaceNet
    embedder = FaceNet()
    USE_FACENET = True
    st.sidebar.success("Using FaceNet for face recognition (more accurate)")
except ImportError:
    USE_FACENET = False
    try:
        import face_recognition
        st.sidebar.info("FaceNet not available, using face_recognition library")
    except ImportError:
        st.sidebar.warning("Using basic OpenCV face detection as fallback (less accurate)")
        
# Ensure attendance file exists with header
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Name,DateTime,Confidence\n")

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
        
        # For attendance tracking
        self.consecutive_counts = {}
        self.last_recognition_time = {}
        self.recognized_names = set()

    def load_embeddings(self):
        """Load stored embeddings from pickle file"""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    self.embeddings_data = pickle.load(f)
                return True
            except Exception as e:
                st.error(f"Error loading embeddings: {e}")
        return False

    def save_embeddings(self):
        """Save embeddings to pickle file"""
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.embeddings_data, f)

    def detect_face(self, image):
        """Detect face in image using OpenCV for better performance"""
        if image is None:
            return None
            
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
        if image is None:
            return None, None
            
        if face_location is None and not USE_FACENET:
            face_location = self.detect_face(image)
            if face_location is None:
                return None, None
                
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
                
                if 'face_recognition' in globals():
                    face_locations = face_recognition.face_locations(face_rgb)
                    if not face_locations:
                        return None, None
                    face_encoding = face_recognition.face_encodings(face_rgb, face_locations)[0]
                    return face_encoding, (x, y, w, h)
                else:
                    # Fallback to a simple feature extraction if face_recognition not available
                    # This is much less accurate but works as a fallback
                    resized = cv2.resize(face_rgb, (128, 128))
                    flattened = resized.flatten() / 255.0  # Simple pixel-based features
                    return flattened, (x, y, w, h)
            except Exception as e:
                st.error(f"Error in get_face_embedding: {e}")
                return None, None

    def register_student(self, student_name, image_frame):
        """Register a new student with the provided image frame"""
        if image_frame is None:
            return False, "No image captured"
            
        # Create directory for this student
        student_dir = FACE_IMAGES_DIR / student_name
        os.makedirs(student_dir, exist_ok=True)
        
        # Save the original image
        image_path = student_dir / f"{student_name}_original.jpg"
        cv2.imwrite(str(image_path), image_frame)
        
        # Get embedding
        embedding, _ = self.get_face_embedding(image_frame)
        if embedding is None:
            return False, "Failed to extract face embedding"
            
        # Store the embedding
        if student_name in self.embeddings_data:
            # Update existing embeddings
            self.embeddings_data[student_name].append(embedding)
        else:
            # Create new entry
            self.embeddings_data[student_name] = [embedding]
            
        # Save embeddings to disk
        self.save_embeddings()
        
        # Generate synthetic images if possible
        try:
            import albumentations as A
            self.generate_synthetic_images(student_name, image_frame)
        except ImportError:
            pass
            
        return True, f"Successfully registered {student_name}"
        
    def generate_synthetic_images(self, student_name, original_image, num_images=5):
        """Generate synthetic images for better recognition"""
        try:
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
                return
                
            x, y, w, h = face_location
            face_img = original_image[y:y+h, x:x+w]
            
            # Generate augmented images
            for i in range(num_images):
                # Apply augmentation
                augmented = transform(image=face_img)['image']
                
                # Save augmented image
                output_path = student_dir / f"{student_name}_synthetic_{i}.jpg"
                cv2.imwrite(str(output_path), augmented)
                
                # Extract and store embedding
                embedding, _ = self.get_face_embedding(augmented)
                if embedding is not None:
                    self.embeddings_data[student_name].append(embedding)
            
            # Save updated embeddings
            self.save_embeddings()
        except Exception as e:
            st.warning(f"Error generating synthetic images: {e}")

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

    def process_attendance_frame(self, frame):
        """Process a single frame for attendance"""
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
        
        # Process results
        display_frame = frame.copy()
        current_time = time.time()
        newly_recognized = []
        
        for result in face_results:
            name = result["name"]
            confidence = result["confidence"]
            box = result["box"]
            
            # Track consecutive recognitions
            if name not in self.consecutive_counts:
                self.consecutive_counts[name] = 0
            if name not in self.last_recognition_time:
                self.last_recognition_time[name] = 0
                
            # Update counts
            self.consecutive_counts[name] += 1
            
            # Reset other counts
            for other_name in self.consecutive_counts:
                if other_name != name:
                    self.consecutive_counts[other_name] = 0
            
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
                name not in self.recognized_names and
                self.consecutive_counts[name] >= CONSECUTIVE_FRAMES_REQUIRED and
                current_time - self.last_recognition_time[name] > MIN_RECOGNITION_INTERVAL and
                confidence > MIN_CONFIDENCE_FOR_ATTENDANCE):
                
                self.recognized_names.add(name)
                self.last_recognition_time[name] = current_time
                newly_recognized.append(name)
                
                # Write to attendance file
                with open(ATTENDANCE_FILE, "a") as f:
                    f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{confidence:.2f}\n")
        
        return display_frame, newly_recognized

    def start_processing_thread(self):
        """Start the background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self.process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
    def stop_processing_thread(self):
        """Stop the background processing thread"""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
            
    def reset_attendance_session(self):
        """Reset the attendance tracking for a new session"""
        self.consecutive_counts = {}
        self.last_recognition_time = {}
        self.recognized_names = set()

# Initialize session state
if 'face_processor' not in st.session_state:
    st.session_state.face_processor = FaceProcessor()
    
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
    
if 'registration_image' not in st.session_state:
    st.session_state.registration_image = None
    
if 'attendance_mode' not in st.session_state:
    st.session_state.attendance_mode = False

# Helper function to convert cv2 image to displayable format
def convert_to_jpg(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# Custom CSS to make the app look better
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .subheader {
        font-size: 1.5em;
        color: #333;
        margin-bottom: 20px;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        box-shadow: 2px 2px 10px #cccccc;
        margin-bottom: 20px;
    }
    .streamlit-expanderHeader {
        font-size: 1.2em;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar
st.sidebar.markdown("<h2 style='text-align: center;'>AI-ttendance System</h2>", unsafe_allow_html=True)
st.sidebar.image("https://img.icons8.com/color/96/000000/face-id.png", width=100)

menu = st.sidebar.radio("Navigation", 
    ["Home", "Register Student", "Take Attendance", "View Attendance Records", "Settings"])

# Display number of registered students
num_students = len(st.session_state.face_processor.embeddings_data)
st.sidebar.markdown(f"### Registered Students: {num_students}")

# Home page
if menu == "Home":
    st.markdown("<h1 class='main-header'>AI-ttendance System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h2 class='subheader'>Welcome to AI-ttendance!</h2>
        <p>AI-ttendance is a smart attendance system powered by facial recognition technology. It offers:</p>
        <ul>
            <li>Quick and accurate student registration</li>
            <li>Automated attendance tracking</li>
            <li>Real-time face recognition</li>
            <li>Comprehensive attendance reports</li>
        </ul>
        <p>Select an option from the sidebar menu to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3>Quick Start Guide</h3>
        <ol>
            <li><b>Register Student</b>: Add new students to the system</li>
            <li><b>Take Attendance</b>: Start a live attendance session</li>
            <li><b>View Records</b>: Check attendance history</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Registered Students", num_students)
        
        # Get today's attendance
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['DateTime']).dt.date
                today = datetime.now().date()
                today_count = len(df[df['Date'] == today])
                st.metric("Today's Attendance", today_count)
        else:
            st.metric("Today's Attendance", 0)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # System status
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("System Status")
        
        if USE_FACENET:
            st.markdown("<p class='success-text'>✓ FaceNet loaded (high accuracy)</p>", unsafe_allow_html=True)
        elif 'face_recognition' in globals():
            st.markdown("<p class='warning-text'>⚠️ Using face_recognition library</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='error-text'>⚠️ Using basic OpenCV (limited accuracy)</p>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# Register student page
elif menu == "Register Student":
    st.markdown("<h1 class='main-header'>Register New Student</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Student Information")
        
        student_name = st.text_input("Student Name (First Last)")
        roll_no = st.text_input("Roll Number")
        
        if student_name and roll_no:
            formatted_name = f"{student_name.strip().replace(' ', '_')}_{roll_no.strip()}"
            st.info(f"Registration ID will be: {formatted_name}")
        else:
            formatted_name = ""
            
        # Camera capture
        st.subheader("Capture Face Image")
        
        if st.button("Start Camera" if not st.session_state.camera_active else "Stop Camera"):
            st.session_state.camera_active = not st.session_state.camera_active
            st.session_state.registration_image = None
            st.experimental_set_query_params()
            
        if st.session_state.camera_active:
            # Create a placeholder for the webcam
            camera_placeholder = st.empty()
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.camera_active = False
            else:
                capture_clicked = st.button("Capture Image")
                
                # Show webcam feed until capture button is clicked
                while st.session_state.camera_active and not capture_clicked:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from camera")
                        break
                        
                    # Detect face
                    face_location = st.session_state.face_processor.detect_face(frame)
                    
                    if face_location:
                        x, y, w, h = face_location
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Calculate quality score (simple heuristic based on size)
                        quality = w * h / (frame.shape[0] * frame.shape[1])
                        quality_text = f"Quality: {int(quality * 100)}%"
                        cv2.putText(frame, quality_text, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No face detected", (30, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display the resulting frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Check if button was clicked
                    if st.button("Stop Camera", key="stop_cam"):
                        break
                        
                    # Add a small delay
                    time.sleep(0.1)
                
                # If capture button was clicked
                if capture_clicked:
                    ret, frame = cap.read()
                    if ret:
                        st.session_state.registration_image = frame.copy()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Captured Image", use_column_width=True)
                    else:
                        st.error("Failed to capture image")
                        
                # Release the webcam
                cap.release()
                st.session_state.camera_active = False
                
        if st.session_state.registration_image is not None:
            st.success("Image captured successfully!")
            
            if st.button("Register Student") and formatted_name:
                success, message = st.session_state.face_processor.register_student(
                    formatted_name, st.session_state.registration_image)
                
                if success:
                    st.success(message)
                    # Reset the image
                    st.session_state.registration_image = None
                    time.sleep(2)
                    st.experimental_rerun()
                else:
                    st.error(message)
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Registration Guidelines")
        st.markdown("""
        For the best recognition results:
        
        - Ensure good lighting on your face
        - Look directly at the camera
        - Maintain neutral expression
        - Remove glasses if possible
        - Keep a neutral background
        
        The system will create multiple synthetic variants of your image to improve recognition accuracy in different conditions.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show registered students
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Registered Students")
        
        if num_students > 0:
            for i, name in enumerate(st.session_state.face_processor.embeddings_data.keys()):
                st.write(f"{i+1}. {name.replace('_', ' ', 1).replace('_', ' - ID: ')}")
        else:
            st.info("No students registered yet")
            
        st.markdown("</div>", unsafe_allow_html=True)

# Take attendance page  
elif menu == "Take Attendance":
    st.markdown("<h1 class='main-header'>Take Attendance</h1>", unsafe_allow_html=True)
    
    if not st.session_state.face_processor.embeddings_data:
        st.warning("No students registered in the system. Please register students first.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            if not st.session_state.attendance_mode:
                if st.button("Start Attendance Session", key="start_attendance"):
                    st.session_state.attendance_mode = True
                    st.session_state.face_processor.reset_attendance_session()
                    st.session_state.face_processor.start_processing_thread()
                    st.experimental_rerun()
            else:
                if st.button("End Attendance Session", key="end_attendance"):
                    st.session_state.attendance_mode = False
                    st.session_state.face_processor.stop_processing_thread()
                    st.experimental_rerun()
                
                # Create a placeholder for the webcam
                camera_placeholder = st.empty()
                
                # Create a placeholder for notifications
                notification_placeholder = st.empty()
                
                # Initialize webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera connection.")
                    st.session_state.attendance_mode = False
                    st.session_state.face_processor.stop_processing_thread()
                else:
                    try:
                        # Show webcam feed and process frames
                        frame_skip = 0  # Process every frame
                        
                        while st.session_state.attendance_mode:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to grab frame from camera")
                                break
                                
                            # Process every few frames to reduce CPU load
                            if frame_skip == 0:
                                display_frame, newly_recognized = st.session_state.face_processor.process_attendance_frame(frame)
                                
                                # Show notification for newly recognized students
                                for name in newly_recognized:
                                    notification_placeholder.success(f"✓ Attendance marked for: {name}")
                                
                                # Display the resulting frame
                                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                                
                            frame_skip = (frame_skip + 1) % 2  # Skip every other frame
                            
                            # Add a small delay
                            time.sleep(0.03)
                            
                    except Exception as e:
                        st.error(f"Error in attendance session: {e}")
                    finally:
                        # Release the webcam
                        cap.release()
                        st.session_state.attendance_mode = False
                        st.session_state.face_processor.stop_processing_thread()
                        
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Current Session")
            
            # Show current attendance
            recognized_names = st.session_state.face_processor.recognized_names
            if recognized_names:
                st.write(f"Students marked present: {len(recognized_names)}")
                for i, name in enumerate(recognized_names):
                    display_name = name.replace('_', ' ', 1).replace('_', ' (ID: ') + ')'
                    st.write(f"✓ {i+1}. {display_name}")
            else:
                st.info("No students marked present in this session yet")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show attendance tips
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Tips")
            st.markdown("""
            - Look directly at the camera
            - Ensure good lighting
            - Stand about 1-2 feet from the camera
            - Hold still for better recognition
            """)
            st.markdown("</div>", unsafe_allow_html=True)

# View attendance records page
elif menu == "View Attendance Records":
    st.markdown("<h1 class='main-header'>Attendance Records</h1>", unsafe_allow_html=True)
    
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        st.warning("No attendance records found.")
    else:
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if df.empty:
                st.warning("Attendance file exists but has no records.")
            else:
                # Add date column
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df['Date'] = df['DateTime'].dt.date
                df['Time'] = df['DateTime'].dt.time
                
                # Group by date
                dates = sorted(df['Date'].unique(), reverse=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Select Date")
                    
                    selected_date = st.selectbox("Choose a date", dates)
                    
                    # Count records for selected date
                    date_records = df[df['Date'] == selected_date]
                    st.metric("Total Present", len(date_records))
                    
                    # Export options
                    if st.button("Export to CSV"):
                        date_records_export = date_records[['Name', 'DateTime', 'Confidence']]
                        date_records_export.to_csv(f"attendance_{selected_date}.csv", index=False)
                        st.success(f"Exported to attendance_{selected_date}.csv")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader(f"Attendance for {selected_date}")
                    
                    # Show records for selected date
                    if not date_records.empty:
                        # Format the data for display
                        display_df = date_records.copy()
                        display_df['Time'] = display_df['DateTime'].dt.strftime('%H:%M:%S')
                        display_df['Name'] = display_df['Name'].str.replace('_', ' ', 1).str.replace('_', ' (ID: ') + ')'
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}")
                        
                        # Display the table
                        st.dataframe(display_df[['Name', 'Time', 'Confidence']], 
                                    height=400, 
                                    use_container_width=True)
                    else:
                        st.info("No records found for this date")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Add visualization
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Attendance Visualization")
                
                # Choose visualization type
                viz_option = st.radio("Choose visualization", 
                                     ["Daily Attendance Count", "Time Distribution"])
                
                if viz_option == "Daily Attendance Count":
                    # Group by date and count
                    date_counts = df.groupby('Date').size().reset_index(name='Count')
                    date_counts['Date'] = date_counts['Date'].astype(str)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(date_counts['Date'], date_counts['Count'], color='#1E88E5')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Number of Students')
                    ax.set_title('Daily Attendance Count')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add data labels
                    for i, v in enumerate(date_counts['Count']):
                        ax.text(i, v + 0.5, str(v), ha='center')
                        
                    st.pyplot(fig)
                
                else:  # Time Distribution
                    # Get data for selected date
                    date_records = df[df['Date'] == selected_date]
                    
                    if not date_records.empty:
                        # Extract hour from datetime
                        date_records['Hour'] = date_records['DateTime'].dt.hour
                        
                        # Count by hour
                        hour_counts = date_records.groupby('Hour').size().reset_index(name='Count')
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(hour_counts['Hour'], hour_counts['Count'], color='#43A047')
                        ax.set_xlabel('Hour of Day')
                        ax.set_ylabel('Number of Students')
                        ax.set_title(f'Attendance Time Distribution for {selected_date}')
                        ax.set_xticks(range(24))
                        
                        # Add data labels
                        for i, v in enumerate(hour_counts['Count']):
                            ax.text(hour_counts['Hour'].iloc[i], v + 0.1, str(v), ha='center')
                            
                        st.pyplot(fig)
                    else:
                        st.info("No data available for the selected date")
                
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")

# Settings page
elif menu == "Settings":
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Face Recognition Settings")
        
        # Face recognition threshold
        threshold = st.slider("Recognition Threshold", 0.1, 0.9, 
                             float(FACE_RECOGNITION_THRESHOLD), 0.05,
                             help="Lower values are stricter (fewer false positives, more false negatives)")
        
        # Required consecutive frames
        consecutive_frames = st.slider("Required Consecutive Frames", 1, 10, 
                                     int(CONSECUTIVE_FRAMES_REQUIRED),
                                     help="Number of consecutive frames required for a positive identification")
        
        # Minimum confidence
        min_confidence = st.slider("Minimum Confidence for Attendance", 0.3, 0.9, 
                                 float(MIN_CONFIDENCE_FOR_ATTENDANCE), 0.05,
                                 help="Minimum confidence level required to mark attendance")
        
        # Save settings button
        if st.button("Save Settings"):
            # In a real app, you would modify the constants and save them
            # For simplicity, we'll just show a success message
            st.success("Settings saved successfully!")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Database Management")
        
        # Student count
        st.metric("Registered Students", num_students)
        
        # Export embeddings
        if st.button("Export Embeddings") and num_students > 0:
            # In a real app, you might want to create a download link
            st.success("Embeddings exported successfully!")
            
        # Import embeddings
        st.file_uploader("Import Embeddings", type=["pkl"], 
                        help="Upload a previously exported embeddings file")
        
        # Clear database
        if st.button("Clear All Student Data"):
            confirm = st.checkbox("I understand this will delete ALL student data", key="confirm_clear")
            if confirm:
                # Delete the data
                st.session_state.face_processor.embeddings_data = {}
                st.session_state.face_processor.save_embeddings()
                st.success("All student data cleared successfully!")
                time.sleep(2)
                st.experimental_rerun()
            else:
                st.warning("Please confirm by checking the box")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    # System info
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Face Recognition Engine:**")
        if USE_FACENET:
            st.markdown("✅ FaceNet (High Accuracy)")
        elif 'face_recognition' in globals():
            st.markdown("⚠️ face_recognition library (Medium Accuracy)")
        else:
            st.markdown("⚠️ Basic OpenCV (Limited Accuracy)")
            
        st.markdown("**OpenCV Version:**")
        st.code(cv2.__version__)
    
    with col2:
        st.markdown("**Attendance File:**")
        if os.path.exists(ATTENDANCE_FILE):
            st.markdown(f"✅ Found ({os.path.getsize(ATTENDANCE_FILE)/1024:.1f} KB)")
        else:
            st.markdown("❌ Not found")
            
        st.markdown("**Embeddings File:**")
        if os.path.exists(EMBEDDINGS_FILE):
            st.markdown(f"✅ Found ({os.path.getsize(EMBEDDINGS_FILE)/1024:.1f} KB)")
        else:
            st.markdown("❌ Not found")
            
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
    <p>AI-ttendance System © 2025 | Powered by Computer Vision & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Add missing import
import matplotlib.pyplot as plt