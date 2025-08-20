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
import shutil
import queue
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import xlsxwriter
from io import BytesIO

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
STUDENT_DATA_FILE = BASE_DIR / "student_data.xlsx"
HAARCASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Face recognition parameters
FACE_RECOGNITION_THRESHOLD = 0.35  # Lower is stricter
CONSECUTIVE_FRAMES_REQUIRED = 3    # Number of consecutive frames for positive ID
MIN_RECOGNITION_INTERVAL = 30      # Seconds between recognitions of same person
MIN_CONFIDENCE_FOR_ATTENDANCE = 0.6 # Minimum confidence to mark attendance

# Create necessary directories
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)

try:
    from keras_facenet import FaceNet
    import tensorflow as tf
    
    # Patch to ensure unique layer names
    class PatchedFaceNet(FaceNet):
        def __init__(self, *args, **kwargs):
            # Initialize the parent FaceNet model
            super().__init__(*args, **kwargs)
            # Rename duplicate lambda layers to ensure uniqueness
            import tensorflow as tf
            for i, layer in enumerate(self.model.layers):
                if layer.name.startswith('lambda'):
                    layer._name = f"lambda_{i}"
            self.model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.output, name='patched_facenet')

    embedder = PatchedFaceNet()
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

# Function to create a download link for a DataFrame as Excel
def get_excel_download_link(df, filename, link_text):
    """Generate a link to download DataFrame as Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    return f'<a href="{href}" download="{filename}.xlsx">{link_text}</a>'

# Function to create a download link for binary data
def get_binary_download_link(bin_data, filename, link_text):
    """Generate a link to download binary data"""
    b64 = base64.b64encode(bin_data).decode()
    href = f'data:application/octet-stream;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{link_text}</a>'

class FaceProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE)
        self.embeddings_data = {}
        self.student_data = pd.DataFrame(columns=["Name", "Roll Number", "Registration Date"])
        self.load_embeddings()
        self.load_student_data()
        
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
    
    def load_student_data(self):
        """Load student data from Excel file"""
        if os.path.exists(STUDENT_DATA_FILE):
            try:
                self.student_data = pd.read_excel(STUDENT_DATA_FILE)
                return True
            except Exception as e:
                st.error(f"Error loading student data: {e}")
        return False
    
    def save_student_data(self):
        """Save student data to Excel file"""
        try:
            self.student_data.to_excel(STUDENT_DATA_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving student data: {e}")
            return False

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

    def register_student(self, student_name, roll_number, image_frame):
        """Register a new student with the provided image frame"""
        if image_frame is None:
            return False, "No image captured"
        
        # Format the student ID
        formatted_name = f"{student_name.strip().replace(' ', '_')}_{roll_number.strip()}"
            
        # Create directory for this student
        student_dir = FACE_IMAGES_DIR / formatted_name
        os.makedirs(student_dir, exist_ok=True)
        
        # Save the original image
        image_path = student_dir / f"{formatted_name}_original.jpg"
        cv2.imwrite(str(image_path), image_frame)
        
        # Get embedding
        embedding, _ = self.get_face_embedding(image_frame)
        if embedding is None:
            return False, "Failed to extract face embedding"
            
        # Store the embedding
        if formatted_name in self.embeddings_data:
            # Update existing embeddings
            self.embeddings_data[formatted_name].append(embedding)
        else:
            # Create new entry
            self.embeddings_data[formatted_name] = [embedding]
            
        # Save embeddings to disk
        self.save_embeddings()
        
        # Add to student data
        new_student = pd.DataFrame({
            "Name": [student_name],
            "Roll Number": [roll_number],
            "Registration Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        self.student_data = pd.concat([self.student_data, new_student], ignore_index=True)
        self.save_student_data()
        
        # Generate synthetic images if possible
        try:
            import albumentations as A
            self.generate_synthetic_images(formatted_name, image_frame)
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
        
    def delete_student(self, student_name):
        """Delete a student's data"""
        print(f"### Attempting to delete student: {student_name}")  # Enhanced debug
        print(f"Current embeddings_data keys: {list(self.embeddings_data.keys())}")  # Show all keys
        
        # Remove from embeddings data
        if student_name in self.embeddings_data:
            print(f"Found {student_name} in embeddings_data with value: {self.embeddings_data[student_name]}")
            del self.embeddings_data[student_name]
            if self.save_embeddings():
                print(f"Successfully saved updated embeddings to {EMBEDDINGS_FILE}")
            else:
                print(f"Failed to save embeddings after deletion. Check file permissions or locks.")
                return False
        else:
            print(f"{student_name} not found in embeddings_data")
            return False
        
        # Remove student directory
        student_dir = os.path.join(FACE_IMAGES_DIR, student_name)
        if os.path.exists(student_dir):
            try:
                import shutil
                shutil.rmtree(student_dir)
                print(f"Successfully deleted directory: {student_dir}")
            except Exception as e:
                print(f"Error deleting directory {student_dir}: {e}")
                return False
        else:
            print(f"Directory {student_dir} does not exist")
        
        # Remove from student data
        name_parts = student_name.split('_')
        if len(name_parts) >= 2:
            name = ' '.join(name_parts[:-1]).replace('_', ' ')  # Reconstruct name
            roll_number = name_parts[-1]
            print(f"Searching for name: {name}, roll_number: {roll_number}")
            original_df = self.student_data.copy()
            self.student_data = self.student_data[
                ~((self.student_data['Name'] == name) & 
                (self.student_data['Roll Number'] == roll_number))
            ]
            if not self.student_data.equals(original_df):  # Check if data changed
                if self.save_student_data():
                    print(f"Successfully updated and saved student data to {STUDENT_DATA_FILE}")
                else:
                    print(f"Failed to save student data after deletion. Reverting changes.")
                    # Revert embeddings
                    self.embeddings_data[student_name] = []  # Placeholder revert
                    self.save_embeddings()
                    return False
            else:
                print(f"No matching student found in student_data for {name}, {roll_number}")
                return False
        else:
            print(f"Invalid student name format: {student_name}")
            return False
        
        print(f"### Successfully deleted {student_name}")
        return True

# Initialize session state
if 'face_processor' not in st.session_state:
    st.session_state.face_processor = FaceProcessor()
    
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
    
if 'registration_image' not in st.session_state:
    st.session_state.registration_image = None
    
if 'attendance_mode' not in st.session_state:
    st.session_state.attendance_mode = False

# Custom CSS to make the app look better
st.markdown("""
<style>
    /* Your existing styles... */

    /* Enhance card visibility */
    .card {
        background-color: #f8f9fa !important; /* Match your preferred look */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px #cccccc;
        margin-bottom: 20px;
        overflow: visible !important; /* Prevent content clipping */
        min-height: 150px !important; /* Ensure minimum height for content */
    }

    /* Target home page cards in columns */
    div[data-testid="stAppViewContainer"] .card {
        display: block !important; /* Force block display */
        opacity: 1 !important; /* Ensure visibility */
    }

    /* Ensure all inner content is visible */
    .card * {
        opacity: 1 !important;
        visibility: visible !important;
        background-color: transparent !important;
        color: #333 !important; /* Ensure text is readable */
    }

    /* Fix lists and headings */
    .card ul, .card ol, .card li, .card p, .card h2, .card h3 {
        color: #333 !important;
        margin: 5px 0 !important;
        padding-left: 20px !important; /* Adjust for list indentation */
    }

    /* Ensure metric and subheader text is styled */
    .card .stMetricLabel, .card .stMetricValue, .card .stSubheader {
        color: #333 !important;
    }

    /* Custom text classes (if defined) */
    .success-text { color: #2e7d32; }
    .warning-text { color: #ed6c02; }
    .error-text { color: #d32f2f; }
</style>""", unsafe_allow_html=True)

# Create sidebar
st.sidebar.markdown("<h2 style='text-align: center;'>AI-ttendance System</h2>", unsafe_allow_html=True)
st.sidebar.image("https://img.icons8.com/color/96/000000/face-id.png", width=100)

menu = st.sidebar.radio("Navigation", 
    ["Home", "Register Student", "Take Attendance", "View Attendance Records", "Settings"])

# Display number of registered students
num_students = len(st.session_state.face_processor.embeddings_data)
st.sidebar.markdown(f"### Registered Students: {num_students}")

# Helper function to convert cv2 image to displayable format
def convert_to_jpg(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# Home page
# Home page
if menu == "Home":
    st.markdown("<h1 class='main-header'>AI-ttendance System</h1>", unsafe_allow_html=True)
    
    # Calculate today's attendance
    today_count = 0
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['DateTime']).dt.date
            today = datetime.now().date()
            today_count = len(df[df['Date'] == today])
    
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
            <li>Excel export functionality</li>
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
            <li><b>View Records</b>: Check attendance history and download reports</li>
            <li><b>Settings</b>: Manage system settings and data</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>Registered Students</h3>
            <p><b>Count:</b> {num_students}</p>
            <p><b>Today's Attendance:</b> {today_count}</p>
        </div>
        """.format(num_students=num_students, today_count=today_count), unsafe_allow_html=True)
        
        # System status
        st.markdown("""
        <div class='card'>
            <h3>System Status</h3>
            <p>
        """.format(), unsafe_allow_html=True)
        
        if USE_FACENET:
            st.markdown("<p class='success-text'>✓ FaceNet loaded (high accuracy)</p>", unsafe_allow_html=True)
        elif 'face_recognition' in globals():
            st.markdown("<p class='warning-text'>⚠️ Using face_recognition library</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='error-text'>⚠️ Using basic OpenCV (limited accuracy)</p>", unsafe_allow_html=True)
            
        # Show backup and restore options
        if st.button("Backup All Data"):
            if os.path.exists(EMBEDDINGS_FILE):
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    embeddings_data = f.read()
                    backup_name = f"ai-ttendance-backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.markdown(
                        get_binary_download_link(embeddings_data, f"{backup_name}.pkl", "Download Backup File"),
                        unsafe_allow_html=True
                    )
                    st.success("Backup created successfully!")
            else:
                st.warning("No embedding data to backup.")
        
        st.markdown("</p></div>", unsafe_allow_html=True)

# Register student page
# Register student page
# Register student page
elif menu == "Register Student":
    st.markdown("<h1 class='main-header'>Register New Student</h1>", unsafe_allow_html=True)
    
    # Create tabs for Register and Manage
    register_tab, manage_tab = st.tabs(["Register New Student", "Manage Existing Students"])
    
    with register_tab:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Student Information")
            
            student_name = st.text_input("Student Full Name")
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
                st.query_params.clear()
                
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
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            quality = w * h / (frame.shape[0] * frame.shape[1])
                            quality_text = f"Quality: {int(quality * 100)}%"
                            cv2.putText(frame, quality_text, (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "No face detected", (30, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        time.sleep(0.1)
                    
                    # If capture button was clicked
                    if capture_clicked:
                        ret, frame = cap.read()
                        if ret:
                            st.session_state.registration_image = frame.copy()
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, channels="RGB", use_container_width=True)  # Replace use_column_width with use_container_width
                            st.session_state.camera_active = False
                    
                    cap.release()
            
            # Move the Stop Camera button outside the loop with a unique key
            if st.session_state.camera_active:
                if st.button("Stop Camera", key="stop_camera_unique"):
                    st.session_state.camera_active = False
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Show registration controls
            if st.session_state.registration_image is not None:
                st.subheader("Captured Image")
                
                # Show the captured image
                frame_rgb = cv2.cvtColor(st.session_state.registration_image, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_container_width=True)  # Replace use_column_width with use_container_width
                
                # Check if we have required info to register
                if not student_name or not roll_no:
                    st.warning("Please enter student name and roll number to register.")
                else:
                    if st.button("Register Student"):
                        # Attempt to register the student
                        success, message = st.session_state.face_processor.register_student(
                            student_name, roll_no, st.session_state.registration_image)
                        
                        if success:
                            st.success(message)
                            # Reset the capture after successful registration
                            st.session_state.registration_image = None
                            time.sleep(1)
                            st.rerun()  # Replace st.experimental_rerun() with st.rerun()
                        else:
                            st.error(message)
            else:
                st.info("Capture an image to register a student.")
                
            # Option to upload image instead of capturing
            st.subheader("Or Upload Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.registration_image = image
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, channels="RGB", use_container_width=True)
                
                face_location = st.session_state.face_processor.detect_face(image)
                if face_location is None:
                    st.warning("No face detected in the uploaded image. Please use a clearer image.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
    
    with manage_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Manage Registered Students")
        
        # Load student data
        processor = st.session_state.face_processor
        if os.path.exists(STUDENT_DATA_FILE):
            student_df = pd.read_excel(STUDENT_DATA_FILE)
            
            if len(student_df) > 0:
                # Create a copy for display
                display_df = student_df.copy()
                display_df['Registration Date'] = pd.to_datetime(display_df['Registration Date']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Add action buttons
                st.dataframe(display_df)
                
                # Allow deleting students
                st.subheader("Delete Student")
                
                # Create a selectbox with student names
                students = [f"{row['Name']} ({row['Roll Number']})" for _, row in student_df.iterrows()]
                
                if students:
                    selected_student = st.selectbox("Select student to delete:", students)
                    
                    if st.button("Delete Selected Student", key="delete_student_btn"):
                        # Extract name and roll number
                        name, roll_number = selected_student.split(" (")
                        roll_number = roll_number.rstrip(")")
                        
                        # Format for the system
                        formatted_name = f"{name.strip().replace(' ', '_')}_{roll_number.strip()}"
                        print(f"### Attempting to delete: {formatted_name}")
                        
                        # Delete from system
                        success = processor.delete_student(formatted_name)
                        if success:
                            st.success(f"Successfully deleted {name}")
                            print(f"### Deletion successful for {formatted_name}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to delete student data. Check console for details.")
                            print(f"### Deletion failed for {formatted_name}")
                else:
                    st.info("No students registered yet.")
                
                # Bulk operations (unchanged for now)
                st.subheader("Bulk Operations")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export Student List", key="export_btn"):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            student_df.to_excel(writer, index=False, sheet_name='Students')
                            workbook = writer.book
                            worksheet = writer.sheets['Students']
                            header_format = workbook.add_format({
                                'bold': True, 'text_wrap': True, 'valign': 'top',
                                'fg_color': '#D7E4BC', 'border': 1
                            })
                            for col_num, value in enumerate(student_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                            worksheet.set_column('A:A', 20)
                            worksheet.set_column('B:B', 15)
                            worksheet.set_column('C:C', 20)
                        excel_data = output.getvalue()
                        b64 = base64.b64encode(excel_data).decode()
                        dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="student_list_{datetime.now().strftime("%Y%m%d")}.xlsx">Download Student List</a>'
                        st.markdown(dl_link, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Reset All Student Data", key="reset_btn"):
                        st.warning("This will delete ALL student data. This action cannot be undone.")
                        if st.button("Confirm Reset", key="confirm_reset"):
                            print("Initiating reset of all student data")
                            processor.embeddings_data = {}
                            if processor.save_embeddings():
                                print("Embeddings cleared and saved")
                            else:
                                print("Failed to save embeddings")
                            processor.student_data = pd.DataFrame(columns=["Name", "Roll Number", "Registration Date"])
                            if processor.save_student_data():
                                print("Student data cleared and saved")
                            else:
                                print("Failed to save student data")
                            if os.path.exists(FACE_IMAGES_DIR):
                                try:
                                    import shutil
                                    for item in os.listdir(FACE_IMAGES_DIR):
                                        item_path = os.path.join(FACE_IMAGES_DIR, item)
                                        if os.path.isdir(item_path):
                                            shutil.rmtree(item_path)
                                    print("Face images directory cleared")
                                except Exception as e:
                                    print(f"Error clearing face images directory: {e}")
                                    st.error(f"Error clearing face images: {e}")
                            st.success("All student data has been reset.")
                            time.sleep(1)
                            st.rerun()
            else:
                st.info("No students registered yet.")
        else:
            st.info("No student data file found. Register some students first.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# Take Attendance page
elif menu == "Take Attendance":
    st.markdown("<h1 class='main-header'>Take Attendance</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Live Attendance")
        
        if 'attendance_camera_active' not in st.session_state:
            st.session_state.attendance_camera_active = False
        if st.button("Start Camera" if not st.session_state.attendance_camera_active else "Stop Camera"):
            st.session_state.attendance_camera_active = not st.session_state.attendance_camera_active
            if not st.session_state.attendance_camera_active and hasattr(st.session_state.face_processor, 'processing_thread'):
                st.session_state.face_processor.stop_processing_thread()
            elif st.session_state.attendance_camera_active:
                st.session_state.face_processor.start_processing_thread()
                st.session_state.face_processor.reset_attendance_session()  # Reset attendance state
        
        if st.session_state.attendance_camera_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.attendance_camera_active = False
            else:
                if st.button("Stop Camera", key="stop_camera_unique"):
                    st.session_state.attendance_camera_active = False
                    st.session_state.face_processor.stop_processing_thread()
                    cap.release()
                    st.session_state.attendance_frame = None
                    st.rerun()

                frame_placeholder = st.empty()
                hint_placeholder = st.empty()
                while st.session_state.attendance_camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from camera")
                        break
                    
                    # Use process_attendance_frame for attendance logging
                    display_frame, newly_recognized = st.session_state.face_processor.process_attendance_frame(frame)
                    attendance_list = st.session_state.face_processor.recognized_names.copy()
                    
                    if newly_recognized:
                        for name in newly_recognized:
                            hint_placeholder.success(f"Attendance recorded for {name.split('_')[0]}!")
                    
                    # Update session state with attendance
                    if 'live_attendance' not in st.session_state:
                        st.session_state.live_attendance = set()
                    st.session_state.live_attendance.update(student.split('_')[0] for student in attendance_list if student != "Unknown")
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                cap.release()
                st.session_state.attendance_frame = frame_rgb
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Attendance Log")
        if 'live_attendance' in st.session_state and st.session_state.live_attendance:
            st.write("Recognized Students:")
            for student in st.session_state.live_attendance:
                st.write(f"- {student}")
            
            attendance_list = st.session_state.face_processor.recognized_names.copy()
            if attendance_list:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                attendance_df = pd.DataFrame({
                    "Student Name": [s.split('_')[0].replace('_', ' ') for s in attendance_list],
                    "Roll Number": [s.split('_')[1] if '_' in s else '' for s in attendance_list],
                    "Timestamp": [timestamp] * len(attendance_list)
                })
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    attendance_df.to_excel(writer, index=False, sheet_name='Attendance')
                    workbook = writer.book
                    worksheet = writer.sheets['Attendance']
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    for col_num, value in enumerate(attendance_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column('A:A', 20)
                    worksheet.set_column('B:B', 15)
                    worksheet.set_column('C:C', 20)
                excel_data = output.getvalue()
                b64 = base64.b64encode(excel_data).decode()
                dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="attendance_{timestamp}.xlsx">Download Attendance as Excel</a>'
                st.markdown(dl_link, unsafe_allow_html=True)
        else:
            st.info("No students recognized yet. Start the camera to detect students.")
        
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "View Attendance Records":
    st.markdown("<h1 class='main-header'>Attendance Records</h1>", unsafe_allow_html=True)
    
    # Check if we have attendance data
    if os.path.exists(ATTENDANCE_FILE):
        # Load attendance data
        df = pd.read_csv(ATTENDANCE_FILE)
        
        if not df.empty:
            # Process the data
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df['Date'] = df['DateTime'].dt.date
            df['Time'] = df['DateTime'].dt.time
            
            # Create tabs for different views
            daily_tab, student_tab, export_tab = st.tabs(["Daily View", "Student View", "Export Data"])
            
            with daily_tab:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Daily Attendance Records")
                
                # Get unique dates
                unique_dates = sorted(df['Date'].unique(), reverse=True)
                selected_date = st.selectbox("Select Date:", unique_dates)
                
                if selected_date:
                    # Filter data for selected date
                    date_df = df[df['Date'] == selected_date]
                    
                    # Display attendance for this date
                    if not date_df.empty:
                        # Get unique students on this date
                        unique_students = date_df['Name'].unique()
                        
                        # Display count
                        st.metric("Students Present", len(unique_students))
                        
                        # Create a table for display
                        display_df = date_df.copy()
                        display_df['Name'] = display_df['Name'].apply(lambda x: x.split('_')[0].replace('_', ' '))
                        display_df['Roll Number'] = display_df['Name'].apply(lambda x: x.split('_')[1] if '_' in x else '')
                        display_df['Time'] = display_df['DateTime'].dt.strftime('%H:%M:%S')
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}")
                        
                        # Select columns to display
                        display_df = display_df[['Name', 'Roll Number', 'Time', 'Confidence']]
                        
                        # Show the table
                        st.dataframe(display_df)
                        
                        # Download button
                        if st.button("Download This Day's Report"):
                            # Create Excel file
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                display_df.to_excel(writer, index=False, sheet_name='Attendance')
                                
                                # Add some formatting
                                workbook = writer.book
                                worksheet = writer.sheets['Attendance']
                                
                                # Add a header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'fg_color': '#D7E4BC',
                                    'border': 1
                                })
                                
                                # Write the column headers with the defined format
                                for col_num, value in enumerate(display_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Set column widths
                                worksheet.set_column('A:A', 20)
                                worksheet.set_column('B:B', 15)
                                worksheet.set_column('C:C', 10)
                                worksheet.set_column('D:D', 10)
                            
                            # Generate download link
                            excel_data = output.getvalue()
                            b64 = base64.b64encode(excel_data).decode()
                            dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="attendance_{selected_date}.xlsx">Download {selected_date} Attendance</a>'
                            st.markdown(dl_link, unsafe_allow_html=True)
                    else:
                        st.info("No attendance records for this date.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with student_tab:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Student Attendance Records")
                
                # Get unique student names
                unique_students = sorted(df['Name'].unique())
                
                # Create a selection box with formatted names
                formatted_students = [name.split('_')[0].replace('_', ' ') + f" ({name.split('_')[1]})" 
                                      if '_' in name else name for name in unique_students]
                
                selected_student = st.selectbox("Select Student:", formatted_students)
                
                if selected_student:
                    # Extract the original name format
                    student_name = selected_student.split(" (")[0]
                    roll_number = selected_student.split(" (")[1].rstrip(")")
                    original_name = f"{student_name.replace(' ', '_')}_{roll_number}"
                    
                    # Filter data for selected student
                    student_df = df[df['Name'] == original_name]
                    
                    # Display attendance for this student
                    if not student_df.empty:
                        # Get unique dates for this student
                        unique_dates = student_df['Date'].unique()
                        
                        # Display count
                        st.metric("Days Present", len(unique_dates))
                        
                        # Create a table for display
                        display_df = student_df.copy()
                        display_df['Date'] = display_df['DateTime'].dt.strftime('%Y-%m-%d')
                        display_df['Time'] = display_df['DateTime'].dt.strftime('%H:%M:%S')
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}")
                        
                        # Select columns to display
                        display_df = display_df[['Date', 'Time', 'Confidence']]
                        
                        # Show the table
                        st.dataframe(display_df)
                        
                        # Display attendance percentage
                        if os.path.exists(STUDENT_DATA_FILE):
                            student_data = pd.read_excel(STUDENT_DATA_FILE)
                            if not student_data.empty:
                                # Get registration date
                                student_row = student_data[(student_data['Name'] == student_name) & 
                                                           (student_data['Roll Number'] == roll_number)]
                                if not student_row.empty:
                                    reg_date = pd.to_datetime(student_row['Registration Date'].values[0]).date()
                                    today = datetime.now().date()
                                    
                                    # Calculate days since registration
                                    days_since_reg = (today - reg_date).days + 1
                                    
                                    # Calculate attendance percentage
                                    attendance_percent = len(unique_dates) / days_since_reg * 100
                                    
                                    st.metric("Attendance Percentage", f"{attendance_percent:.1f}%")
                                    
                                    # Display attendance chart
                                    st.subheader("Attendance History")
                                    
                                    # Create date range from registration to today
                                    date_range = pd.date_range(reg_date, today)
                                    
                                    # Create a dataframe with all dates
                                    attendance_data = pd.DataFrame({
                                        'Date': date_range,
                                        'Present': 0
                                    })
                                    
                                    # Mark days present
                                    for date in unique_dates:
                                        attendance_data.loc[attendance_data['Date'] == date, 'Present'] = 1
                                    
                                    # Create a chart
                                    fig, ax = plt.subplots(figsize=(10, 3))
                                    ax.bar(attendance_data['Date'], attendance_data['Present'], color='green', alpha=0.7)
                                    ax.set_ylim(0, 1.2)
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Attendance')
                                    ax.set_yticks([0, 1])
                                    ax.set_yticklabels(['Absent', 'Present'])
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                        
                        # Download button
                        if st.button("Download Student's Report"):
                            # Create Excel file
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                display_df.to_excel(writer, index=False, sheet_name='Attendance')
                                
                                # Add some formatting
                                workbook = writer.book
                                worksheet = writer.sheets['Attendance']
                                
                                # Add a header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'fg_color': '#D7E4BC',
                                    'border': 1
                                })
                                
                                # Write the column headers with the defined format
                                for col_num, value in enumerate(display_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Set column widths
                                worksheet.set_column('A:A', 15)
                                worksheet.set_column('B:B', 15)
                                worksheet.set_column('C:C', 10)
                            
                            # Generate download link
                            excel_data = output.getvalue()
                            b64 = base64.b64encode(excel_data).decode()
                            student_name_file = student_name.replace(' ', '_')
                            dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{student_name_file}_{roll_number}_attendance.xlsx">Download {student_name}\'s Attendance</a>'
                            st.markdown(dl_link, unsafe_allow_html=True)
                    else:
                        st.info("No attendance records for this student.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with export_tab:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Export Attendance Data")
                
                # Date range selection
                col1, col2 = st.columns(2)
                
                with col1:
                    min_date = df['Date'].min()
                    max_date = df['Date'].max()
                    start_date = st.date_input("Start Date", min_date)
                
                with col2:
                    end_date = st.date_input("End Date", max_date)
                
                # Filter data by date range
                date_filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not date_filtered_df.empty:
                    st.write(f"Found {len(date_filtered_df)} attendance records between {start_date} and {end_date}")
                    
                    # Create summary view
                    st.subheader("Summary")
                    
                    # Get unique students and dates
                    unique_students = date_filtered_df['Name'].unique()
                    unique_dates = sorted(date_filtered_df['Date'].unique())
                    
                    # Create summary dataframe
                    summary_data = []
                    
                    for student in unique_students:
                        student_records = date_filtered_df[date_filtered_df['Name'] == student]
                        student_dates = student_records['Date'].unique()
                        
                        # Format student name
                        display_name = student.split('_')[0].replace('_', ' ')
                        if '_' in student:
                            roll_number = student.split('_')[1]
                        else:
                            roll_number = ""
                        
                        # Calculate attendance percentage
                        attendance_pct = len(student_dates) / len(unique_dates) * 100
                        
                        summary_data.append({
                            'Name': display_name,
                            'Roll Number': roll_number,
                            'Days Present': len(student_dates),
                            'Total Days': len(unique_dates),
                            'Attendance %': f"{attendance_pct:.1f}%"
                        })
                    
                    # Convert to dataframe
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Show the summary
                    st.dataframe(summary_df)
                    
                    # Option to download different formats
                    st.subheader("Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Download Summary Report"):
                            # Create Excel file
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                summary_df.to_excel(writer, index=False, sheet_name='Summary')
                                
                                # Add some formatting
                                workbook = writer.book
                                worksheet = writer.sheets['Summary']
                                
                                # Add a header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'fg_color': '#D7E4BC',
                                    'border': 1
                                })
                                
# Write the column headers with the defined format
                                for col_num, value in enumerate(summary_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Set column widths
                                worksheet.set_column('A:A', 20)
                                worksheet.set_column('B:B', 15)
                                worksheet.set_column('C:C', 12)
                                worksheet.set_column('D:D', 10)
                                worksheet.set_column('E:E', 12)
                            
                            # Generate download link
                            excel_data = output.getvalue()
                            b64 = base64.b64encode(excel_data).decode()
                            dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="attendance_summary_{start_date}_to_{end_date}.xlsx">Download Summary Report</a>'
                            st.markdown(dl_link, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("Download Detailed Report"):
                            # Create detailed dataframe with all records
                            detailed_df = date_filtered_df.copy()
                            detailed_df['Name'] = detailed_df['Name'].apply(lambda x: x.split('_')[0].replace('_', ' '))
                            detailed_df['Roll Number'] = detailed_df['Name'].apply(lambda x: x.split('_')[1] if '_' in x else '')
                            detailed_df['Date'] = detailed_df['DateTime'].dt.strftime('%Y-%m-%d')
                            detailed_df['Time'] = detailed_df['DateTime'].dt.strftime('%H:%M:%S')
                            
                            # Select columns
                            detailed_df = detailed_df[['Name', 'Roll Number', 'Date', 'Time', 'Confidence']]
                            
                            # Create Excel file with detailed data and pivoted attendance
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Detailed sheet
                                detailed_df.to_excel(writer, sheet_name='Detailed Records', index=False)
                                
                                # Create attendance pivot table (student vs date)
                                pivot_df = pd.pivot_table(
                                    date_filtered_df,
                                    index='Name',
                                    columns=pd.to_datetime(date_filtered_df['Date']).dt.strftime('%Y-%m-%d'),
                                    values='Confidence',
                                    aggfunc='first',
                                    fill_value=''
                                )
                                
                                # Convert pivot to binary present/absent
                                binary_pivot = pivot_df.copy()
                                binary_pivot = binary_pivot.notnull().astype(int)
                                binary_pivot = binary_pivot.replace({1: 'Present', 0: 'Absent'})
                                
                                # Format names in pivot
                                binary_pivot.index = [idx.split('_')[0].replace('_', ' ') for idx in binary_pivot.index]
                                
                                # Save pivot to Excel
                                binary_pivot.to_excel(writer, sheet_name='Attendance Matrix')
                                
                                # Add some formatting
                                workbook = writer.book
                                
                                # Format detailed sheet
                                ws1 = writer.sheets['Detailed Records']
                                header_format = workbook.add_format({
                                    'bold': True, 'text_wrap': True, 'valign': 'top',
                                    'fg_color': '#D7E4BC', 'border': 1
                                })
                                
                                for col_num, value in enumerate(detailed_df.columns.values):
                                    ws1.write(0, col_num, value, header_format)
                                
                                # Format pivot sheet
                                ws2 = writer.sheets['Attendance Matrix']
                                present_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                                absent_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                                
                                # Apply conditional formatting
                                for row in range(1, len(binary_pivot.index) + 1):
                                    for col in range(1, len(binary_pivot.columns) + 1):
                                        ws2.conditional_format(row, col, row, col, {
                                            'type': 'cell',
                                            'criteria': 'equal to',
                                            'value': '"Present"',
                                            'format': present_format
                                        })
                                        ws2.conditional_format(row, col, row, col, {
                                            'type': 'cell',
                                            'criteria': 'equal to',
                                            'value': '"Absent"',
                                            'format': absent_format
                                        })
                            
                            # Generate download link
                            excel_data = output.getvalue()
                            b64 = base64.b64encode(excel_data).decode()
                            dl_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="attendance_detailed_{start_date}_to_{end_date}.xlsx">Download Detailed Report</a>'
                            st.markdown(dl_link, unsafe_allow_html=True)
                    
                    # Create a backup of all attendance data
                    if st.button("Create Backup of All Attendance Data"):
                        # Make a copy of the original file with timestamp
                        backup_file = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        if os.path.exists(ATTENDANCE_FILE):
                            # Read the entire file
                            with open(ATTENDANCE_FILE, 'rb') as f:
                                csv_data = f.read()
                                
                            # Create download link
                            b64 = base64.b64encode(csv_data).decode()
                            dl_link = f'<a href="data:text/csv;base64,{b64}" download="{backup_file}">Download Complete Attendance Backup (CSV)</a>'
                            st.markdown(dl_link, unsafe_allow_html=True)
                else:
                    st.info("No attendance records found for the selected date range.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No attendance records found. Take attendance first.")
    else:
        st.info("No attendance records found. Take attendance first.")

# Settings page
elif menu == "Settings":
    st.markdown("<h1 class='main-header'>System Settings</h1>", unsafe_allow_html=True)
    
    # Create tabs for different settings
    general_tab, data_tab, advanced_tab = st.tabs(["General Settings", "Data Management", "Advanced Settings"])
    
    with general_tab:
        st.markdown("""
        <div class='card'>
            <h3>Face Recognition Settings</h3>
        """, unsafe_allow_html=True)
        
        # Face recognition threshold
        recognition_threshold = st.slider(
            "Recognition Threshold",
            min_value=0.10,
            max_value=0.50,
            value=FACE_RECOGNITION_THRESHOLD,
            step=0.01,
            help="Lower value is stricter matching. Recommended: 0.35-0.40"
        )
        
        # Save settings
        if st.button("Update Recognition Threshold"):
            globals()['FACE_RECOGNITION_THRESHOLD'] = recognition_threshold
            st.success(f"Recognition threshold updated to {recognition_threshold}")
        
        # Other settings
        st.markdown("""
        <div class='card'>
            <h3>Attendance Settings</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            consecutive_frames = st.number_input(
                "Consecutive Frames Required",
                min_value=1,
                max_value=10,
                value=CONSECUTIVE_FRAMES_REQUIRED,
                help="Number of consecutive recognitions before marking attendance"
            )
        
        with col2:
            min_interval = st.number_input(
                "Minimum Recognition Interval (seconds)",
                min_value=10,
                max_value=300,
                value=MIN_RECOGNITION_INTERVAL,
                help="Minimum time between recognitions of the same person"
            )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Minimum Confidence for Attendance",
            min_value=0.30,
            max_value=0.90,
            value=MIN_CONFIDENCE_FOR_ATTENDANCE,
            step=0.05,
            help="Minimum confidence level required to mark attendance"
        )
        
        # Update settings
        if st.button("Update Attendance Settings"):
            globals()['CONSECUTIVE_FRAMES_REQUIRED'] = consecutive_frames
            globals()['MIN_RECOGNITION_INTERVAL'] = min_interval
            globals()['MIN_CONFIDENCE_FOR_ATTENDANCE'] = confidence_threshold
            st.success("Attendance settings updated successfully!")
        
    with data_tab:
        st.markdown("""
            <div class='card'>
            <h3>Data Management</h3>
        """, unsafe_allow_html=True)

        st.subheader("<h4>System Backup</h4>")
        # Add backup logic here if needed

    with advanced_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Advanced Settings</h3>")
        # Add advanced settings logic here if needed
        st.text("</div>")

    with data_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data Management")
        
        # Backup options
        st.markdown("<h4>System Backup</h4>", unsafe_allow_html=True)  # Changed from st.text to st.markdown
        
        # Rest of your code (e.g., backup button, reset options, etc.)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Create Full System Backup"):
            # Create a zip archive of embeddings, student data and attendance
            try:
                import zipfile
                from io import BytesIO
                
                # Create a BytesIO object
                zip_buffer = BytesIO()
                
                # Create a zip file
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add embeddings file
                    if os.path.exists(EMBEDDINGS_FILE):
                        zipf.write(EMBEDDINGS_FILE, os.path.basename(EMBEDDINGS_FILE))
                    
                    # Add student data
                    if os.path.exists(STUDENT_DATA_FILE):
                        zipf.write(STUDENT_DATA_FILE, os.path.basename(STUDENT_DATA_FILE))
                    
                    # Add attendance data
                    if os.path.exists(ATTENDANCE_FILE):
                        zipf.write(ATTENDANCE_FILE, os.path.basename(ATTENDANCE_FILE))
                    
                    # Add face images directory
                    if os.path.exists(FACE_IMAGES_DIR):
                        for foldername, subfolders, filenames in os.walk(FACE_IMAGES_DIR):
                            for filename in filenames:
                                file_path = os.path.join(foldername, filename)
                                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(FACE_IMAGES_DIR)))
                
                # Get the zip content
                zip_buffer.seek(0)
                zip_data = zip_buffer.getvalue()
                
                # Create download link
                b64 = base64.b64encode(zip_data).decode()
                backup_name = f"ai-ttendance-full-backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dl_link = f'<a href="data:application/zip;base64,{b64}" download="{backup_name}.zip">Download Complete System Backup</a>'
                st.markdown(dl_link, unsafe_allow_html=True)
                st.success("System backup created successfully!")
            except Exception as e:
                st.error(f"Error creating backup: {e}")
        
        # Reset options
        st.subheader("Reset Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset All Attendance Data", key="reset_attendance"):
                st.warning("This will delete ALL attendance records. This action cannot be undone.")
                if st.button("Confirm Reset Attendance", key="confirm_reset_attendance"):
                    # Create a backup first
                    if os.path.exists(ATTENDANCE_FILE):
                        backup_path = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        shutil.copy(ATTENDANCE_FILE, backup_path)
                        
                        # Create a new attendance file with header
                        with open(ATTENDANCE_FILE, "w") as f:
                            f.write("Name,DateTime,Confidence\n")
                        
                        st.success("All attendance data has been reset. A backup was created.")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.info("No attendance data found to reset.")
            
        with col2:
            if st.button("Reset All Student Data", key="reset_all_students"):
                st.warning("This will delete ALL student data including face images. This action cannot be undone.")
                if st.button("Confirm Reset All Student Data", key="confirm_reset_all"):
                    # Clear embeddings data
                    processor = st.session_state.face_processor
                    processor.embeddings_data = {}
                    processor.save_embeddings()
                    
                    # Clear student dataframe
                    processor.student_data = pd.DataFrame(columns=["Name", "Roll Number", "Registration Date"])
                    processor.save_student_data()
                    
                    # Remove all student directories
                    if os.path.exists(FACE_IMAGES_DIR):
                        import shutil
                        for item in os.listdir(FACE_IMAGES_DIR):
                            item_path = os.path.join(FACE_IMAGES_DIR, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                    
                    st.success("All student data has been reset.")
                    time.sleep(1)
                    st.experimental_rerun()
        # Restore backup
        st.subheader("Restore From Backup")
        
        # Option to upload and restore backup
        uploaded_backup = st.file_uploader("Upload Backup File (.zip or .pkl)", type=["zip", "pkl"])
        
        if uploaded_backup is not None:
            if uploaded_backup.name.endswith('.pkl'):
                # Try to restore embeddings
                try:
                    embeddings_data = pickle.loads(uploaded_backup.read())
                    st.session_state.face_processor.embeddings_data = embeddings_data
                    st.session_state.face_processor.save_embeddings()
                    st.success("Embeddings restored successfully!")
                except Exception as e:
                    st.error(f"Error restoring embeddings: {e}")
            
            elif uploaded_backup.name.endswith('.zip'):
                # Try to restore full backup
                try:
                    import zipfile
                    from io import BytesIO
                    
                    # Extract to a temp directory
                    temp_dir = BASE_DIR / "temp_restore"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    with zipfile.ZipFile(BytesIO(uploaded_backup.read())) as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Restore embeddings
                    embeddings_path = temp_dir / os.path.basename(EMBEDDINGS_FILE)
                    if os.path.exists(embeddings_path):
                        shutil.copy(embeddings_path, EMBEDDINGS_FILE)
                        st.session_state.face_processor.load_embeddings()
                    
                    # Restore student data
                    student_data_path = temp_dir / os.path.basename(STUDENT_DATA_FILE)
                    if os.path.exists(student_data_path):
                        shutil.copy(student_data_path, STUDENT_DATA_FILE)
                        st.session_state.face_processor.load_student_data()
                    
                    # Restore attendance data
                    attendance_path = temp_dir / os.path.basename(ATTENDANCE_FILE)
                    if os.path.exists(attendance_path):
                        shutil.copy(attendance_path, ATTENDANCE_FILE)
                    
                    # Restore face images
                    face_images_backup = temp_dir / os.path.basename(FACE_IMAGES_DIR)
                    if os.path.exists(face_images_backup):
                        if os.path.exists(FACE_IMAGES_DIR):
                            shutil.rmtree(FACE_IMAGES_DIR)
                        shutil.copytree(face_images_backup, FACE_IMAGES_DIR)
                    
                    # Clean up temp directory
                    shutil.rmtree(temp_dir)
                    
                    st.success("Full system backup restored successfully!")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error restoring backup: {e}")
            else:
                st.error("Invalid backup file format")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    with advanced_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Advanced Settings")
        
        # Show backend information
        st.write("Face Recognition Method:")
        if USE_FACENET:
            st.success("✓ Using FaceNet (high accuracy)")
        elif 'face_recognition' in globals():
            st.warning("⚠️ Using face_recognition library (medium accuracy)")
        else:
            st.error("⚠️ Using basic OpenCV (limited accuracy)")
            
            # Suggestion to install better libraries
            st.write("For better recognition accuracy, install FaceNet or face_recognition:")
            st.code("pip install keras-facenet")
            st.code("pip install face-recognition")
        
        # Camera settings
        st.subheader("Camera Settings")
        
        # Option to select camera
        camera_options = ["Default Camera (0)", "External Camera (1)", "Virtual Camera (2)"]
        selected_camera = st.selectbox("Select Camera", camera_options)
        
        camera_id_mapping = {
            "Default Camera (0)": 0,
            "External Camera (1)": 1,
            "Virtual Camera (2)": 2
        }
        
        if st.button("Test Selected Camera"):
            camera_id = camera_id_mapping[selected_camera]
            
            # Try to open the camera
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                st.error(f"Could not open camera with ID {camera_id}. Please check your camera connection.")
            else:
                st.success(f"Successfully connected to camera {camera_id}")
                
                # Capture one frame to show
                ret, frame = cap.read()
                if ret:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Camera Preview")
                
                # Release the camera
                cap.release()
        
        # Database location settings
        st.subheader("Database Locations")
        
        st.info(f"Face Images Directory: {FACE_IMAGES_DIR}")
        st.info(f"Embeddings File: {EMBEDDINGS_FILE}")
        st.info(f"Attendance File: {ATTENDANCE_FILE}")
        st.info(f"Student Data File: {STUDENT_DATA_FILE}")
        
        # Option to change database locations (for development purposes)
        if st.checkbox("Change Database Locations (Advanced)"):
            st.warning("Changing database locations will require moving existing data manually.")
            
            new_face_dir = st.text_input("Face Images Directory", value=str(FACE_IMAGES_DIR))
            new_embeddings_file = st.text_input("Embeddings File", value=str(EMBEDDINGS_FILE))
            new_attendance_file = st.text_input("Attendance File", value=str(ATTENDANCE_FILE))
            new_student_file = st.text_input("Student Data File", value=str(STUDENT_DATA_FILE))
            
            if st.button("Update Database Locations"):
                st.error("This feature is not yet implemented in the current version.")
            
        # Fix for white boxes issue
        st.subheader("UI Fixes")
        
        if st.button("Fix White Boxes Issue"):
            # This is a workaround for the white boxes issue
            st.markdown("""
            <style>
                /* Reset Streamlit's default white box behavior */
                div[data-testid="stVerticalBlock"] {
                    background-color: transparent !important;
                    border: none !important;
                }

                /* Ensure cards are visible and styled correctly */
                .card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 2px 2px 10px #cccccc;
                margin-bottom: 20px;
            }

                /* Style headers */
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

                /* Ensure tabs and columns are visible */
                .stTabs [data-baseweb] {
                    background-color: transparent !important;
                    border-radius: 10px;
                    padding: 10px;
                }

                .stTabs [role="tab"] {
                    background-color: #f0f0f0 !important;
                    margin: 5px;
                    padding: 10px;
                }

                /* Fix for button alignment */
                .stButton > button {
                    width: auto !important;
                }

                /* Ensure content is not hidden */
                div[data-testid="stMarkdownContainer"] > * {
                    opacity: 1 !important;
                    visibility: visible !important;
                }
            </style>
            """, unsafe_allow_html=True)
            st.success("Applied fix for white boxes issue. Please refresh the page to see changes.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# Apply global CSS fix for white boxes issue
st.markdown("""
<style>
    /* Reset Streamlit's default white box behavior */
    div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
        border: none !important;
    }

    /* Ensure cards are visible and styled correctly */
    .card {
        background-color: rgba(255, 255, 255, 0.9) !important; /* Slight transparency */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px #cccccc;
        margin-bottom: 20px;
        overflow: visible !important; /* Ensure content isn’t clipped */
    }

    /* Style headers */
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

    /* Ensure tabs and columns are visible */
    .stTabs [data-baseweb] {
        background-color: transparent !important;
        border-radius: 10px;
        padding: 10px;
    }

    .stTabs [role="tab"] {
        background-color: #f0f0f0 !important;
        margin: 5px;
        padding: 10px;
    }

    /* Fix for button alignment */
    .stButton > button {
        width: auto !important;
    }

    /* Ensure content is not hidden */
    div[data-testid="stMarkdownContainer"] > * {
        opacity: 1 !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>AI-ttendance System &copy; 2025</p>
    <p>Version 1.0.0</p>
</div>
""", unsafe_allow_html=True)