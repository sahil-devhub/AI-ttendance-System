<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/91caea8a-f645-42ea-a5f5-ed1b9cad9f34" /><img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/1a3783e3-1dcf-4650-9d27-5eec3c6bf3f2" /># 🤖 AI-Powered Face Recognition Attendance System


<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-purple?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-yellow?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)

</div>

An intelligent and automated attendance system that leverages real-time facial recognition to streamline the process of tracking presence. Built with a clean and interactive web interface using Streamlit.

---

<div align="center">
  <img src="screenshots/image_f26916.png" alt="Live Attendance Monitoring" width="800"/>
</div>

## ✨ Key Features

-   **👨‍💻 Real-time Face Recognition**: Uses a live camera feed to detect and identify registered students automatically.
-   **🚀 Fast & Accurate**: Employs Haar Cascades for robust face detection and a pre-trained Torch model for high-accuracy facial embedding generation.
-   **📝 Easy Student Registration**: A simple form to register new students by uploading 5 of their pictures. The system automatically processes and stores their facial data.
-   **📊 Interactive Dashboard**: A clean web interface to view live attendance logs, check overall attendance records, and manage student data.
-   **📥 Data Export**: Attendance records can be easily searched, viewed, and downloaded as an Excel file.

## 📸 Screenshots

<table align="center">
  <tr>
    <td align="center"><strong>Live Attendance Monitoring</strong></td>
    <td align="center"><strong>Attendance Dashboard</strong></td>
  </tr>
  <tr>
    <td><img src="screenshots/image_f26916.png" alt="Live Feed" width="400"></td>
    <td><img src="screenshots/image_f26879.png" alt="Attendance Dashboard" width="400"></td>
  </tr>
  <tr>
    <td align="center"><strong>Easy Student Registration</strong></td>
    <td align="center"><strong>Registered Students List</strong></td>
  </tr>
  <tr>
    <td><img src="screenshots/image_f2705f.png" alt="Registration Page" width="400"></td>
    <td><img src="screenshots/image_f273c4.png" alt="Student List" width="400"></td>
  </tr>
</table>

## ⚙️ How It Works

The application operates in a few simple steps:

1.  **Registration**: A user registers a new student by providing their name, roll number, and 5 clear photos.
2.  **Embedding Generation**: The system detects the face in each photo, processes them using the `openface.nn4.small2.v1.t7` Torch model to generate a unique numerical vector (an "embedding"), and stores the average embedding for that student.
3.  **Real-time Recognition**: The main application opens a camera stream. For each frame, it detects faces using OpenCV's Haar Cascade classifier.
4.  **Identification & Logging**: For each detected face, it generates an embedding and compares it against the stored embeddings. If a match is found with high confidence, the student's attendance is automatically logged with a timestamp in `attendance.csv`.

## 🛠️ Technology Stack

-   **Backend**: Python
-   **Web Framework**: Streamlit
-   **Computer Vision**: OpenCV
-   **Data Manipulation**: Pandas, NumPy
-   **Facial Embeddings**: Pre-trained PyTorch model (`openface.nn4.small2.v1.t7`)
-   **Data Storage**: Pickle (for embeddings), CSV/Excel (for logs)

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   `pip` package manager
-   A webcam connected to your computer

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing libraries like streamlit, opencv-python, pandas, numpy, scipy, and openpyxl).*

4.  **Download the Face Recognition Model:**
    -   You will need the `openface.nn4.small2.v1.t7` model file. Make sure it is placed in the root directory of the project.

### Running the Application

1.  Execute the following command in your terminal:
    ```bash
    streamlit run app3.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## usage

1.  **Register Students**: Navigate to the `📝 Register New Student` tab from the sidebar. Fill in the details and upload 5 photos.
2.  **Start Attendance**: Go to the `🏠 Home` tab and click the "▶️ Start Camera" button.
3.  **View Records**: Check the `📊 View Attendance` and `👨‍🎓 View Students` tabs to see the logged data.

## 📂 Project Structure
