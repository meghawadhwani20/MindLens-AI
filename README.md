# MindLens AI - Wellness & Stress Tracker

MindLens AI is a comprehensive wellness application that combines daily habit tracking, real-time physiological sensor data, and AI-powered stress analysis to provide users with holistic health insights. Built with **FastAPI** and **Supabase**, it leverages **Computer Vision** and **Generative AI** to assess physical and mental well-being.

## 🚀 Features

- **📊 Daily Habit Tracker**: Log sleep, water intake, screen time, and other daily activities.
- **💓 Real-time Sensor Integration**: Reads **Heart Rate**, **SpO2**, and **Body Temperature** from an ESP32 device via USB Serial (COM5).
  - *Smart Fallback*: Automatically generates realistic simulated data if sensors are not connected or return invalid values.
- **📸 AI Stress Scanner**: Uses **DeepFace** and **MediaPipe** to analyze facial expressions and body posture from uploaded photos to estimate stress levels.
- **🤖 AI Wellness Chatbot**: Integrated OpenAI-powered assistant for personalized wellness advice.
- **🔐 Secure Authentication**: User signup, login, and password management powered by **Supabase**.
- **📈 Dashboard**: Visualize your health trends and historical data.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Frontend**: HTML5, Bootstrap 5, Jinja2 Templates, JavaScript
- **Database**: Supabase (PostgreSQL)
- **AI & ML**: 
  - **OpenCV & MediaPipe**: For posture and face landmark detection.
  - **DeepFace**: For emotion analysis.
  - **OpenAI API**: For the intelligent chatbot.
- **Hardware Interface**: PySerial (for ESP32 communication).

---

## ⚙️ Prerequisites

Before running the project, ensure you have:

1.  **Python 3.10** or higher installed.
2.  A **Supabase** account (for database).
3.  An **OpenAI API Key** (for the chatbot).
4.  *(Optional)* An **ESP32** microcontroller connected via USB (COM5) sending JSON data:
    ```json
    {"heart_rate": 72.5, "spo2": 98.0, "temperature": 36.6}
    ```

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/minlensai.git
cd minlensai
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your credentials:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run the Application
Start the local development server:
```bash
uvicorn backend.main:app --reload
```
The application will be available at: **http://127.0.0.1:8000**

---

## 📖 Usage Guide

1.  **Home & Signup**: Open the app and sign up for a new account.
2.  **Habit Tracking**: Go to the **Habit Tracker** page.
    - If an ESP32 is connected, sensor data will auto-fill every 2 seconds.
    - If no device is found, the system will simulate realistic temperature data for testing.
    - Fill in your daily habits and click **Submit**.
3.  **Stress Scan**: Upload 3-5 photos of yourself. The AI will analyze your expression and posture to generate a stress report.
4.  **Chatbot**: Use the **AI Assistant** page to ask for health tips or advice based on your data.

---

## 📂 Project Structure

```
minlensai-main/
├── backend/                # FastAPI application logic
│   ├── main.py             # App entry point and API routes
│   ├── serial_reader.py    # ESP32 serial communication & data simulation
│   ├── stress_scanner.py   # AI logic for image analysis
│   ├── chatbot.py          # OpenAI chatbot integration
│   ├── db.py               # Supabase connection
│   └── models.py           # Pydantic data models
├── frontend/
│   ├── templates/          # HTML files (Jinja2)
│   └── static/             # CSS, JS, and images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Landing page |
| `GET` | `/latest-data` | Returns real-time or simulated sensor data |
| `POST` | `/submit` | Saves sensor data to the database |
| `POST` | `/submit-habits` | Saves daily habit logs |
| `POST` | `/scan` | Uploads images for AI stress analysis |
| `POST` | `/signup` | User registration |
| `POST` | `/login` | User authentication |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.
