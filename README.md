# 🚦 Traffic Eye - Smart Traffic Violation Detection System
 AI-powered traffic violation detection system using YOLOv8 and OCR for helmet and license plate analysis with automated email alerts. An AI-based system that automatically detects helmetless riders, multiple riders, and expired PUCs from live or uploaded video footage, and issues email penalties to registered vehicle owners.

---

## 🔍 Features

- 🧠 YOLOv8-based helmet and plate detection
- 📸 Image & Video input via Streamlit UI
- 🔍 OCR for license plate recognition
- ✅ PUC expiry validation using registry CSV
- 📧 Automatic penalty email alerts
- 📊 Real-time logs and violation summary

---

## 📁 Folder Structure

bash
traffic-eye/
├── assets/               # Visuals for README/demo
├── data/                 # Input videos/images and vehicle registry CSV
├── models/               # YOLOv8 trained weights
├── src/                  # Detection + Email + OCR + Streamlit app
├── report/               # Project Report      
└── requirements.txt      # Entry point for UI# Dependencies

---

## 🚀 Getting Started

### 1. Clone this repository

bash
git clone https://github.com/yourusername/traffic-eye.git
cd traffic-eye


### 2. Install dependencies

bash
pip install -r requirements.txt


### 3. Launch the Streamlit App

bash
streamlit run src/streamlit_app.py


---

## 🛠 Requirements

* Python 3.8+
* Streamlit
* Ultralytics YOLOv8
* OpenCV, NumPy, Pandas
* pytesseract (for OCR)
* SMTP (Gmail App Password) for email alerts

---


## 📄 License

[MIT](LICENSE)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss.

---

## 🙋‍♂ Author

Made with ❤ by Darsh Shah

