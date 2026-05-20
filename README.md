# AutoML Platform 🚀

An end-to-end AI-powered AutoML platform that allows users to upload datasets, automatically train multiple machine learning models, compare their performance, generate professional PDF reports, and download the best trained model — all through a modern web interface.

---

## 🌐 Live Demo

### Frontend Deployment

[AutoML Platform Live Demo](https://auto-ml-platform.vercel.app?utm_source=chatgpt.com)

---

# ✨ Features

* 📂 Upload CSV datasets
* 🤖 Automatic ML model training
* 📊 Classification & Regression support
* 🏆 Best model selection
* 📈 Model performance comparison charts
* 📄 Professional PDF report generation
* 📥 Download trained model (.pkl)
* ⚡ Modern React + Vite frontend
* 🔥 FastAPI backend architecture
* 🌍 CORS-enabled API integration

---

# 🧠 Supported Machine Learning Models

## Classification Models

* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* K-Nearest Neighbors
* Support Vector Machine

## Regression Models

* Linear Regression
* Random Forest Regressor
* Decision Tree Regressor

---

# 🛠️ Tech Stack

## Frontend

* React.js
* Vite
* Tailwind CSS
* Axios

## Backend

* FastAPI
* Hypercorn
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* ReportLab
* Joblib

---

# 📁 Project Structure

```bash
AutoML-Platform/
│
├── frontend/          # React Frontend
├── backend/           # FastAPI Backend
├── datasets/          # Uploaded datasets
├── README.md
└── .gitignore
```

---

# ⚙️ Local Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/M-Veda/AutoML-Platform.git
cd AutoML-Platform
```

---

# 🔹 Backend Setup

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt

hypercorn api:app --bind 0.0.0.0:8000
```

Backend runs on:

```bash
http://127.0.0.1:8000
```

---

# 🔹 Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs on:

```bash
http://localhost:5173
```

---

# 📊 Workflow

1. Upload dataset
2. Select target column
3. Train multiple ML models
4. Compare model performance
5. Automatically select best model
6. Generate PDF analysis report
7. Download trained model

---

# 📄 PDF Report Includes

* Dataset summary
* Best model information
* Performance metrics
* AI recommendation
* Model comparison chart
* Timestamped analysis report

---

# 🚀 Deployment

## Frontend

Deployed successfully on:

* [Vercel](https://vercel.com?utm_source=chatgpt.com)

## Backend

Backend deployment configuration included for cloud platforms like:

* Render
* Railway
* PythonAnywhere

> Note: Full ML deployment on free hosting tiers may face storage limitations because of large scientific Python dependencies such as SciPy and Scikit-learn.

---

# 🔮 Future Improvements

* Deep Learning support
* SHAP explainability
* Hyperparameter tuning
* Model leaderboard
* Authentication system
* Cloud dataset storage
* Docker deployment
* Auto feature engineering

---

# 👨‍💻 Author

## Veda

B.Tech AI & ML Student
Passionate about AI, Machine Learning, Automation, and Full-Stack Development.

GitHub:
[M-Veda GitHub Profile](https://github.com/M-Veda?utm_source=chatgpt.com)

---

# ⭐ If you found this project useful

Give this repository a star ⭐ on GitHub.
