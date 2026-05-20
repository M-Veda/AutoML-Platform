from datetime import datetime
import shutil
import os
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from train_engine import train_models
from predict_engine import make_prediction

from reportlab.platypus import (
    Image,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

import matplotlib.pyplot as plt

app = FastAPI()

latest_training_results = {}

# =========================
# ENABLE CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CREATE DATASET DIRECTORY
# =========================

os.makedirs("datasets", exist_ok=True)

# =========================
# REQUEST MODELS
# =========================

class TrainRequest(BaseModel):

    filename: str
    target_column: str

class PredictRequest(BaseModel):

    features: dict

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {
        "message": "AutoML Platform Backend Running Successfully"
    }

# =========================
# DATASET UPLOAD
# =========================

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):

    file_path = f"datasets/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Dataset uploaded successfully",
        "filename": file.filename,
        "saved_path": file_path
    }

# =========================
# TRAIN MODELS
# =========================

@app.post("/train-models")
def train_ml_models(request: TrainRequest):

    global latest_training_results

    dataset_path = f"datasets/{request.filename}"

    result = train_models(
        dataset_path,
        request.target_column
    )

    latest_training_results = result

    return result

# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict(request: PredictRequest):

    result = make_prediction(
        request.features
    )

    return result

# =========================
# DOWNLOAD MODEL
# =========================

@app.get("/download-model")
def download_model():

    return FileResponse(
        "outputs/best_model.pkl",
        media_type="application/octet-stream",
        filename="best_model.pkl"
    )

# =========================
# DOWNLOAD REPORT
# =========================

@app.get("/download_report")
def download_report():

    styles = getSampleStyleSheet()

    pdf_path = "automl_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    # =========================
    # LOAD TRAINING RESULTS
    # =========================

    try:

        with open(
            "outputs/training_results.json",
            "r"
        ) as f:

            report_data = json.load(f)

    except:

        report_data = {}

    story = []

    # =========================
    # LOGO
    # =========================

    if os.path.exists("logo.png"):

        logo = Image(
            "logo.png",
            width=120,
            height=120
        )

        story.append(logo)

        story.append(
            Spacer(1, 20)
        )

    # =========================
    # TITLE
    # =========================

    title = Paragraph(
        "🚀 AutoML Analysis Report",
        styles["Title"]
    )

    story.append(title)

    story.append(
        Spacer(1, 20)
    )

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    story.append(
        Paragraph(
            f"Generated On: {current_time}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # =========================
    # PROJECT INFO TABLE
    # =========================

    info_data = [

        ["Best Model",
         report_data.get("best_model", "Unknown")],

        ["Problem Type",
         report_data.get("problem_type", "Unknown")]
    ]

    info_table = Table(
        info_data,
        colWidths=[200, 250]
    )

    info_table.setStyle(
        TableStyle([

            ("BACKGROUND",
             (0, 0),
             (-1, -1),
             colors.lightblue),

            ("GRID",
             (0, 0),
             (-1, -1),
             1,
             colors.black),

            ("FONTNAME",
             (0, 0),
             (-1, -1),
             "Helvetica-Bold")
        ])
    )

    story.append(info_table)

    story.append(
        Spacer(1, 25)
    )

    # =========================
    # DATASET INFO
    # =========================

    dataset_info = report_data.get(
        "dataset_info",
        {}
    )

    dataset_data = [

        ["Rows",
         str(dataset_info.get("rows", "N/A"))],

        ["Columns",
         str(dataset_info.get("columns", "N/A"))]
    ]

    dataset_table = Table(
        dataset_data,
        colWidths=[200, 250]
    )

    dataset_table.setStyle(
        TableStyle([

            ("BACKGROUND",
             (0, 0),
             (-1, -1),
             colors.lightgrey),

            ("GRID",
             (0, 0),
             (-1, -1),
             1,
             colors.black),

            ("FONTNAME",
             (0, 0),
             (-1, -1),
             "Helvetica-Bold")
        ])
    )

    story.append(dataset_table)

    story.append(
        Spacer(1, 25)
    )

    # =========================
    # PERFORMANCE CHART
    # =========================

    model_names = []
    model_scores = []

    for model_name, metrics in report_data.get(
        "results",
        {}
    ).items():

        score = (
            metrics.get("Accuracy")
            or metrics.get("R2 Score")
            or metrics.get("F1 Score")
            or 0
        )

        model_names.append(model_name)
        model_scores.append(score)

    if model_names:

        plt.figure(figsize=(10, 5))

        plt.bar(
            model_names,
            model_scores
        )

        plt.xlabel("Models")
        plt.ylabel("Performance Score")
        plt.title("Model Performance Comparison")

        plt.xticks(rotation=20)

        chart_path = "model_performance_chart.png"

        plt.tight_layout()

        plt.savefig(chart_path)

        plt.close()

        chart_heading = Paragraph(
            "📈 Performance Comparison Chart",
            styles["Heading2"]
        )

        story.append(chart_heading)

        story.append(
            Spacer(1, 15)
        )

        chart_image = Image(
            chart_path,
            width=450,
            height=250
        )

        story.append(chart_image)

        story.append(
            Spacer(1, 25)
        )

    # =========================
    # FOOTER
    # =========================

    footer = Paragraph(
        "Generated by AutoML Platform",
        styles["Italic"]
    )

    story.append(footer)

    # =========================
    # BUILD PDF
    # =========================

    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="automl_report.pdf"
    )