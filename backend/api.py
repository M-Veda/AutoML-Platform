from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reportlab.platypus import (
    Image,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table
)

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import letter
from backend.train_engine import train_models
from backend.predict_engine import make_prediction

import shutil
import os

app = FastAPI()
report_data = {}
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
# REQUEST MODEL
# =========================

class TrainRequest(BaseModel):

    filename: str

    target_column: str

class PredictRequest(BaseModel):

    features: dict

# =========================
# CREATE DATASET DIRECTORY
# =========================

os.makedirs("datasets", exist_ok=True)

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {
        "message": "AutoML Platform Backend Running Successfully"
    }

# =========================
# DATASET UPLOAD ROUTE
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
# TRAIN MODELS ROUTE
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
# PREDICT ROUTE
# =========================

@app.post("/predict")
def predict(request: PredictRequest):

    result = make_prediction(
        request.features
    )

    return result

# =========================
# DOWNLOAD BEST MODEL
# =========================

@app.get("/download_report")
def download_report():
    
    from reportlab.platypus import (
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
    from reportlab.platypus import Image
    import json

    styles = getSampleStyleSheet()

    pdf_path = "automl_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    # =========================
    # DOWNLOAD BEST MODEL
    # =========================

    @app.get("/download-model")
    def download_model():

        return FileResponse(
            "outputs/best_model.pkl",
            media_type="application/octet-stream",
            filename="best_model.pkl"
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

            report_data.update(report_data)

    except:

        report_data = {}

    # =========================
    # PDF CONTENT
    # =========================

    story = []

    # =========================
    # LOGO
    # =========================

    logo = Image(
        "backend/logo.png",
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

    title_style = styles["Title"]

    title_style.textColor = colors.darkblue

    title = Paragraph(
        "🚀 AutoML Analysis Report",
        title_style
    )

    story.append(title)

    story.append(
        Spacer(1, 20)
    )

    from datetime import datetime

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
        Spacer(1, 15)
    )

    # =========================
    # PROJECT INFO
    # =========================

    info_data = [

        [
            "Best Model",
            report_data.get(
                "best_model",
                "Unknown"
            )
        ],

        [
            "Problem Type",
            report_data.get(
                "problem_type",
                "Unknown"
            )
        ]
    ]

    info_table = Table(
        info_data,
        colWidths=[200, 250]
    )

    info_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.lightblue
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.black
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.black
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
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

        [
            "Rows",
            str(dataset_info.get("rows", "N/A"))
        ],

        [
            "Columns",
            str(dataset_info.get("columns", "N/A"))
        ]
    ]

    dataset_table = Table(
        dataset_data,
        colWidths=[200, 250]
    )

    dataset_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.black
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            )
        ])
    )

    story.append(dataset_table)

    story.append(
        Spacer(1, 25)
    )

    # =========================
    # CREATE PERFORMANCE CHART
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

    # =========================
    # ADD CHART TO PDF
    # =========================

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
    # MODEL PERFORMANCE
    # =========================

    heading = Paragraph(
        "📊 Model Performance Summary",
        styles["Heading2"]
    )

    story.append(heading)

    story.append(
        Spacer(1, 15)
    )

    for model_name, metrics in report_data.get(
        "results",
        {}
    ).items():

        model_title = Paragraph(
            f"<b>{model_name}</b>",
            styles["Heading3"]
        )

        story.append(model_title)

        metric_data = [

            ["Metric", "Value"]
        ]

        for metric_name, value in metrics.items():

            metric_data.append(

                [
                    metric_name,
                    str(round(value, 4))
                ]
            )

        metric_table = Table(
            metric_data,
            colWidths=[250, 150]
        )

        metric_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.darkblue
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.black
                ),

                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.lightgreen
            ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, 0),
                    10
                ),
            ])
        )

        story.append(metric_table)

        story.append(
            Spacer(1, 20)
        )

    # =========================
    # AI RECOMMENDATION
    # =========================

    recommendation_heading = Paragraph(
        "🤖 AI Recommendation",
        styles["Heading2"]
    )

    story.append(
        recommendation_heading
    )

    story.append(
        Spacer(1, 10)
    )

    recommendation_text = Paragraph(

        f"""
        <b>{
            report_data.get(
                'best_model',
                'Unknown'
            )
        }</b>
        achieved the best performance and is recommended
        for production-level predictions.
        """,

        styles["BodyText"]
    )

    story.append(
        recommendation_text
    )

    # =========================
    # FOOTER
    # =========================

    story.append(
        Spacer(1, 30)
    )

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