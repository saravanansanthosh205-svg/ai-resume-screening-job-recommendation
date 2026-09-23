import os
import re

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session
)

import mysql.connector

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "ai-resume-screening-project-secret"
)


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# JOB PROFILES
# =========================================================

JOB_PROFILES = [

    {
        "title": "Python Developer",
        "description": """
        Python Flask Django REST API MySQL SQL backend
        programming software development database
        object oriented programming
        """
    },

    {
        "title": "Full Stack Developer",
        "description": """
        Python Flask HTML CSS JavaScript React
        MySQL SQL REST API frontend backend
        web development full stack development
        """
    },

    {
        "title": "Frontend Developer",
        "description": """
        HTML CSS JavaScript React frontend
        responsive web design UI UX Bootstrap
        web development
        """
    },

    {
        "title": "Data Analyst",
        "description": """
        Python SQL MySQL Excel pandas numpy
        data analysis data visualization statistics
        Power BI
        """
    },

    {
        "title": "AI / ML Intern",
        "description": """
        Python machine learning artificial intelligence
        AI data science pandas numpy scikit learn
        algorithms statistics
        """
    },

    {
        "title": "Software Developer",
        "description": """
        Python Java JavaScript programming
        software development SQL MySQL
        OOP data structures algorithms
        """
    }

]


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    try:

        connection = mysql.connector.connect(

            host=os.getenv(
                "DB_HOST",
                "localhost"
            ),

            user=os.getenv(
                "DB_USER",
                "root"
            ),

            password=os.getenv(
                "DB_PASSWORD",
                "saravanan2005"
            ),

            database=os.getenv(
                "DB_NAME",
                "resume_db"
            )

        )

        return connection

    except mysql.connector.Error as error:

        print(
            "Database connection error:",
            error
        )

        return None


# =========================================================
# EXTRACT TEXT FROM PDF / TXT
# =========================================================

def extract_text(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    # -------------------------
    # TXT FILE
    # -------------------------

    if extension == ".txt":

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                return file.read()

        except Exception as error:

            print(
                "TXT reading error:",
                error
            )

            return ""


    # -------------------------
    # PDF FILE
    # -------------------------

    if extension == ".pdf":

        text = ""

        try:

            reader = PdfReader(
                file_path
            )

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

        except Exception as error:

            print(
                "PDF reading error:",
                error
            )

            return ""

        return text


    return ""


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SKILLS
# =========================================================

SKILLS = [

    "python",
    "java",
    "javascript",
    "html",
    "css",
    "react",
    "flask",
    "django",
    "mysql",
    "sql",
    "rest api",
    "api",
    "bootstrap",
    "php",
    "c",
    "c++",
    "excel",
    "pandas",
    "numpy",
    "machine learning",
    "artificial intelligence",
    "ai",
    "data science",
    "power bi",
    "git",
    "github",
    "docker",
    "linux",
    "oop",
    "data structures",
    "algorithms",
    "web development"
]


# =========================================================
# EXTRACT SKILLS
# =========================================================

def extract_skills(text):

    normalized = normalize_text(
        text
    )

    detected = []

    for skill in SKILLS:

        skill_normalized = normalize_text(
            skill
        )

        if skill_normalized in normalized:

            detected.append(
                skill
            )

    return sorted(
        list(
            set(detected)
        )
    )


# =========================================================
# CALCULATE JOB RECOMMENDATIONS
# =========================================================

def calculate_recommendations(
    resume_text
):

    normalized_resume = normalize_text(
        resume_text
    )

    detected_skills = extract_skills(
        resume_text
    )

    recommendations = []


    # -----------------------------------------------------
    # TF-IDF NLP
    # -----------------------------------------------------

    documents = [

        normalized_resume

    ] + [

        normalize_text(
            job["description"]
        )

        for job in JOB_PROFILES

    ]


    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    try:

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

        similarity_scores = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:]
        )[0]

    except Exception as error:

        print(
            "NLP error:",
            error
        )

        similarity_scores = [
            0
            for _ in JOB_PROFILES
        ]


    # -----------------------------------------------------
    # PROCESS EACH JOB
    # -----------------------------------------------------

    for index, job in enumerate(
        JOB_PROFILES
    ):

        nlp_score = (
            similarity_scores[index]
            * 100
        )


        # Job skills

        job_skills = extract_skills(
            job["description"]
        )


        # Matching skills

        matched_skills = [

            skill

            for skill in job_skills

            if skill in detected_skills

        ]


        # Skill score

        if job_skills:

            skill_score = (
                len(matched_skills)
                / len(job_skills)
            ) * 100

        else:

            skill_score = 0


        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        final_score = (

            nlp_score * 0.60

        ) + (

            skill_score * 0.40

        )


        recommendations.append({

            "title": job["title"],

            "description": job["description"].strip(),

            "score": round(
                final_score
            ),

            "nlp_score": round(
                nlp_score
            ),

            "skill_score": round(
                skill_score
            ),

            "matched_skills": matched_skills

        })


    # -----------------------------------------------------
    # SORT HIGH TO LOW
    # -----------------------------------------------------

    recommendations.sort(

        key=lambda x: x["score"],

        reverse=True

    )


    return (
        recommendations,
        detected_skills
    )


# =========================================================
# SAVE RESULT TO MYSQL
# =========================================================

def save_result(
    resume_name,
    detected_skills,
    recommendations
):

    connection = get_connection()

    if not connection:

        return


    cursor = None

    try:

        cursor = connection.cursor()


        # -------------------------------------------------
        # CREATE TABLE
        # -------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS
            resume_results (

                id INT AUTO_INCREMENT PRIMARY KEY,

                resume_name VARCHAR(255),

                detected_skills TEXT,

                top_job VARCHAR(255),

                top_score INT,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
            """
        )


        # -------------------------------------------------
        # TOP JOB
        # -------------------------------------------------

        top_job = ""

        top_score = 0

        if recommendations:

            top_job = recommendations[0][
                "title"
            ]

            top_score = recommendations[0][
                "score"
            ]


        # -------------------------------------------------
        # INSERT
        # -------------------------------------------------

        cursor.execute(

            """
            INSERT INTO resume_results
            (
                resume_name,
                detected_skills,
                top_job,
                top_score
            )

            VALUES
            (
                %s,
                %s,
                %s,
                %s
            )
            """,

            (

                resume_name,

                ", ".join(
                    detected_skills
                ),

                top_job,

                top_score

            )

        )


        connection.commit()


    except mysql.connector.Error as error:

        print(
            "Database save error:",
            error
        )


    finally:

        if cursor:

            cursor.close()

        connection.close()


# =========================================================
# HOME PAGE
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template(

        "index.html",

        results=None,

        skills=[],

        resume_name=None,

        error=None

    )


# =========================================================
# ANALYZE RESUME
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    uploaded_file = request.files.get(
        "resume"
    )


    # -----------------------------------------------------
    # FILE CHECK
    # -----------------------------------------------------

    if (

        not uploaded_file

        or not uploaded_file.filename

    ):

        return render_template(

            "index.html",

            results=None,

            skills=[],

            resume_name=None,

            error="Please select a resume."

        )


    # -----------------------------------------------------
    # EXTENSION CHECK
    # -----------------------------------------------------

    extension = os.path.splitext(

        uploaded_file.filename

    )[1].lower()


    if extension not in [

        ".pdf",

        ".txt"

    ]:

        return render_template(

            "index.html",

            results=None,

            skills=[],

            resume_name=None,

            error=(
                "Only PDF and TXT "
                "resumes are supported."
            )

        )


    # -----------------------------------------------------
    # FILE NAME
    # -----------------------------------------------------

    resume_name = uploaded_file.filename


    safe_name = re.sub(

        r"[^A-Za-z0-9._-]",

        "_",

        resume_name

    )


    file_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        safe_name

    )


    uploaded_file.save(
        file_path
    )


    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    resume_text = extract_text(
        file_path
    )


    if not resume_text.strip():

        return render_template(

            "index.html",

            results=None,

            skills=[],

            resume_name=resume_name,

            error=(

                "Could not read the resume. "

                "Please upload a text-based "

                "PDF or TXT file."

            )

        )


    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    (

        recommendations,

        detected_skills

    ) = calculate_recommendations(

        resume_text

    )


    # -----------------------------------------------------
    # SAVE TO DATABASE
    # -----------------------------------------------------

    save_result(

        resume_name,

        detected_skills,

        recommendations

    )


    # -----------------------------------------------------
    # SAVE SESSION
    # -----------------------------------------------------

    session["analysis"] = {

        "resume_name": resume_name,

        "skills": detected_skills,

        "results": recommendations

    }


    # -----------------------------------------------------
    # SHOW RESULTS
    # -----------------------------------------------------

    return render_template(

        "index.html",

        results=recommendations,

        skills=detected_skills,

        resume_name=resume_name,

        error=None

    )


# =========================================================
# DOWNLOAD PDF REPORT
# =========================================================

@app.route(
    "/download-report"
)
def download_report():

    analysis = session.get(
        "analysis"
    )


    if not analysis:

        return (

            "No analysis available. "
            "Please analyze a resume first.",

            404

        )


    file_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        "AI_Resume_Analysis_Report.pdf"

    )


    # -----------------------------------------------------
    # CREATE PDF
    # -----------------------------------------------------

    pdf = canvas.Canvas(

        file_path,

        pagesize=A4

    )


    width, height = A4

    y = height - 50


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(

        50,

        y,

        "AI Resume Analysis Report"

    )


    y -= 35


    # -----------------------------------------------------
    # RESUME NAME
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        11
    )

    pdf.drawString(

        50,

        y,

        "Resume:"

    )


    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(

        120,

        y,

        analysis["resume_name"]

    )


    y -= 30


    # -----------------------------------------------------
    # DETECTED SKILLS
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(

        50,

        y,

        "Detected Skills"

    )


    y -= 20


    pdf.setFont(
        "Helvetica",
        10
    )


    skills_text = ", ".join(

        analysis["skills"]

    )


    if not skills_text:

        skills_text = "No specific skills detected."


    pdf.drawString(

        50,

        y,

        skills_text[:100]

    )


    y -= 35


    # -----------------------------------------------------
    # TOP RECOMMENDATION
    # -----------------------------------------------------

    results = analysis["results"]


    if results:

        top = results[0]


        pdf.setFont(
            "Helvetica-Bold",
            13
        )

        pdf.drawString(

            50,

            y,

            "Top Job Recommendation"

        )


        y -= 22


        pdf.setFont(
            "Helvetica-Bold",
            11
        )

        pdf.drawString(

            50,

            y,

            top["title"]

        )


        pdf.setFont(
            "Helvetica",
            11
        )

        pdf.drawString(

            250,

            y,

            f"Match Score: {top['score']}%"

        )


        y -= 25


    # -----------------------------------------------------
    # ALL JOB RESULTS
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        13
    )

    pdf.drawString(

        50,

        y,

        "Job Recommendation Results"

    )


    y -= 25


    for number, job in enumerate(
        results,
        start=1
    ):

        # New page if needed

        if y < 130:

            pdf.showPage()

            y = height - 50


        pdf.setFont(
            "Helvetica-Bold",
            11
        )


        pdf.drawString(

            50,

            y,

            f"{number}. {job['title']}"

        )


        y -= 18


        pdf.setFont(
            "Helvetica",
            10
        )


        pdf.drawString(

            70,

            y,

            f"Final Match Score: {job['score']}%"

        )


        y -= 16


        pdf.drawString(

            70,

            y,

            f"NLP Similarity: {job['nlp_score']}%"

        )


        y -= 16


        pdf.drawString(

            70,

            y,

            f"Skill Match: {job['skill_score']}%"

        )


        y -= 16


        matched = ", ".join(

            job["matched_skills"]

        )


        if not matched:

            matched = "None"


        # PDF-safe text

        matched = matched[:100]


        pdf.drawString(

            70,

            y,

            f"Matched Skills: {matched}"

        )


        y -= 28


    # -----------------------------------------------------
    # SCORE FORMULA
    # -----------------------------------------------------

    if y < 100:

        pdf.showPage()

        y = height - 50


    pdf.setFont(
        "Helvetica-Bold",
        12
    )


    pdf.drawString(

        50,

        y,

        "Score Calculation"

    )


    y -= 20


    pdf.setFont(
        "Helvetica",
        10
    )


    pdf.drawString(

        50,

        y,

        "Final Score = 60% NLP Similarity + 40% Skill Match"

    )


    y -= 35


    pdf.setFont(
        "Helvetica-Oblique",
        9
    )


    pdf.drawString(

        50,

        y,

        "This report is generated by an AI-based resume"

    )


    y -= 14


    pdf.drawString(

        50,

        y,

        "screening system. Results are recommendations"

    )


    y -= 14


    pdf.drawString(

        50,

        y,

        "and should not be considered as final hiring decisions."

    )


    # -----------------------------------------------------
    # SAVE PDF
    # -----------------------------------------------------

    pdf.save()


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    return send_file(

        file_path,

        as_attachment=True,

        download_name=(
            "AI_Resume_Analysis_Report.pdf"
        ),

        mimetype="application/pdf"

    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health"
)
def health():

    return {

        "status": "ok",

        "message": (
            "AI Resume Screening "
            "API is running"
        )

    }


# =========================================================
# RUN LOCAL
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),

        debug=True

    )
