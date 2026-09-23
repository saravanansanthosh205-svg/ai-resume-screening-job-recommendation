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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "ai-resume-screening-project-secret"
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# =========================================================
# JOB PROFILES
# =========================================================

JOB_PROFILES = [

    {
        "title": "Python Developer",

        "skills": [
            "python",
            "flask",
            "django",
            "mysql",
            "sql",
            "git",
            "api",
            "rest"
        ],

        "description":
            "Develop Python applications, APIs and backend services.",

        "keywords":
            "python flask django mysql sql git api rest "
            "backend programming software development"
    },


    {
        "title": "Full Stack Developer",

        "skills": [
            "python",
            "html",
            "css",
            "javascript",
            "react",
            "mysql",
            "flask",
            "api"
        ],

        "description":
            "Build complete web applications using frontend and backend technologies.",

        "keywords":
            "python html css javascript react mysql flask "
            "api full stack frontend backend web development"
    },


    {
        "title": "Frontend Developer",

        "skills": [
            "html",
            "css",
            "javascript",
            "react",
            "bootstrap",
            "responsive",
            "git"
        ],

        "description":
            "Create responsive and interactive web interfaces.",

        "keywords":
            "html css javascript react bootstrap "
            "responsive ui frontend web git"
    },


    {
        "title": "Data Analyst",

        "skills": [
            "python",
            "sql",
            "excel",
            "pandas",
            "numpy",
            "statistics",
            "power bi"
        ],

        "description":
            "Analyze data and create useful reports and insights.",

        "keywords":
            "python sql excel pandas numpy statistics "
            "power bi data analysis visualization"
    },


    {
        "title": "AI / ML Intern",

        "skills": [
            "python",
            "machine learning",
            "ai",
            "pandas",
            "numpy",
            "scikit-learn",
            "nlp"
        ],

        "description":
            "Work with machine learning, NLP and AI applications.",

        "keywords":
            "python machine learning ai pandas numpy "
            "scikit-learn nlp artificial intelligence"
    },


    {
        "title": "Software Developer",

        "skills": [
            "python",
            "java",
            "sql",
            "git",
            "oop",
            "api",
            "testing"
        ],

        "description":
            "Develop, test and maintain software applications.",

        "keywords":
            "python java sql git oop api "
            "software development testing programming"
    }

]


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    return mysql.connector.connect(

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
            ""
        ),

        database=os.getenv(
            "DB_NAME",
            "ai_resume_db"
        ),

        port=int(
            os.getenv(
                "DB_PORT",
                "3306"
            )
        )
    )


# =========================================================
# RESUME TEXT EXTRACTION
# =========================================================

def extract_text(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()


    # TXT FILE
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
                "TXT extraction error:",
                error
            )

            return ""


    # PDF FILE
    if extension == ".pdf":

        try:

            from pypdf import PdfReader

            reader = PdfReader(
                file_path
            )

            text = ""

            for page in reader.pages:

                text += (
                    page.extract_text()
                    or ""
                )

            return text

        except Exception as error:

            print(
                "PDF extraction error:",
                error
            )

            return ""


    return ""


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(resume_text):

    resume_text = normalize_text(
        resume_text
    )

    detected_skills = []


    # Collect all skills

    all_skills = set()

    for job in JOB_PROFILES:

        for skill in job["skills"]:

            all_skills.add(
                skill
            )


    # Check longer skills first

    all_skills = sorted(
        all_skills,
        key=len,
        reverse=True
    )


    # Search skills

    for skill in all_skills:

        pattern = (

            r"(?<![a-z0-9+#.])"

            + re.escape(
                skill.lower()
            )

            + r"(?![a-z0-9+#.])"
        )


        if re.search(
            pattern,
            resume_text
        ):

            detected_skills.append(
                skill
            )


    return detected_skills


# =========================================================
# AI / NLP RECOMMENDATION
# =========================================================

def calculate_recommendations(
    resume_text
):

    resume_text = normalize_text(
        resume_text
    )


    # -----------------------------------------------------
    # STEP 1: SKILL EXTRACTION
    # -----------------------------------------------------

    resume_skills = extract_skills(
        resume_text
    )


    # -----------------------------------------------------
    # STEP 2: PREPARE TEXT DOCUMENTS
    # -----------------------------------------------------

    documents = [

        resume_text

    ] + [

        job["keywords"]

        for job in JOB_PROFILES

    ]


    # -----------------------------------------------------
    # STEP 3: TF-IDF
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2)

    )


    tfidf_matrix = vectorizer.fit_transform(
        documents
    )


    # -----------------------------------------------------
    # STEP 4: COSINE SIMILARITY
    # -----------------------------------------------------

    similarity_scores = cosine_similarity(

        tfidf_matrix[0:1],

        tfidf_matrix[1:]

    ).flatten()


    recommendations = []


    # -----------------------------------------------------
    # STEP 5: CALCULATE SCORE
    # -----------------------------------------------------

    for index, job in enumerate(
        JOB_PROFILES
    ):

        matched_skills = []


        for skill in job["skills"]:

            if skill in resume_skills:

                matched_skills.append(
                    skill
                )


        # Skill match

        skill_score = (

            len(matched_skills)

            /

            len(job["skills"])

        ) * 100


        # NLP similarity

        nlp_score = (

            similarity_scores[index]

            * 100

        )


        # Final score

        final_score = (

            (nlp_score * 0.60)

            +

            (skill_score * 0.40)

        )


        recommendations.append({

            "title":
                job["title"],

            "description":
                job["description"],

            "score":
                round(final_score),

            "nlp_score":
                round(nlp_score),

            "skill_score":
                round(skill_score),

            "matched":
                matched_skills

        })


    # -----------------------------------------------------
    # STEP 6: SORT JOBS
    # -----------------------------------------------------

    recommendations.sort(

        key=lambda item:
            item["score"],

        reverse=True

    )


    return (
        recommendations,
        resume_skills
    )


# =========================================================
# SAVE RESULT TO MYSQL
# =========================================================

def save_result(

    filename,
    skills,
    recommendations

):

    try:

        connection = get_connection()

        cursor = connection.cursor()


        top_job = recommendations[0]


        query = """

        INSERT INTO resume_results

        (
            resume_name,
            extracted_skills,
            recommended_job,
            match_score
        )

        VALUES (%s, %s, %s, %s)

        """


        cursor.execute(

            query,

            (

                filename,

                ", ".join(
                    skills
                ),

                top_job["title"],

                top_job["score"]

            )

        )


        connection.commit()

        cursor.close()

        connection.close()


    except Exception as error:

        print(
            "Database save skipped:",
            error
        )


# =========================================================
# HOME / RESUME SCREENING
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def index():

    recommendations = None

    detected_skills = []

    resume_name = None

    error = None


    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        uploaded_file = request.files.get(
            "resume"
        )


        # -------------------------------------------------
        # CHECK FILE
        # -------------------------------------------------

        if (

            not uploaded_file

            or not uploaded_file.filename

        ):

            error = (
                "Please select a resume."
            )


        else:

            extension = os.path.splitext(

                uploaded_file.filename

            )[1].lower()


            # -------------------------------------------------
            # FILE TYPE
            # -------------------------------------------------

            if extension not in [

                ".pdf",

                ".txt"

            ]:

                error = (

                    "Only PDF and TXT "
                    "resumes are supported."
                )


            else:

                resume_name = (
                    uploaded_file.filename
                )


                # -------------------------------------------------
                # SAFE FILE NAME
                # -------------------------------------------------

                safe_name = re.sub(

                    r"[^A-Za-z0-9._-]",

                    "_",

                    resume_name

                )


                file_path = os.path.join(

                    app.config[
                        "UPLOAD_FOLDER"
                    ],

                    safe_name

                )


                uploaded_file.save(
                    file_path
                )


                # -------------------------------------------------
                # EXTRACT TEXT
                # -------------------------------------------------

                resume_text = extract_text(
                    file_path
                )


                if not resume_text.strip():

                    error = (

                        "Could not read the resume. "
                        "Please upload a text-based "
                        "PDF or TXT file."
                    )


                else:

                    # ---------------------------------------------
                    # AI ANALYSIS
                    # ---------------------------------------------

                    (

                        recommendations,

                        detected_skills

                    ) = calculate_recommendations(

                        resume_text

                    )


                    # ---------------------------------------------
                    # SAVE TO DATABASE
                    # ---------------------------------------------

                    save_result(

                        resume_name,

                        detected_skills,

                        recommendations

                    )


                    # ---------------------------------------------
                    # SAVE ANALYSIS FOR PDF
                    # ---------------------------------------------

                    session["analysis"] = {

                        "resume_name":
                            resume_name,

                        "skills":
                            detected_skills,

                        "results":
                            recommendations

                    }


    # =====================================================
    # RENDER PAGE
    # =====================================================

    return render_template(

        "index.html",

        results=recommendations,

        skills=detected_skills,

        resume_name=resume_name,

        error=error

    )


# =========================================================
# DOWNLOAD PDF REPORT
# =========================================================

@app.get(
    "/download-report"
)

def download_report():

    analysis = session.get(
        "analysis"
    )


    # -----------------------------------------------------
    # NO ANALYSIS
    # -----------------------------------------------------

    if not analysis:

        return (

            "No analysis available. "
            "Please analyze a resume first.",

            404

        )


    # -----------------------------------------------------
    # PDF PATH
    # -----------------------------------------------------

    file_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

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


    # =====================================================
    # TITLE
    # =====================================================

    pdf.setFont(

        "Helvetica-Bold",

        20

    )

    pdf.drawString(

        50,

        y,

        "AI Resume Analysis Report"

    )


    y -= 35


    pdf.setFont(

        "Helvetica",

        10

    )

    pdf.drawString(

        50,

        y,

        "AI-Based Resume Screening and Job Recommendation System"

    )


    y -= 35


    # =====================================================
    # RESUME NAME
    # =====================================================

    pdf.setFont(

        "Helvetica-Bold",

        13

    )

    pdf.drawString(

        50,

        y,

        "Resume"

    )


    y -= 20


    pdf.setFont(

        "Helvetica",

        11

    )

    pdf.drawString(

        50,

        y,

        analysis[
            "resume_name"
        ]

    )


    y -= 35


    # =====================================================
    # DETECTED SKILLS
    # =====================================================

    pdf.setFont(

        "Helvetica-Bold",

        13

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

        analysis[
            "skills"
        ]

    )


    # Prevent very long line

    if not skills_text:

        skills_text = (
            "No skills detected"
        )


    # Simple wrapping

    skill_words = skills_text.split()

    current_line = ""

    for word in skill_words:

        test_line = (

            current_line

            + " "

            + word

        ).strip()


        if len(test_line) > 90:

            pdf.drawString(

                50,

                y,

                current_line

            )

            y -= 15

            current_line = word

        else:

            current_line = test_line


    if current_line:

        pdf.drawString(

            50,

            y,

            current_line

        )

        y -= 20


    y -= 15


    # =====================================================
    # TOP RECOMMENDATION
    # =====================================================

    top_job = analysis[
        "results"
    ][0]


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

        12

    )

    pdf.drawString(

        50,

        y,

        f'{top_job["title"]} - '
        f'{top_job["score"]}%'

    )


    y -= 35


    # =====================================================
    # JOB RECOMMENDATIONS
    # =====================================================

    pdf.setFont(

        "Helvetica-Bold",

        13

    )

    pdf.drawString(

        50,

        y,

        "Job Recommendations"

    )


    y -= 25


    for index, job in enumerate(

        analysis[
            "results"
        ],

        start=1

    ):


        # New page

        if y < 120:

            pdf.showPage()

            y = height - 50


        pdf.setFont(

            "Helvetica-Bold",

            11

        )

        pdf.drawString(

            55,

            y,

            f'{index}. {job["title"]}'

        )


        y -= 18


        pdf.setFont(

            "Helvetica",

            10

        )

        pdf.drawString(

            70,

            y,

            f'Match Score: '
            f'{job["score"]}%'

        )


        y -= 16


        pdf.drawString(

            70,

            y,

            f'NLP Similarity: '
            f'{job["nlp_score"]}%'

        )


        y -= 16


        pdf.drawString(

            70,

            y,

            f'Skill Match: '
            f'{job["skill_score"]}%'

        )


        y -= 16


        matched = ", ".join(

            job["matched"]

        )


        if not matched:

            matched = "None"


        pdf.drawString(

            70,

            y,

            "Matched Skills: "
            + matched

        )


        y -= 28


    # =====================================================
    # SCORE CALCULATION
    # =====================================================

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

        "Final Score = "
        "60% NLP Similarity + "
        "40% Skill Match"

    )


    y -= 35


    # =====================================================
    # DISCLAIMER
    # =====================================================

    pdf.setFont(

        "Helvetica-Oblique",

        9

    )

    pdf.drawString(

        50,

        y,

        "This report is for project-level "
        "recommendation purposes."

    )


    y -= 14


    pdf.drawString(

        50,

        y,

        "It should not be treated as an "
        "automated hiring decision."

    )


    # =====================================================
    # SAVE PDF
    # =====================================================

    pdf.save()


    # =====================================================
    # SEND PDF TO USER
    # =====================================================

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

@app.get(
    "/health"
)

def health():

    return {

        "status": "ok",

        "message":
            "AI Resume Screening API is running"

    }


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(

            os.getenv(

                "PORT",

                "5000"

            )

        )

    )
