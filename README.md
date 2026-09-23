# AI Resume Screening and Job Recommendation System

## Local
pip install -r requirements.txt
Run database.sql in MySQL.
Set the variables from .env.example.
python app.py
Open http://127.0.0.1:5000

## Public deployment
Push to GitHub and create a Render Web Service.

Build command:
pip install -r requirements.txt

Start command:
gunicorn app:app

Add DB_HOST, DB_PORT, DB_USER, DB_PASSWORD and DB_NAME as Render environment variables.
Use a cloud MySQL-compatible database; do not use localhost for the public database.

Do not upload .env or real passwords to GitHub.
