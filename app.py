import os, re
from flask import Flask, render_template, request
import mysql.connector

app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=5*1024*1024
app.config["UPLOAD_FOLDER"]="uploads"
os.makedirs("uploads",exist_ok=True)

JOBS=[
{"title":"Python Developer","skills":["python","mysql","flask","html","css","javascript"],"description":"Develop Python web applications and backend services."},
{"title":"Web Developer","skills":["html","css","javascript","bootstrap","react"],"description":"Build responsive websites and web interfaces."},
{"title":"Data Analyst","skills":["python","sql","excel","pandas","numpy","data analysis"],"description":"Analyze data and prepare reports and insights."},
{"title":"Full Stack Developer","skills":["python","html","css","javascript","mysql","react","flask"],"description":"Work on frontend and backend application development."},
{"title":"AI/ML Intern","skills":["python","machine learning","ai","pandas","numpy","scikit-learn"],"description":"Assist with AI and machine learning projects."}
]

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST","localhost"),
        user=os.getenv("DB_USER","root"),
        password=os.getenv("DB_PASSWORD",""),
        database=os.getenv("DB_NAME","ai_resume_db"),
        port=int(os.getenv("DB_PORT","3306"))
    )

def extract_text(path):
    ext=os.path.splitext(path)[1].lower()
    if ext==".txt":
        with open(path,"r",encoding="utf-8",errors="ignore") as f:return f.read()
    if ext==".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except Exception:return ""
    return ""

def clean(text): return re.sub(r"\s+"," ",text.lower()).strip()

def screen(text):
    text=clean(text); out=[]
    for job in JOBS:
        matched=[s for s in job["skills"] if s in text]
        out.append({"title":job["title"],"score":round(len(matched)/len(job["skills"])*100),
                    "matched":matched,"description":job["description"]})
    return sorted(out,key=lambda x:x["score"],reverse=True)

def save_result(filename,skills,job,score):
    try:
        con=get_connection(); cur=con.cursor()
        cur.execute("INSERT INTO resume_results (resume_name,extracted_skills,recommended_job,match_score) VALUES (%s,%s,%s,%s)",
                    (filename,", ".join(skills),job,score))
        con.commit(); cur.close(); con.close()
    except Exception as e: print("Database save skipped:",e)

@app.route("/",methods=["GET","POST"])
def index():
    results=None; name=None; error=None
    if request.method=="POST":
        file=request.files.get("resume")
        if not file or not file.filename: error="Please select a resume file."
        elif os.path.splitext(file.filename)[1].lower() not in {".pdf",".txt"}: error="Only PDF and TXT files are supported."
        else:
            name=file.filename
            safe=re.sub(r"[^A-Za-z0-9._-]","_",name)
            path=os.path.join("uploads",safe); file.save(path)
            text=extract_text(path)
            if not text.strip(): error="Could not extract text. Try a text-based PDF or TXT resume."
            else:
                results=screen(text); lower=clean(text)
                skills=[]
                for job in JOBS:
                    for s in job["skills"]:
                        if s in lower and s not in skills: skills.append(s)
                save_result(name,skills,results[0]["title"],results[0]["score"])
    return render_template("index.html",results=results,resume_name=name,error=error)

@app.get("/health")
def health(): return {"status":"ok"}

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")))
