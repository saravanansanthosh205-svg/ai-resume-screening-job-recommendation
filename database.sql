CREATE DATABASE IF NOT EXISTS ai_resume_db;
USE ai_resume_db;
CREATE TABLE IF NOT EXISTS resume_results (
 id INT AUTO_INCREMENT PRIMARY KEY,
 resume_name VARCHAR(255) NOT NULL,
 extracted_skills TEXT,
 recommended_job VARCHAR(100),
 match_score INT,
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
