from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pathlib import Path
import csv
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
USERS_CSV = DATA_DIR / "users.csv"
WELLNESS_CSV = DATA_DIR / "wellness_responses.csv"

QUESTIONS = [
    {
        "id": "sleep",
        "category": "Sleep & Rest",
        "question": "How would you rate your sleep quality during the last 7 days?",
        "options": ["Very good", "Good", "Average", "Poor", "Very poor"]
    },
    {
        "id": "fatigue",
        "category": "Energy & Fatigue",
        "question": "How often have you felt physically or mentally exhausted after duty?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": "stress",
        "category": "Stress",
        "question": "How would you describe your overall stress level recently?",
        "options": ["Very low", "Low", "Moderate", "High", "Very high"]
    },
    {
        "id": "mood",
        "category": "Mood & Emotions",
        "question": "How would you describe your mood in general over the last 2 weeks?",
        "options": ["Very positive", "Positive", "Neutral", "Low", "Very low"]
    },
    {
        "id": "work_balance",
        "category": "Work & Family Balance",
        "question": "How satisfied are you with your current work and family balance?",
        "options": ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"]
    },
    {
        "id": "recovery",
        "category": "Recovery",
        "question": "How often do you get enough time to recover between duties?",
        "options": ["Always", "Often", "Sometimes", "Rarely", "Never"]
    },
    {
        "id": "motivation",
        "category": "Motivation",
        "question": "How would you describe your motivation and energy for daily activities?",
        "options": ["Very high", "High", "Moderate", "Low", "Very low"]
    },
    {
        "id": "support",
        "category": "Social Support",
        "question": "Do you feel that you have someone you can approach for support when needed?",
        "options": ["Definitely", "Mostly", "Not sure", "Rarely", "No"]
    }
]

def ensure_csv_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_CSV.exists():
        with USERS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["created_at", "personnel_id", "name", "email", "password_hash"])
    if not WELLNESS_CSV.exists():
        headers = ["submitted_at", "personnel_id"] + [q["id"] for q in QUESTIONS]
        with WELLNESS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def find_user(personnel_id):
    ensure_csv_files()
    with USERS_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["personnel_id"].lower() == personnel_id.lower():
                return row
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        personnel_id = request.form.get("personnel_id", "").strip()
        password = request.form.get("password", "")
        user = find_user(personnel_id)

        if user and check_password_hash(user["password_hash"], password):
            session["personnel_id"] = user["personnel_id"]
            session["name"] = user["name"]
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid Personnel ID or password.")
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    ensure_csv_files()
    personnel_id = request.form.get("personnel_id", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not all([personnel_id, name, email, password]):
        return render_template("login.html", error="Please complete all registration fields.")

    if find_user(personnel_id):
        return render_template("login.html", error="Personnel ID already exists.")

    with USERS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            personnel_id, name, email, generate_password_hash(password)
        ])

    session["personnel_id"] = personnel_id
    session["name"] = name
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    if "personnel_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session.get("name", "Personnel"))

@app.route("/assessment")
def assessment():
    if "personnel_id" not in session:
        return redirect(url_for("login"))
    return render_template("assessment.html", questions=QUESTIONS, name=session.get("name", "Personnel"))

@app.route("/submit-assessment", methods=["POST"])
def submit_assessment():
    if "personnel_id" not in session:
        return jsonify({"success": False, "message": "Please log in."}), 401

    answers = {}
    for q in QUESTIONS:
        value = request.form.get(q["id"])
        if not value:
            return jsonify({"success": False, "message": f"Please answer: {q['question']}"}), 400
        answers[q["id"]] = value

    ensure_csv_files()
    with WELLNESS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.now().isoformat(timespec="seconds"), session["personnel_id"]]
            + [answers[q["id"]] for q in QUESTIONS]
        )

    return jsonify({"success": True, "message": "Assessment submitted securely."})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    ensure_csv_files()
    app.run(debug=True)
