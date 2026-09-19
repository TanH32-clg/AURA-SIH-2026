# WelfareWatch Frontend Prototype

A Bootstrap-based frontend + small Flask backend for the AI-Based Predictive Personnel Stress & Welfare Monitoring System.

## Features

- Professional responsive landing page
- Personnel login
- Demo account registration
- Personnel dashboard
- Voluntary wellness self-assessment
- 8-question assessment with progress indicator
- Assessment responses saved to CSV
- User records saved to CSV
- Passwords stored as hashes
- Responsive Bootstrap UI
- Privacy-first messaging and human-in-the-loop positioning

## Run locally

### 1. Create environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
python app.py
```

Open:
http://127.0.0.1:5000

## CSV storage

The application automatically creates:

- `data/users.csv`
- `data/wellness_responses.csv`

For a real deployment, replace CSV storage with a secure database and use environment variables for secrets.

## Important prototype note

This is a demonstration/prototype. It is not a medical diagnostic system. Do not use prototype risk outputs for disciplinary, employment, medical, or operational decisions without proper validation, governance, consent, security controls and domain approval.
