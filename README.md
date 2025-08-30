# SmartAlert - Network Alert Dashboard  

This project is a **graduation team project** that combines **machine learning**, practical **SOC workflows**, and a clean **dashboard** to reduce alert fatigue and speed up triage for entry-level SOC analysts.  

---

## 🎥 Demo Video  

[![Watch the demo on YouTube](https://img.youtube.com/vi/APvQzbvzH0g/0.jpg)](https://youtu.be/APvQzbvzH0g)  

---

## 📌 Overview  

This repository contains a **Flask-based dashboard** that:  
- Reads network flow data (CSV).  
- Runs a pre-trained ML model to classify flows.  
- Sends **Telegram and Email alerts** for detected attacks.  

⚠️ **Important:** This repo does **NOT** include secret credentials or model artifacts. Make sure to set the required environment variables before running.  

---

## 🚀 Features  

- ✅ Resilient CSV reader for network flow data.  
- ✅ Pre-trained model (`saved_model.pkl`) and label encoder (`label_encoder.pkl`) for predictions.  
- ✅ Telegram + Email alerts for non-benign predictions.  
- ✅ Flask-based dashboard (`templates/index.html`) with search, filters, and export.  
- ✅ AI-powered recommendations using Google Generative AI (optional).  

---

## 📂 Required Files  

- `saved_model.pkl` — trained ML model (**not included**)  
- `label_encoder.pkl` — fitted label encoder (**not included**)  
- `traffic.csv` — runtime data (default input file)  

---

## 🔑 Environment Variables  

Create a `.env` file (or export variables in your shell):  

```env
MODEL_PATH=saved_model.pkl
LABEL_ENCODER_PATH=label_encoder.pkl
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
EMAIL_SENDER=you@example.com
EMAIL_PASSWORD=app-password
EMAIL_RECEIVER=recipient@example.com
GOOGLE_API_KEY=your_google_api_key   # optional
```

⚠️ **Never commit `.env` or credentials to GitHub.**  

---

## ⚙️ Setup (Windows PowerShell)  

```powershell
# Create a virtual environment
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install flask pandas requests python-dotenv
# Optional (for AI recommendations)
# pip install google-generativeai

# Run the app
python .\Flask_1.0_public.py
```

Example of setting env vars in PowerShell:  

```powershell
$env:MODEL_PATH = "C:\path\to\saved_model.pkl"
$env:EMAIL_SENDER = "me@example.com"
$env:EMAIL_PASSWORD = "app-password"
python .\Flask_1.0_public.py
```

---

## 🔒 Security & Publishing  

- `.gitignore` is configured to exclude `.env`, `*.pkl`, and `uploads/`.  
- Always use **application-specific passwords** (e.g., Gmail App Password).  
- Rotate any tokens if committed by mistake.  

---

## 🛠 Troubleshooting  

- **Missing model files** → check `MODEL_PATH` & `LABEL_ENCODER_PATH`.  
- **Telegram/Email alerts fail** → verify credentials & internet access.  
- **No data** → ensure `traffic.csv` exists and matches trained model features.  

---

## 🧑‍💻 Training the Model  

To generate `saved_model.pkl` and `label_encoder.pkl`, `model_one.py`.  

### Quick Training  

```powershell
pip install -r requirements.txt
python model_one.py # --input shuffled_data.csv --model-out saved_model.pkl --label-out label_encoder.pkl
```

The trainer will:  
- Drop meta columns (`src_ip`, `dst_ip`, `timespan`).  
- Convert features to numeric, fill missing values with 0.  
- Train a `RandomForestClassifier`.  
- Save the model + encoder.  

---

## 📡 Collecting Live Network Flows (CICFlowMeter)  

You can generate `traffic.csv` with **CICFlowMeter**.  

```powershell
cicflowmeter.exe -i "Wi-Fi" -c traffic.csv --fields "src_ip","dst_ip","timestamp","dst_port","protocol","flow_duration","tot_fwd_pkts","tot_bwd_pkts","flow_byts_s","flow_pkts_s", ...
```

- Run as **Administrator** if needed.  
- Replace `"Wi-Fi"` with your network interface.  
- Ensure the generated CSV matches features expected by the model.  

---

## 👨‍💻 Team  

This project was built by:  
- Ahmad AbuAldahab  
- Ammar Alrafee  
- Mohammad Kraizem  

---

## 📜 License  

Add your preferred license (e.g., MIT, Apache 2.0).  
