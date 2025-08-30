from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import smtplib
from email.message import EmailMessage
import requests
import hashlib
import csv, io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Optional: load environment variables from a local .env file (for development only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== تحميل النموذج والمحول (اقرأ من المتغيرات البيئية بدلاً من حفظ القيم هنا) =====
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_model.pkl")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "label_encoder.pkl")

def _load_pickle_or_raise(path, name):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found at {path}. Set {name.upper()}_PATH or place the file next to the app.")
    with open(path, "rb") as f:
        return pickle.load(f)

model = _load_pickle_or_raise(MODEL_PATH, "model")
label_encoder = _load_pickle_or_raise(LABEL_ENCODER_PATH, "label_encoder")

# ===== إعداد تيليجرام/إيميل (اقرأ القيم من المتغيرات البيئية) =====
# ضع هذه المتغيرات في بيئتك أو ملف .env — لا تقم بارتكابها للمستودع
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ===== حالات تشغيل =====
sent_attack_hashes = set()
email_sent = False  # لمنع تكرار إرسال الإيميل في نفس التشغيل

# ==========================
#     NULL-safe helpers
# ==========================
def is_na(val):
    try:
        return pd.isna(val)
    except Exception:
        return val is None

def safe_val(val, default=""):
    if hasattr(val, "values"):
        try:
            val = val.values[0]
        except Exception:
            pass
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    if isinstance(val, pd.Series) and not val.empty:
        val = val.iloc[0]
    if is_na(val):
        return default
    return val

def get_first_val(val):
    return safe_val(val, default="")

# ==========================
#   CSV Safe Reader (FIX)
# ==========================
def _detect_expected_cols(path: str, default: int = 71) -> int:
    """
    يحاول قراءة الهيدر لتحديد عدد الأعمدة المتوقع. إن فشل يرجع default.
    """
    try:
        with open(path, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header:
                return len(header)
    except Exception:
        pass
    return default

def read_csv_safely(path: str) -> pd.DataFrame:
    """
    1) محاولة read_csv عادية.
    2) إن فشلت بـ ParserError: نستخدم engine='python', on_bad_lines='skip'.
    3) إن فشلت أيضًا: نفلتر يدويًا الأسطر التي تطابق عدد الأعمدة المتوقع (من الهيدر).
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        pass

    # خطة B: بايثون إنجن + تخطي الأسطر السيئة
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except pd.errors.ParserError:
        pass

    # خطة C: فلترة يدوية حسب طول الهيدر
    exp_cols = _detect_expected_cols(path)
    rows = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header and len(header) == exp_cols:
                rows.append(header)
            for rec in r:
                if len(rec) == exp_cols:
                    rows.append(rec)
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for rec in rows:
        w.writerow(rec)
    buf.seek(0)
    try:
        return pd.read_csv(buf)
    except Exception:
        return pd.DataFrame()

# ==========================
#     تنبيه تيليجرام
# ==========================
def send_telegram_alert(row):
    message = (
        f"🚨 اكتشاف هجوم!\n"
        f"🔸 Src IP: {safe_val(row.get('src_ip'), 'N/A')}\n"
        f"🔸 Dst IP: {safe_val(row.get('dst_ip'), 'N/A')}\n"
        f"🔸 Prediction: {safe_val(row.get('Prediction'), 'Unknown')}\n"
    )
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=data, timeout=10)
        print("✅ تم إرسال تنبيه عبر تيليجرام.")
    except Exception as e:
        print("❌ فشل إرسال تيليجرام:", str(e))

# ==========================
#     إرسال بريد إلكتروني
# ==========================
def send_attack_email(attacks_df: pd.DataFrame):
    msg = EmailMessage()
    msg["Subject"] = "🚨 تنبيه: تم اكتشاف هجوم"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    columns_to_include = [
        'src_ip', 'dst_ip', 'Protocol', 'timespan', 'Dst Port',
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'Flow Byts/s', 'Flow Pkts/s', 'Prediction'
    ]
    column_headers = [
        "Source IP",
        "Destination IP",
        "Protocol",
        "Timestamp",
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Bwd Packets",
        "Bytes/sec",
        "Packets/sec",
        "AI Prediction"
    ]

    columns_to_include = [c for c in columns_to_include if c in attacks_df.columns]
    filtered_headers = [h for c, h in zip(columns_to_include, column_headers)]

    df_filtered = attacks_df[columns_to_include].copy()
    df_filtered = df_filtered.fillna("")  # NULL-safe

    html = "<html><body>"
    html += "<h3>🚨 تنبيهات الهجوم المكتشفة:</h3>"
    html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; font-family: Arial;'>"
    html += "<tr style='background-color: #f2f2f2;'>"
    for header in filtered_headers:
        html += f"<th>{header}</th>"
    html += "</tr>"

    for _, r in df_filtered.iterrows():
        html += "<tr>"
        for col in columns_to_include:
            val = safe_val(r[col], default="")
            html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</table><br><p>📡 تم الاكتشاف بواسطة نظام المراقبة الذكي.</p></body></html>"

    msg.set_content("تم اكتشاف هجوم. الرجاء فتح البريد بصيغة HTML.")
    msg.add_alternative(html, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ تم إرسال الإيميل.")
    except Exception as e:
        print("❌ فشل إرسال الإيميل:", str(e))

# ==========================
#   زر AI Recommendations
# ==========================
@app.route("/analyze_ai", methods=["POST"])
def analyze_ai():
    import google.generativeai as genai
    data = request.get_json()
    row = data.get('row')
    if isinstance(row, dict):
        row = {k: ("" if is_na(v) else v) for k, v in row.items()}

    prompt = (
        "You are a cybersecurity SOC analyst assistant.\n"
        "Given the following network alert row, identify the most likely attack type, "
        "then provide clear, actionable recommendations that the SOC team should follow to contain, mitigate, or resolve this threat.\n"
        "Focus ONLY on practical SOC response steps (such as blocking IP, isolating assets, updating firewall rules, etc). Avoid lengthy explanations or definitions.\n"
        "Row data:\n"
        "src_ip, dst_ip, timespan, Dst Port, Protocol, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, "
        "Flow Byts/s, Flow Pkts/s, Prediction\n"
        f"{row}\n"
        "Summarize your recommendations in 2-3 concise bullet points."
    )

    GOOGLE_API_KEY = "AIzaSyAE0O9f5gsRJrwwR5laAeMn8_t7gP5Vrjg"
    genai.configure(api_key=GOOGLE_API_KEY)
    gmodel = genai.GenerativeModel('gemma-3n-e2b-it')
    try:
        response = gmodel.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"ERROR !!!! Gemini: {str(e)}"

    return jsonify({"answer": answer})

# ==========================
#        الصفحة الرئيسية
# ==========================
@app.route("/", methods=["GET"])
def index():
    global email_sent
    results = None
    prediction_counts = {}
    total = 0
    file_datetime = None

    filepath = "traffic.csv"
    if os.path.exists(filepath):
        file_timestamp = os.path.getmtime(filepath)
        file_datetime = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # ↓↓↓ القراءة الآمنة بدل read_csv العادية
        df = read_csv_safely(filepath)

        # لو فاضي نرجّع صفحة بدون بيانات
        if df.empty:
            return render_template("index.html", results=[], prediction_counts={}, total=0, file_timestamp=file_datetime)

        # الميتاداتا
        meta_cols = ['src_ip', 'dst_ip', 'timespan']
        meta_data = df[meta_cols] if all(col in df.columns for col in meta_cols) else None

        # استبعاد الأعمدة غير التدريبية
        columns_to_exclude = meta_cols + ['Label']
        df_pred = df.drop(columns=[c for c in columns_to_exclude if c in df.columns], errors='ignore')

        # ترتيب مطابق لتدريب الموديل
        trained_features = model.feature_names_in_
        df_pred = df_pred.reindex(columns=trained_features)

        # تنظيف رقمي للموديل
        df_pred = df_pred.replace([float('inf'), float('-inf')], pd.NA)
        for col in df_pred.columns:
            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
        df_pred = df_pred.fillna(0)

        # تنبؤ
        predictions = model.predict(df_pred)
        predicted_labels = label_encoder.inverse_transform(predictions)
        df["Prediction"] = predicted_labels

        # إعادة إرفاق الميتاداتا
        if meta_data is not None:
            df = pd.concat([meta_data.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

        # تنبيهات NULL-safe
        if "Prediction" in df.columns:
            df_for_alerts = df.copy().fillna("")
            attack_rows = df_for_alerts[df_for_alerts["Prediction"].astype(str).str.lower() != "benign"]
            if not attack_rows.empty:
                if not email_sent:
                    send_attack_email(attack_rows)
                    email_sent = True
                for _, r in attack_rows.iterrows():
                    fingerprint = f"{safe_val(r.get('src_ip'))}-{safe_val(r.get('dst_ip'))}-{safe_val(r.get('Prediction'))}"
                    hash_id = hashlib.md5(fingerprint.encode()).hexdigest()
                    if hash_id not in sent_attack_hashes:
                        sent_attack_hashes.add(hash_id)
                        send_telegram_alert(r)

        # بروتوكول → اسم (NULL-safe)
        def protocol_number_to_name(v):
            if is_na(v):
                return ""
            try:
                ival = int(v)
            except Exception:
                return str(v)
            return "TCP" if ival == 6 else ("UDP" if ival == 17 else f"Other ({ival})")

        if "Protocol" in df.columns:
            df["Protocol"] = df["Protocol"].apply(protocol_number_to_name)

        # الأعمدة المعروضة
        columns_to_display = [
            'src_ip', 'dst_ip', 'Protocol', 'Dst Port', 'timespan',
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'Flow Byts/s', 'Flow Pkts/s', 'Prediction'
        ]
        df = df[[c for c in columns_to_display if c in df.columns]]

        # NULL-safe للواجهة
        df = df.fillna("")

        results = df.to_dict(orient="records")
        prediction_counts = df.get('Prediction', pd.Series(dtype=str)).fillna("").value_counts().to_dict()
        total = len(df)
    else:
        results = "⚠️ الملف غير موجود في المسار المحدد."
        file_datetime = None

    return render_template("index.html", results=results, prediction_counts=prediction_counts, total=total, file_timestamp=file_datetime)

# ==========================
#   API loaders (مرنة)
# ==========================
def load_dashboard_data(only_attacks=False, source_file="traffic.csv"):
    global email_sent, sent_attack_hashes
    filepath = source_file
    if not os.path.exists(filepath):
        return [], {}

    df = read_csv_safely(filepath)
    if df.empty:
        return [], {}

    meta_cols = ['src_ip', 'dst_ip', 'timespan']
    exclude = meta_cols + ['Label']
    df_pred = df.drop(columns=[c for c in exclude if c in df], errors='ignore')

    df_pred = df_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    df_pred = (
        df_pred.replace([float('inf'), float('-inf')], pd.NA)
               .apply(pd.to_numeric, errors='coerce')
               .fillna(0)
    )

    df["Prediction"] = label_encoder.inverse_transform(model.predict(df_pred))

    if all(col in df.columns for col in meta_cols):
        df = pd.concat([df[meta_cols].reset_index(drop=True),
                        df.reset_index(drop=True)], axis=1)

    def proto_name(v):
        if is_na(v):
            return ""
        try:
            v = int(v)
        except Exception:
            return str(v)
        return {6: "TCP", 17: "UDP"}.get(v, f"Other ({v})")

    if "Protocol" in df:
        df["Protocol"] = df["Protocol"].map(proto_name)

    if only_attacks and "Prediction" in df:
        df = df[df["Prediction"].astype(str).str.lower() != "benign"]

    # Check if data source is realtime before sending alerts
    if "Prediction" in df.columns and source_file == "traffic.csv":
        df_for_alerts = df.copy().fillna("")
        attack_rows = df_for_alerts[df_for_alerts["Prediction"].astype(str).str.lower() != "benign"]
        if not attack_rows.empty:
            if not email_sent:
                send_attack_email(attack_rows)
                email_sent = True
            for _, r in attack_rows.iterrows():
                fingerprint = f"{safe_val(r.get('src_ip'))}-{safe_val(r.get('dst_ip'))}-{safe_val(r.get('Prediction'))}"
                hash_id = hashlib.md5(fingerprint.encode()).hexdigest()
                if hash_id not in sent_attack_hashes:
                    sent_attack_hashes.add(hash_id)
                    send_telegram_alert(r)


    display_cols = [
        'src_ip', 'dst_ip', 'Protocol', 'Dst Port', 'timespan',
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'Flow Byts/s', 'Flow Pkts/s', 'Prediction'
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    df = df.fillna("")

    df = df.rename(columns={
        'src_ip': 'Source IP',
        'dst_ip': 'Destination IP',
        'timespan': 'TIMESPAN',
        'Dst Port': 'Destination Port',
        'Flow Duration': 'Flow Duration (μs)',
        'Tot Fwd Pkts': 'Total Fwd Pkts',
        'Tot Bwd Pkts': 'Total Bwd Pkts',
        'Flow Byts/s': 'Bytes/sec',
        'Flow Pkts/s': 'Packets/sec'
    })

    results = df.to_dict(orient="records")
    counts = df.get("Prediction", pd.Series(dtype=str)).fillna("").value_counts().to_dict()
    return results, counts

@app.route("/realtime_data", methods=["GET"])
def realtime_data():
    results, counts = load_dashboard_data(only_attacks=False, source_file="traffic.csv")
    return jsonify({"results": results, "prediction_counts": counts})

@app.route("/attack_log_data", methods=["GET"])
def attack_log_data():
    results, counts = load_dashboard_data(only_attacks=False, source_file="test2.csv")
    return jsonify({"results": results, "prediction_counts": counts})

if __name__ == "__main__":
    app.run(debug=True)
