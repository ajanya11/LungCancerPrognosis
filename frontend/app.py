from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import sqlite3
import os
import io
import uuid
import datetime
import numpy as np
import sys
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =====================================================
# APP INIT
# =====================================================
app = Flask(__name__)
app.secret_key = "supersecretkey"

# =====================================================
# PATHS
# =====================================================
# app.py already lives inside the "frontend" folder, so BASE_DIR IS the
# frontend directory. Do NOT append "frontend" again below — that was
# creating a phantom frontend/frontend/static/... folder that Flask never
# serves, which is why uploaded images / Grad-CAM heatmaps looked broken.
BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
STATIC_ROOT    = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
DB_NAME = os.path.join(BASE_DIR, "users.db")

prediction_store = {}

# =====================================================
# PROJECT ROOT ON PATH
# =====================================================
# fusion/ and utils/ live one level above the frontend/ folder.
BASE_PROJECT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if BASE_PROJECT not in sys.path:
    sys.path.append(BASE_PROJECT)

# =====================================================
# LAZY PIPELINE LOAD
# =====================================================
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        from fusion.fusion_pipeline import FinalFusionPipeline
        pipeline = FinalFusionPipeline()
    return pipeline

# =====================================================
# DATABASE
# =====================================================
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =====================================================
# Convert absolute disk path → web URL
# =====================================================
def to_web_url(abs_path: str) -> Optional[str]:
    """
    Given an absolute filesystem path anywhere under
    frontend/static/, return the corresponding /static/... URL.

    Examples
    --------
    .../frontend/static/uploads/foo.png  →  /static/uploads/foo.png
    .../frontend/static/gradcam/bar.jpg  →  /static/gradcam/bar.jpg
    """
    if not abs_path:
        return None

    # Normalise separators
    p = os.path.normpath(abs_path)
    static = os.path.normpath(STATIC_ROOT)

    # Check if the file lives under our static root
    if p.startswith(static):
        rel = os.path.relpath(p, static)          # e.g. "uploads/foo.png"
        url = "/static/" + rel.replace(os.sep, "/")
        print(f"[to_web_url] {abs_path!r}  →  {url!r}")
        return url

    # Fallback: if path contains "static/" anywhere, slice from there
    normalised = abs_path.replace("\\", "/")
    if "static/" in normalised:
        idx = normalised.index("static/")
        url = "/" + normalised[idx:]
        print(f"[to_web_url] fallback  {abs_path!r}  →  {url!r}")
        return url

    print(f"[to_web_url] MISS  {abs_path!r}")
    return None


# =====================================================
# Convert web URL → absolute disk path  (for PDF)
# =====================================================
def _url_to_abs(web_url: str) -> Optional[str]:
    """
    /static/uploads/foo.png  →  .../frontend/static/uploads/foo.png
    """
    if not web_url:
        return None

    # Strip leading slash, replace forward-slashes with OS separator
    rel = web_url.lstrip("/").replace("/", os.sep)

    candidate = os.path.join(BASE_DIR, rel)
    if os.path.isfile(candidate):
        return candidate

    print(f"[_url_to_abs] MISS  url={web_url!r}")
    return None


# =====================================================
# PDF REPORT GENERATOR
# =====================================================
def generate_pdf_report(data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER
    from reportlab.graphics.shapes import Drawing, Rect

    PAGE_W, PAGE_H = A4
    CONTENT_W = PAGE_W - 4 * cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=2 * cm,  rightMargin=2 * cm,
    )

    C_BLUE        = colors.HexColor("#1d4ed8")
    C_RED         = colors.HexColor("#dc2626")
    C_GREEN       = colors.HexColor("#16a34a")
    C_ORANGE      = colors.HexColor("#ea580c")
    C_MUTED       = colors.HexColor("#64748b")
    C_BG          = colors.HexColor("#f1f5f9")
    C_DIV         = colors.HexColor("#e2e8f0")
    C_DARK        = colors.HexColor("#1e293b")
    C_INFO_BG     = colors.HexColor("#f0f9ff")
    C_INFO_BORDER = colors.HexColor("#bae6fd")
    C_PANEL_DARK  = colors.HexColor("#0f172a")

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s = ps("T",   fontSize=20, fontName="Helvetica-Bold",
                        textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ps("S",   fontSize=8.5, fontName="Helvetica",
                        textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=2)
    sec_s   = ps("H",   fontSize=11, fontName="Helvetica-Bold",
                        textColor=C_BLUE, spaceBefore=14, spaceAfter=6)
    body_s  = ps("B",   fontSize=9.5, fontName="Helvetica",
                        textColor=C_DARK, leading=15)
    sm_s    = ps("SM",  fontSize=8,   fontName="Helvetica",
                        textColor=C_MUTED, leading=12)
    disc_s  = ps("D",   fontSize=7.5, fontName="Helvetica-Oblique",
                        textColor=C_MUTED, leading=11, alignment=TA_CENTER)
    tag_s   = ps("TG",  fontSize=6.5, fontName="Helvetica-Bold",
                        textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=3)
    pill_s  = ps("PIL", fontSize=7,   fontName="Helvetica-Bold",
                        textColor=colors.white, alignment=TA_CENTER)
    note_s  = ps("NT",  fontSize=8,   fontName="Helvetica",
                        textColor=C_MUTED, leading=12)

    risk_raw = (data.get("final_risk") or "unknown").lower()
    if "high" in risk_raw:
        risk_color, risk_label = C_RED,    "HIGH RISK"
    elif "mod" in risk_raw:
        risk_color, risk_label = C_ORANGE, "MODERATE RISK"
    else:
        risk_color, risk_label = C_GREEN,  "LOW RISK"

    cancer_type = data.get("cancer_type", "N/A") or "N/A"
    confidence  = data.get("confidence", 0)

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=C_DIV, spaceAfter=6, spaceBefore=6)

    def kv_table(rows, col_widths=None):
        col_widths = col_widths or [5.5 * cm, CONTENT_W - 5.5 * cm]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (0, -1), C_BG),
            ("TEXTCOLOR",      (0, 0), (0, -1), colors.HexColor("#374151")),
            ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID",           (0, 0), (-1, -1), 0.4, C_DIV),
            ("PADDING",        (0, 0), (-1, -1), 6),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def safe_img(abs_path, size_cm):
        target_px = size_cm * cm
        if abs_path:
            print(f"[safe_img] loading  path={abs_path!r}  exists={os.path.isfile(abs_path)}")
        else:
            print("[safe_img] abs_path is None — rendering placeholder")

        if abs_path and os.path.isfile(abs_path):
            try:
                img = RLImage(abs_path, width=target_px, height=target_px,
                              kind="proportional")
                print(f"[safe_img] OK  {abs_path!r}")
                return img
            except Exception as exc:
                print(f"[safe_img] RLImage FAILED  path={abs_path!r}  err={exc}")

        placeholder = Table(
            [[Paragraph("Image\nunavailable", sm_s)]],
            colWidths=[target_px],
            rowHeights=[target_px],
        )
        placeholder.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL_DARK),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ]))
        return placeholder

    def colour_scale_legend():
        bar_w = CONTENT_W - 3.2 * cm
        bar_h = 0.28 * cm
        steps = 120
        jet = [
            colors.HexColor("#00008b"), colors.HexColor("#0000ff"),
            colors.HexColor("#00bfff"), colors.HexColor("#00ff00"),
            colors.HexColor("#ffff00"), colors.HexColor("#ff8000"),
            colors.HexColor("#ff0000"),
        ]
        d  = Drawing(bar_w, bar_h)
        sw = bar_w / steps
        for i in range(steps):
            t   = i / (steps - 1)
            seg = t * (len(jet) - 1)
            lo  = int(seg)
            hi  = min(lo + 1, len(jet) - 1)
            c   = colors.linearlyInterpolatedColor(jet[lo], jet[hi], 0, 1, seg - lo)
            d.add(Rect(i * sw, 0, sw + 0.5, bar_h, fillColor=c, strokeColor=None))

        tbl = Table(
            [[Paragraph("Low", sm_s), d, Paragraph("High activation", sm_s)]],
            colWidths=[1.3 * cm, bar_w, 2.2 * cm],
        )
        tbl.setStyle(TableStyle([
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",      (0, 0), (0,  0),  "LEFT"),
            ("ALIGN",      (2, 0), (2,  0),  "RIGHT"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX",        (0, 0), (-1, -1), 0.4, C_DIV),
            ("PADDING",    (0, 0), (-1, -1), 5),
        ]))
        return tbl

    # PDF uses stored absolute paths directly
    orig_path   = data.get("original_abs_path")   or data.get("raw_image_path")
    binary_path = data.get("gradcam_binary_abs_path")
    type_path   = data.get("gradcam_type_abs_path")

    # Fallback chain
    if not (type_path   and os.path.isfile(type_path)):   type_path   = binary_path
    if not (type_path   and os.path.isfile(type_path)):   type_path   = orig_path
    if not (binary_path and os.path.isfile(binary_path)): binary_path = orig_path
    if not (orig_path   and os.path.isfile(orig_path)):   orig_path   = binary_path

    print(f"[PDF] orig={orig_path!r}")
    print(f"[PDF] binary={binary_path!r}")
    print(f"[PDF] type={type_path!r}")

    story = []

    story.append(Paragraph("Lung Cancer AI Prognosis Report", title_s))
    story.append(Paragraph("Multimodal Deep Learning Risk Assessment", sub_s))
    story.append(Paragraph(
        f"Generated: {data.get('timestamp', datetime.datetime.now().strftime('%d %B %Y, %H:%M'))}"
        f"  |  Report ID: {data.get('prediction_id', 'N/A')}"
        f"  |  Patient: {data.get('username', 'N/A')}",
        sub_s,
    ))
    story.append(Spacer(1, 6))
    story.append(hr())

    banner_style = ps("BNR", fontSize=14, fontName="Helvetica-Bold",
                      textColor=colors.white, alignment=TA_CENTER)
    banner = Table([[Paragraph(risk_label, banner_style)]], colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("PADDING",    (0, 0), (-1, -1), 12),
    ]))
    story.append(banner)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Prediction Summary", sec_s))
    story.append(kv_table([
        ["Cancer Type",      cancer_type],
        ["Survival Chance",  f"{data.get('survival', 0)}%"],
        ["Model Confidence", f"{confidence}%"],
        ["Risk Level",       data.get("final_risk", "N/A")],
    ]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Clinical Parameters", sec_s))
    clinical = data.get("clinical_data") or {}
    story.append(kv_table([
        ["Age",            str(clinical.get("age",     "N/A"))],
        ["Cancer Stage",   str(clinical.get("stage",   "N/A"))],
        ["Smoking Status", str(clinical.get("smoking", "N/A"))],
    ]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Grad-CAM Activation Maps", sec_s))

    IMG_CM  = 4.8
    label_w = IMG_CM * cm
    GAP_W   = 0.2 * cm
    extra_w = max(CONTENT_W - 3 * label_w - 2 * GAP_W, 0.1 * cm)
    col_w   = [label_w, label_w, label_w, extra_w]

    row_tags = [
        Paragraph("ORIGINAL CT",      tag_s),
        Paragraph("CANCER DETECTION", tag_s),
        Paragraph("SUBTYPE",          tag_s),
        "",
    ]
    row_imgs = [
        safe_img(orig_path,   IMG_CM),
        safe_img(binary_path, IMG_CM),
        safe_img(type_path,   IMG_CM),
        "",
    ]
    row_pills = [
        Paragraph("Input",          pill_s),
        Paragraph(f"{confidence}%", pill_s),
        Paragraph(f"{confidence}%", pill_s),
        "",
    ]

    panel_tbl = Table([row_tags, row_imgs, row_pills], colWidths=col_w)
    panel_tbl.setStyle(TableStyle([
        ("ALIGN",      (0, 0), (2, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",    (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 1), (0, 1), C_PANEL_DARK),
        ("BACKGROUND", (1, 1), (1, 1), C_PANEL_DARK),
        ("BACKGROUND", (2, 1), (2, 1), C_PANEL_DARK),
        ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#94a3b8")),
        ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#dc2626")),
        ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#3b82f6")),
        ("BOX",        (0, 0), (0, 2), 1.0, C_DIV),
        ("BOX",        (1, 0), (1, 2), 1.0, C_DIV),
        ("BOX",        (2, 0), (2, 2), 1.0, C_DIV),
    ]))

    note_tbl = Table(
        [[Paragraph(
            "Red / orange = high model attention.  "
            "Blue / green = low attention.  "
            "Warmer regions are the primary areas influencing the prediction.",
            note_s,
        )]],
        colWidths=[CONTENT_W],
    )
    note_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_INFO_BG),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_INFO_BORDER),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))

    story.append(KeepTogether([
        panel_tbl,
        Spacer(1, 8),
        colour_scale_legend(),
        Spacer(1, 6),
        note_tbl,
    ]))

    story.append(Spacer(1, 6))

    suggestions = data.get("suggestions") or []
    if suggestions:
        story.append(Paragraph("Clinical Suggestions", sec_s))
        for i, s in enumerate(suggestions, 1):
            story.append(Paragraph(f"  {i}. {s}", body_s))
        story.append(Spacer(1, 6))

    story.append(hr())
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI research system and is intended for "
        "informational purposes only. It does not constitute a medical diagnosis or clinical "
        "advice. Always consult a qualified healthcare professional for medical decisions.",
        disc_s,
    ))

    doc.build(story)
    return buf.getvalue()


# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
            conn.commit()
            conn.close()
            flash("Signup successful")
            return redirect(url_for('login'))
        except Exception:
            flash("Username already exists")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session['user_id']  = user["id"]
            session['username'] = user["username"]
            return redirect(url_for('home'))
        flash("Invalid login")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', user=session['username'])

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/accuracy')
def accuracy():
    return render_template("accuracy.html")

@app.route('/data')
def data():
    return render_template("data.html")

@app.route('/more_info')
def more_info():
    return render_template("More info.html")

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    username     = session.get("username")
    user_records = [v for v in prediction_store.values()
                    if v.get("username") == username]
    user_records = sorted(user_records,
                          key=lambda x: x.get("timestamp", ""),
                          reverse=True)

    high_count = sum(1 for r in user_records if 'high' in (r.get('final_risk') or '').lower())
    mod_count  = sum(1 for r in user_records if 'mod'  in (r.get('final_risk') or '').lower())
    low_count  = sum(1 for r in user_records if 'low'  in (r.get('final_risk') or '').lower())

    return render_template(
        "history.html",
        records=user_records,
        username=username,
        high_count=high_count,
        mod_count=mod_count,
        low_count=low_count,
    )

@app.route('/delete_prediction/<prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if prediction_id in prediction_store:
        del prediction_store[prediction_id]
        flash("Prediction deleted")
    return redirect(url_for('history'))


# =====================================================
# PREDICTION
# =====================================================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template("prediction.html")

    image_path = None

    try:
        user_name = session.get("username", "guest")

        # ── Upload CT image ──────────────────────────────────────────────────
        file = request.files.get("ct_image")
        if not file or not file.filename:
            raise ValueError("No CT image uploaded")
        filename   = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)   # ← ABSOLUTE path
        file.save(image_path)
        print(f"[upload] Image saved: {image_path}")

        # ── Clinical inputs ──────────────────────────────────────────────────
        age     = int(request.form.get("age", 50))
        smoking = request.form.get("smoking", "No")
        stage   = request.form.get("stage",   "II")

        smoking_str   = "Yes" if str(smoking).lower() in ("current", "former", "yes", "1") else "No"
        stage_map     = {"1": "I", "2": "II", "3": "III", "4": "IV"}
        stage_str     = stage_map.get(str(stage), str(stage).upper())
        clinical_data = {"age": age, "stage": stage_str, "smoking": smoking_str}

        # ── Optional genomic data ────────────────────────────────────────────
        genomic_data = None
        genomic_file = request.files.get("genomic_data")
        if genomic_file and genomic_file.filename:
            gname = secure_filename(genomic_file.filename)
            gpath = os.path.join(UPLOAD_FOLDER, gname)
            genomic_file.save(gpath)
            try:
                genomic_data = np.loadtxt(gpath, delimiter=",")
            except Exception as ge:
                print("Genomic load error:", ge)

        # ── CT inference ─────────────────────────────────────────────────────
        from utils.ct_inference import run_ct_inference
        ct_result = run_ct_inference(image_path=image_path, username=user_name)
        print("CT result keys:", list(ct_result.keys()))

        if "error" in ct_result:
            raise RuntimeError(ct_result["error"])

        # ── Fusion pipeline ──────────────────────────────────────────────────
        fusion = get_pipeline().predict(
            ct_result=ct_result,
            clinical_data=clinical_data,
            genomic_data=genomic_data,
        )
        print("Fusion:", {k: v for k, v in fusion.items() if k != "breakdown"})

        final_risk  = fusion.get("risk_level",      "Unknown")
        cancer_type = fusion.get("cancer_type",      "N/A")
        survival    = fusion.get("survival_chance",  0)
        confidence  = fusion.get("model_confidence", 0)
        suggestions = fusion.get("suggestions",      [])

        # ── All paths from ct_result are now ABSOLUTE ────────────────────────
        # ct_inference.py returns absolute paths for original_image,
        # gradcam, gradcam_binary, gradcam_type.
        # Fusion pipeline may also return absolute paths via those same keys.
        # We pick from fusion first, fall back to ct_result.

        orig_abs        = fusion.get("original_image")    or ct_result.get("original_image")    or image_path
        gradcam_abs     = fusion.get("gradcam")            or ct_result.get("gradcam")
        binary_abs      = fusion.get("gradcam_binary")     or ct_result.get("gradcam_binary")
        type_abs        = fusion.get("gradcam_type")       or ct_result.get("gradcam_type")

        # Ensure we always have at least the uploaded image as fallback
        if not (orig_abs and os.path.isfile(orig_abs)):
            orig_abs = image_path

        # ── Build web URLs from absolute paths ───────────────────────────────
        original_url       = to_web_url(orig_abs)
        gradcam_url        = to_web_url(gradcam_abs)
        gradcam_binary_url = to_web_url(binary_abs)
        gradcam_type_url   = to_web_url(type_abs)

        # URL fallback chain — always show *something*
        if not original_url:
            original_url = to_web_url(image_path)
        if not gradcam_binary_url:
            gradcam_binary_url = gradcam_url or original_url
        if not gradcam_type_url:
            gradcam_type_url = gradcam_url or original_url
        if not gradcam_url:
            gradcam_url = gradcam_binary_url or original_url

        print("[URLs]", original_url, gradcam_binary_url, gradcam_type_url, gradcam_url)

        # ── Absolute-path fallback chain (for PDF) ───────────────────────────
        binary_abs_resolved = binary_abs or gradcam_abs or image_path
        type_abs_resolved   = type_abs   or gradcam_abs or image_path

        # ── Store prediction ─────────────────────────────────────────────────
        prediction_id = str(uuid.uuid4())[:8].upper()
        prediction_store[prediction_id] = {
            "prediction_id": prediction_id,
            "username":      user_name,
            "timestamp":     datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
            "final_risk":    final_risk,
            "cancer_type":   cancer_type,
            "survival":      survival,
            "confidence":    confidence,
            "suggestions":   suggestions,
            "clinical_data": clinical_data,

            # Web URLs (for HTML templates)
            "original_url":       original_url,
            "gradcam_url":        gradcam_url,
            "gradcam_binary_url": gradcam_binary_url,
            "gradcam_type_url":   gradcam_type_url,

            # Absolute paths (for PDF generator)
            "original_abs_path":       orig_abs,
            "gradcam_abs_path":        gradcam_abs,
            "gradcam_binary_abs_path": binary_abs_resolved,
            "gradcam_type_abs_path":   type_abs_resolved,
            "raw_image_path":          image_path,
        }

        return render_template(
            "prediction.html",
            final_risk=final_risk,
            cancer_type=cancer_type,
            survival=survival,
            confidence=confidence,
            gradcam_url=gradcam_url,
            gradcam_binary_url=gradcam_binary_url,
            gradcam_type_url=gradcam_type_url,
            original_url=original_url,
            suggestions=suggestions,
            username=user_name,
            prediction_id=prediction_id,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        fallback_url = to_web_url(image_path) if image_path else None
        return render_template(
            "prediction.html",
            final_risk="Error",
            cancer_type="N/A",
            survival=0,
            confidence=0,
            gradcam_url=fallback_url,
            gradcam_binary_url=fallback_url,
            gradcam_type_url=fallback_url,
            original_url=fallback_url,
            suggestions=[f"System error: {str(e)}"],
            username=session.get("username", "guest"),
            prediction_id=None,
        )


# =====================================================
# DOWNLOAD REPORT
# =====================================================
@app.route('/download_report/<prediction_id>')
def download_report(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    data = prediction_store.get(prediction_id)
    if not data:
        flash("Report not found. Please run a new prediction.")
        return redirect(url_for('prediction'))

    try:
        pdf_bytes = generate_pdf_report(data)
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"LungAI_Report_{prediction_id}.pdf",
        )
    except ImportError as ie:
        flash(f"Missing library: {ie}. Run: pip install reportlab Pillow")
        return redirect(url_for('prediction'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Could not generate report: {e}")
        return redirect(url_for('prediction'))


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    print("Server running...")
    # use_reloader=False: prevents the dev-server watchdog from restarting
    # mid-request when an uploaded CT image / gradcam file is written into
    # static/ (which it watches) — that restart is what caused
    # ERR_CONNECTION_RESET on /prediction.
    # threaded=True: lets the server handle more than one request at a time,
    # useful while the CT inference/model call is running.
    app.run(debug=True, use_reloader=False, threaded=True)
