# SE-GUARD ML Backend
## Python + Machine Learning Fake Detection Engine

---

## 📁 Project Structure

```
se_guard_backend/
├── app.py                    ← Flask REST API server
├── requirements.txt          ← pip dependencies
├── README.md                 ← This file
│
├── models/
│   ├── detection_models.py   ← All 5 ML detectors
│   └── saved/                ← Auto-created .pkl model files
│
├── static/
│   └── api.js                ← Frontend JS bridge (copy to HTML folder)
│
└── tests/
    └── test_models.py        ← Test suite
```

---

## ⚙️ Setup (Windows / Mac / Linux)

### Step 1 — Install Python (if not already installed)
Download Python 3.10+ from https://python.org/downloads
Make sure to check **"Add Python to PATH"** on Windows.

### Step 2 — Install dependencies

```bash
cd se_guard_backend
pip install -r requirements.txt
```

All libraries are standard pip packages — no Conda, no CUDA, no special setup:

| Library        | Purpose                          |
|----------------|----------------------------------|
| flask          | REST API web server              |
| flask-cors     | Allow frontend to call backend   |
| scikit-learn   | RandomForest, SVC, IsolationForest |
| numpy          | Numerical arrays                 |
| pandas         | Synthetic training data          |
| joblib         | Save/load trained models         |
| nltk           | NLP text utilities               |
| bcrypt         | Password hashing (register/login)|
| PyJWT          | JSON Web Token auth              |
| python-dotenv  | Environment variables            |

### Step 3 — Start the server

```bash
python app.py
```

First run trains all 5 ML models (takes ~10 seconds), then saves them.
Subsequent runs load instantly from saved `.pkl` files.

**Expected output:**
```
[SE-GUARD] Training FakeProfileDetector …
[SE-GUARD] FakeProfileDetector trained ✓
[SE-GUARD] Training MessageAbuseDetector …
[SE-GUARD] MessageAbuseDetector trained ✓
... (all 5 models)
[SE-GUARD] ✅ All SE-GUARD ML models loaded and ready.
[SE-GUARD] Server starting on http://127.0.0.1:5000
```

### Step 4 — Connect the frontend

Copy `static/api.js` to your HTML files folder (same folder as `se_guard_auth.html` and `se-guard-dashboard.html`).

The dashboard already references it:
```html
<script src="api.js"></script>
```

---

## 🤖 The 5 ML Detectors

### 1. Fake Profile Detector
**Algorithm:** RandomForest (150 trees)
**Detects:** Bot accounts, fake registrations, sock puppets

Input features:
```json
{
  "account_age_days"    : 2,
  "posts"               : 0,
  "completeness"        : 0.1,
  "email_domain_score"  : 0.2,
  "phone_verified"      : 0,
  "photo_uploaded"      : 0,
  "reviews_count"       : 0,
  "avg_rating"          : 5.0,
  "login_frequency"     : 0.02,
  "ip_country_mismatch" : 1
}
```

Output:
```json
{
  "verdict"          : "FAKE",
  "fake_probability" : 94.2,
  "risk_level"       : "HIGH",
  "flags"            : ["Very new account", "No profile photo", "IP/country mismatch"],
  "recommendation"   : "BLOCK"
}
```

---

### 2. Message Abuse Detector
**Algorithm:** TF-IDF (trigrams) + Logistic Regression + Hard rules
**Detects:** Abusive language, threats, payment pressure

Input:
```json
{ "text": "pay me right now or I will destroy you" }
```

Output:
```json
{
  "verdict"          : "THREAT",
  "flagged"          : true,
  "type"             : "THREAT",
  "confidence"       : 97.3,
  "matched_keywords" : ["destroy"],
  "action"           : "BLOCK_MESSAGE",
  "ml_scores"        : {"CLEAN":2.1, "ABUSIVE":3.4, "THREAT":91.2, "PRESSURE":3.3}
}
```

---

### 3. Fake Review Detector
**Algorithm:** TF-IDF + LinearSVC (calibrated) + Heuristic rules
**Detects:** Bot reviews, paid reviews, review bombing

Input:
```json
{ "text": "BEST PRODUCT EVER!!!! AMAZING WONDERFUL PERFECT BUY NOW", "rating": 5 }
```

Output:
```json
{
  "verdict"          : "FAKE",
  "fake_probability" : 91.5,
  "risk_level"       : "HIGH",
  "flags"            : ["Excessive superlatives (5)", "Excessive exclamation marks (4)"],
  "action"           : "HIDE_REVIEW"
}
```

---

### 4. Suspicious Payment Detector
**Algorithm:** IsolationForest (unsupervised anomaly detection)
**Detects:** Fraud, money laundering, unusual transactions

Input:
```json
{
  "amount"                 : 150000,
  "hour_of_day"            : 2,
  "retries"                : 6,
  "new_device"             : 1,
  "vpn_flag"               : 1,
  "amount_vs_history_ratio": 12.0,
  "time_since_last_txn_min": 0.5
}
```

Output:
```json
{
  "verdict"          : "SUSPICIOUS",
  "risk_probability" : 93.7,
  "risk_level"       : "HIGH",
  "anomaly_score"    : -0.4821,
  "flags"            : ["Unusually high amount", "Multiple payment retries", "VPN/Proxy detected"],
  "action"           : "BLOCK_PAYMENT"
}
```

---

### 5. Fake Product Listing Detector
**Algorithm:** RandomForest (150 trees)
**Detects:** Scam listings, counterfeit products, misleading prices

Input:
```json
{
  "price_vs_category_avg_ratio" : 0.03,
  "description_length"          : 5,
  "image_count"                 : 0,
  "seller_age_days"             : 1,
  "seller_rating"               : 5.0,
  "seller_total_sales"          : 0,
  "discount_pct"                : 97,
  "has_contact_info_in_desc"    : 1
}
```

Output:
```json
{
  "verdict"          : "FAKE",
  "fake_probability" : 96.1,
  "risk_level"       : "HIGH",
  "flags"            : ["Price suspiciously low vs category", "No product images", "Unrealistic discount"],
  "action"           : "REMOVE_LISTING"
}
```

---

## 🌐 All API Endpoints

| Method | Endpoint              | Description                     |
|--------|-----------------------|---------------------------------|
| GET    | /api/health           | Server + model health check     |
| GET    | /api/stats            | Detection stats and audit log   |
| GET    | /api/detect/demo      | Example payloads for all models |
| POST   | /api/auth/register    | Create account (bcrypt hashed)  |
| POST   | /api/auth/login       | Login → JWT token               |
| POST   | /api/detect/profile   | Fake profile detection          |
| POST   | /api/detect/message   | Message abuse detection         |
| POST   | /api/detect/review    | Fake review detection           |
| POST   | /api/detect/payment   | Suspicious payment detection    |
| POST   | /api/detect/product   | Fake product listing detection  |
| POST   | /api/detect/batch     | All detectors in one call       |

---

## 🧪 Run Tests

```bash
python tests/test_models.py
```

Expected:
```
============================================================
  1. FAKE PROFILE DETECTOR
============================================================
  ✅ PASS  Obvious fake profile → FAKE
  ✅ PASS  Obvious fake → HIGH risk
  ...
  ALL TESTS PASSED ✅
```

---

## 🔌 Frontend Integration (api.js)

The `api.js` file is automatically loaded by the dashboard.
Here's how each dashboard action connects to the ML backend:

### Message Sending
```javascript
// In the dashboard's sendMsg() function, add:
const result = await SGUARD.onSendMessage(msgText, showNotification);
if (result.flagged) {
  // message is already shown by onSendMessage
  return; // optionally block sending
}
```

### Product Submission
```javascript
// In handleAddProduct(), add after collecting product data:
const check = await SGUARD.onProductSubmit(productObj);
if (check.risk_level === "HIGH") {
  showNotification("⚠️ Product flagged: " + check.flags.join(", "), "warning");
}
```

### Payment Initiation
```javascript
// Before opening payment modal:
const check = await SGUARD.onPaymentInitiate(amount, { retries: 0 });
if (check.verdict === "SUSPICIOUS") {
  showNotification("🚨 Suspicious payment blocked!", "error");
  return;
}
```

### Review/Testimonial Submit
```javascript
const check = await SGUARD.onReviewSubmit(reviewText, rating);
if (check.verdict === "FAKE") {
  showNotification("⚠️ Review flagged as potentially fake", "warning");
}
```

---

## 🔒 Auth Flow

```
1. User fills Register form  →  POST /api/auth/register
   ← { token, role }

2. Store token:  localStorage.setItem("sg_token", token)

3. User logs in  →  POST /api/auth/login
   ← { token, role, name }

4. All detect/* calls automatically include token in header:
   Authorization: Bearer <token>
```

---

## 💡 Tips

- **Backend offline?** The frontend `api.js` has rule-based fallbacks for message detection — the app still works without the backend.
- **Model retraining:** Delete `models/saved/` and restart the server to retrain all models from scratch.
- **Production:** Replace the in-memory user store with SQLite or PostgreSQL. Use `gunicorn app:app` instead of `python app.py`.

---

## 🚀 Quick Start (copy-paste)

```bash
# 1. Install
pip install flask flask-cors scikit-learn numpy pandas joblib nltk bcrypt PyJWT python-dotenv

# 2. Start backend
cd se_guard_backend
python app.py

# 3. Copy api.js to your HTML folder
cp static/api.js /path/to/your/html/folder/

# 4. Test
python tests/test_models.py

# 5. Open browser: http://127.0.0.1:5000/api/health
```
