"""
se_guard_backend/app.py
========================
SE-GUARD — Flask REST API Server

Endpoints:
  POST /api/detect/profile    — Fake profile detection
  POST /api/detect/message    — Message abuse / threat detection
  POST /api/detect/review     — Fake review detection
  POST /api/detect/payment    — Suspicious payment detection
  POST /api/detect/product    — Fake product listing detection
  POST /api/detect/batch      — Run all detectors at once

  POST /api/auth/register     — Register user (bcrypt hashed)
  POST /api/auth/login        — Login → JWT token

  GET  /api/health            — Health check
  GET  /api/stats             — Detection statistics
"""

import os, time, datetime, logging, json
from functools import wraps
from collections import defaultdict

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import bcrypt, jwt
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

from models.detection_models import load_all_models, get_models

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[SE-GUARD] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # allow frontend on any port

SECRET_KEY = os.getenv("SE_GUARD_SECRET", "se-guard-super-secret-dev-key-2025")

load_dotenv()

# ─── MongoDB Connection ───────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://burmansachin08_db_user:kd3xppSCgbDI5wsL@seguard.qwlckrx.mongodb.net/?retryWrites=true&w=majority&appName=seguard")
DB_NAME = os.getenv("MONGO_DB_NAME", os.getenv("DB_NAME", "seguard"))

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    log.info(f"[MongoDB] Connected to {MONGO_URI} - Database: {DB_NAME}")
    MONGO_ENABLED = True
except Exception as e:
    log.warning(f"[MongoDB] Connection failed: {e}. Using in-memory storage.")
    mongo_client = None
    db = None
    MONGO_ENABLED = False

# ─── Collection References (Section-wise) ─────────────────────────────────
# Users collection (all user profiles)
users_col = db["users"] if MONGO_ENABLED else None

# Section-wise data collections
business_data_col = db["business_data"] if MONGO_ENABLED else None
client_data_col = db["client_data"] if MONGO_ENABLED else None
freelancer_data_col = db["freelancer_data"] if MONGO_ENABLED else None

# Shared collections
products_col = db["products"] if MONGO_ENABLED else None
orders_col = db["orders"] if MONGO_ENABLED else None
payments_col = db["payments"] if MONGO_ENABLED else None
messages_col = db["messages"] if MONGO_ENABLED else None
reviews_col = db["reviews"] if MONGO_ENABLED else None
reports_col = db["reports"] if MONGO_ENABLED else None

# Detection logs collection
detection_logs_col = db["detection_logs"] if MONGO_ENABLED else None

# ─── In-memory stores (fallback when MongoDB is unavailable) ──────────────
USERS     = {}   # email → {hash, role, created_at}
LOG_STORE = []   # detection audit log (last 500)
STATS     = defaultdict(lambda: defaultdict(int))  # stats[endpoint][verdict]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ok(data: dict, status: int = 200):
    data["status"] = "ok"
    data["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify(data), status

def _err(msg: str, status: int = 400):
    return jsonify({"status": "error", "message": msg,
                    "timestamp": datetime.datetime.utcnow().isoformat()+"Z"}), status

def _log_detection(endpoint: str, result: dict, raw_input: dict):
    entry = {
        "endpoint"  : endpoint,
        "verdict"   : result.get("verdict", "UNKNOWN"),
        "risk"      : result.get("risk_level", "-"),
        "action"    : result.get("action", "-"),
        "input_keys": list(raw_input.keys()),
        "ts"        : time.time(),
        "timestamp" : datetime.datetime.utcnow()
    }
    
    # Store in MongoDB if available
    if MONGO_ENABLED and detection_logs_col is not None:
        try:
            detection_logs_col.insert_one(entry)
        except Exception as e:
            log.warning(f"[MongoDB] Failed to log detection: {e}")
    
    # Also keep in-memory for quick stats
    LOG_STORE.append(entry)
    if len(LOG_STORE) > 500:
        LOG_STORE.pop(0)
    STATS[endpoint][result.get("verdict","?")] += 1


# ─── MongoDB Helper Functions ─────────────────────────────────────────────
def save_user(email: str, user_data: dict):
    """Save or update user in MongoDB."""
    if MONGO_ENABLED and users_col is not None:
        try:
            users_col.update_one(
                {"email": email},
                {"$set": user_data},
                upsert=True
            )
        except Exception as e:
            log.warning(f"[MongoDB] Failed to save user: {e}")


def get_user(email: str) -> dict:
    """Get user from MongoDB or in-memory."""
    if MONGO_ENABLED and users_col is not None:
        try:
            user = users_col.find_one({"email": email})
            if user:
                user.pop("_id", None)
                return user
        except Exception as e:
            log.warning(f"[MongoDB] Failed to get user: {e}")
    return USERS.get(email)


def save_section_data(section: str, data: dict):
    """Save data to section-specific collection.
    Sections: business, client, freelancer
    """
    if not MONGO_ENABLED:
        return
    
    collection_map = {
        "business": business_data_col,
        "client": client_data_col,
        "freelancer": freelancer_data_col
    }
    
    collection = collection_map.get(section)
    if collection is not None:
        try:
            data["timestamp"] = datetime.datetime.utcnow()
            result = collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            log.warning(f"[MongoDB] Failed to save {section} data: {e}")
    return None


def get_section_data(section: str, query: dict = None, limit: int = 100):
    """Get data from section-specific collection."""
    if not MONGO_ENABLED:
        return []
    
    collection_map = {
        "business": business_data_col,
        "client": client_data_col,
        "freelancer": freelancer_data_col
    }
    
    collection = collection_map.get(section)
    if collection is not None:
        try:
            cursor = collection.find(query or {}).limit(limit).sort("timestamp", -1)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except Exception as e:
            log.warning(f"[MongoDB] Failed to get {section} data: {e}")
    return []


def save_to_collection(collection_name: str, data: dict):
    """Save data to a specific collection."""
    if not MONGO_ENABLED or db is None:
        return None
    
    try:
        collection = db[collection_name]
        data["timestamp"] = datetime.datetime.utcnow()
        result = collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        log.warning(f"[MongoDB] Failed to save to {collection_name}: {e}")
    return None


def get_from_collection(collection_name: str, query: dict = None, limit: int = 100):
    """Get data from a specific collection."""
    if not MONGO_ENABLED or db is None:
        return []
    
    try:
        collection = db[collection_name]
        cursor = collection.find(query or {}).limit(limit).sort("timestamp", -1)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results
    except Exception as e:
        log.warning(f"[MongoDB] Failed to get from {collection_name}: {e}")
    return []

def _require_json(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not request.is_json:
            return _err("Content-Type must be application/json")
        return f(*a, **kw)
    return wrapper

def _jwt_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return _err("Missing token", 401)
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return _err("Token expired", 401)
        except jwt.InvalidTokenError:
            return _err("Invalid token", 401)
        return f(*a, **kw)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
@_require_json
def register():
    data  = request.get_json()
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")
    role  = data.get("role", "client")
    fname = data.get("firstName", "")
    lname = data.get("lastName", "")

    if not email or "@" not in email:
        return _err("Invalid email address")
    if len(pwd) < 8:
        return _err("Password must be at least 8 characters")
    
    # Check if user exists in MongoDB or in-memory
    existing_user = get_user(email)
    if existing_user:
        return _err("Email already registered", 409)

    hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
    user_data = {
        "hash"      : hashed,
        "role"      : role,
        "firstName" : fname,
        "lastName"  : lname,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "email"     : email
    }
    
    # Save to both MongoDB and in-memory
    USERS[email] = user_data
    save_user(email, user_data)

    # Auto-run profile detection on new registrant
    models = get_models()
    profile_result = models["profile"].predict({
        "account_age_days": 0,
        "completeness"    : 0.2,
        "phone_verified"  : 0,
        "photo_uploaded"  : 0,
        "reviews_count"   : 0,
    })

    token = jwt.encode({
        "sub" : email,
        "role": role,
        "exp" : datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }, SECRET_KEY, algorithm="HS256")

    log.info(f"Registered: {email} ({role})")
    return _ok({
        "message"       : "Account created successfully",
        "token"         : token,
        "role"          : role,
        "initial_trust" : profile_result
    }, 201)


@app.route("/api/auth/login", methods=["POST"])
@_require_json
def login():
    data  = request.get_json()
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")

    user = get_user(email)
    if not user or not bcrypt.checkpw(pwd.encode(), user["hash"].encode()):
        return _err("Invalid email or password", 401)

    token = jwt.encode({
        "sub" : email,
        "role": user["role"],
        "exp" : datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }, SECRET_KEY, algorithm="HS256")

    log.info(f"Login: {email}")
    return _ok({
        "message": "Login successful",
        "token"  : token,
        "role"   : user["role"],
        "name"   : f"{user['firstName']} {user['lastName']}".strip() or email
    })


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/detect/profile", methods=["POST"])
@_require_json
def detect_profile():
    """
    Detect whether a user profile is fake.

    Body (all optional):
    {
      "account_age_days"    : 3,
      "posts"               : 0,
      "completeness"        : 0.1,
      "email_domain_score"  : 0.3,
      "phone_verified"      : 0,
      "photo_uploaded"      : 0,
      "reviews_count"       : 0,
      "avg_rating"          : 5.0,
      "login_frequency"     : 0.05,
      "ip_country_mismatch" : 1
    }
    """
    data   = request.get_json()
    result = get_models()["profile"].predict(data)
    _log_detection("profile", result, data)
    return _ok({"detection": result, "model": "RandomForest", "version": "1.0"})


@app.route("/api/detect/message", methods=["POST"])
@_require_json
def detect_message():
    """
    Detect abusive / threatening / pressure messages.

    Body:
    { "text": "pay me right now or else" }
    """
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return _err("Field 'text' is required")
    result = get_models()["message"].predict(text)
    _log_detection("message", result, data)
    return _ok({"detection": result, "model": "TF-IDF + LogisticRegression", "version": "1.0"})


@app.route("/api/detect/review", methods=["POST"])
@_require_json
def detect_review():
    """
    Detect fake reviews / testimonials.

    Body:
    {
      "text"  : "absolutely amazing best product ever!!",
      "rating": 5
    }
    """
    data   = request.get_json()
    text   = data.get("text", "").strip()
    rating = int(data.get("rating", 5))
    if not text:
        return _err("Field 'text' is required")
    result = get_models()["review"].predict(text, rating)
    _log_detection("review", result, data)
    return _ok({"detection": result, "model": "TF-IDF + LinearSVC", "version": "1.0"})


@app.route("/api/detect/payment", methods=["POST"])
@_require_json
def detect_payment():
    """
    Detect suspicious payment transactions.

    Body:
    {
      "amount"                    : 150000,
      "hour_of_day"               : 2,
      "retries"                   : 5,
      "new_device"                : 1,
      "vpn_flag"                  : 1,
      "amount_vs_history_ratio"   : 8.0,
      "time_since_last_txn_min"   : 1
    }
    """
    data   = request.get_json()
    result = get_models()["payment"].predict(data)
    _log_detection("payment", result, data)
    return _ok({"detection": result, "model": "IsolationForest", "version": "1.0"})


@app.route("/api/detect/product", methods=["POST"])
@_require_json
def detect_product():
    """
    Detect fake product listings.

    Body:
    {
      "price_vs_category_avg_ratio" : 0.05,
      "description_length"          : 5,
      "image_count"                 : 0,
      "seller_age_days"             : 2,
      "seller_rating"               : 5.0,
      "seller_total_sales"          : 0,
      "discount_pct"                : 95,
      "has_contact_info_in_desc"    : 1
    }
    """
    data   = request.get_json()
    result = get_models()["product"].predict(data)
    _log_detection("product", result, data)
    return _ok({"detection": result, "model": "RandomForest", "version": "1.0"})


@app.route("/api/detect/batch", methods=["POST"])
@_require_json
def detect_batch():
    """
    Run multiple detectors in one call.

    Body:
    {
      "profile" : { ... },
      "message" : { "text": "..." },
      "review"  : { "text": "...", "rating": 5 },
      "payment" : { ... },
      "product" : { ... }
    }
    Only keys present in the body are processed.
    """
    data    = request.get_json()
    models  = get_models()
    results = {}

    if "profile" in data:
        r = models["profile"].predict(data["profile"])
        results["profile"] = r
        _log_detection("profile", r, data["profile"])

    if "message" in data:
        txt = data["message"].get("text", "")
        r = models["message"].predict(txt)
        results["message"] = r
        _log_detection("message", r, data["message"])

    if "review" in data:
        txt    = data["review"].get("text", "")
        rating = int(data["review"].get("rating", 5))
        r = models["review"].predict(txt, rating)
        results["review"] = r
        _log_detection("review", r, data["review"])

    if "payment" in data:
        r = models["payment"].predict(data["payment"])
        results["payment"] = r
        _log_detection("payment", r, data["payment"])

    if "product" in data:
        r = models["product"].predict(data["product"])
        results["product"] = r
        _log_detection("product", r, data["product"])

    # Overall risk summary
    risk_levels = [v.get("risk_level", "LOW") for v in results.values()]
    overall = "HIGH" if "HIGH" in risk_levels else "MEDIUM" if "MEDIUM" in risk_levels else "LOW"

    return _ok({"results": results, "overall_risk": overall})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION-WISE DATA STORAGE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/data/<section>", methods=["POST"])
@_require_json
def save_section_data_endpoint(section):
    """
    Save data to section-specific collection.
    Sections: business, client, freelancer
    
    Body: Any JSON data to be stored
    """
    if section not in ["business", "client", "freelancer"]:
        return _err("Invalid section. Use: business, client, or freelancer", 400)
    
    data = request.get_json()
    data["section"] = section
    
    # Add user info if available in token
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            data["user_email"] = payload.get("sub")
            data["user_role"] = payload.get("role")
        except:
            pass
    
    inserted_id = save_section_data(section, data)
    
    if inserted_id:
        return _ok({
            "message": f"Data saved to {section} collection",
            "id": inserted_id
        }, 201)
    else:
        # Fallback: store in memory if MongoDB is unavailable
        if section not in ["business", "client", "freelancer"]:
            return _err("Invalid section", 400)
        return _ok({
            "message": f"Data received for {section} (MongoDB unavailable)",
            "data": data
        }, 201)


@app.route("/api/data/<section>", methods=["GET"])
def get_section_data_endpoint(section):
    """
    Get data from section-specific collection.
    Sections: business, client, freelancer
    
    Query params: limit (default 100)
    """
    if section not in ["business", "client", "freelancer"]:
        return _err("Invalid section. Use: business, client, or freelancer", 400)
    
    limit = request.args.get("limit", 100, type=int)
    
    # Build query based on user role if token provided
    query = {}
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_role = payload.get("role")
            user_email = payload.get("sub")
            # Users can only see their own data unless they're admin
            if user_role != "admin":
                query["user_email"] = user_email
        except:
            pass
    
    results = get_section_data(section, query, limit)
    
    return _ok({
        "section": section,
        "count": len(results),
        "data": results
    })


@app.route("/api/collection/<collection_name>", methods=["POST"])
@_require_json
def save_collection_data(collection_name):
    """
    Save data to a specific collection.
    Collections: products, orders, payments, messages, reviews, reports
    
    Body: Any JSON data to be stored
    """
    valid_collections = ["products", "orders", "payments", "messages", "reviews", "reports", "inquiries", "checkout_sessions"]
    if collection_name not in valid_collections:
        return _err(f"Invalid collection. Use: {', '.join(valid_collections)}", 400)
    
    data = request.get_json()
    
    # Add user info if available
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            data["user_email"] = payload.get("sub")
            data["user_role"] = payload.get("role")
        except:
            pass
    
    inserted_id = save_to_collection(collection_name, data)
    
    if inserted_id:
        return _ok({
            "message": f"Data saved to {collection_name}",
            "id": inserted_id
        }, 201)
    else:
        return _ok({
            "message": f"Data received for {collection_name} (MongoDB unavailable)",
            "data": data
        }, 201)


@app.route("/api/collection/<collection_name>", methods=["GET"])
def get_collection_data(collection_name):
    """
    Get data from a specific collection.
    Collections: products, orders, payments, messages, reviews, reports
    
    Query params: limit (default 100)
    """
    valid_collections = ["products", "orders", "payments", "messages", "reviews", "reports", "inquiries", "checkout_sessions"]
    if collection_name not in valid_collections:
        return _err(f"Invalid collection. Use: {', '.join(valid_collections)}", 400)
    
    limit = request.args.get("limit", 100, type=int)
    
    # Build query based on user
    query = {}
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_role = payload.get("role")
            user_email = payload.get("sub")
            if user_role != "admin":
                query["user_email"] = user_email
        except:
            pass
    
    results = get_from_collection(collection_name, query, limit)
    
    return _ok({
        "collection": collection_name,
        "count": len(results),
        "data": results
    })


@app.route("/api/db/status", methods=["GET"])
def db_status():
    """Check MongoDB connection status."""
    return _ok({
        "mongodb_enabled": MONGO_ENABLED,
        "mongodb_uri": MONGO_URI if MONGO_ENABLED else None,
        "database_name": DB_NAME if MONGO_ENABLED else None,
        "collections": {
            "users": users_col is not None,
            "business_data": business_data_col is not None,
            "client_data": client_data_col is not None,
            "freelancer_data": freelancer_data_col is not None,
            "products": products_col is not None,
            "orders": orders_col is not None,
            "payments": payments_col is not None,
            "messages": messages_col is not None,
            "reviews": reviews_col is not None,
            "reports": reports_col is not None,
            "detection_logs": detection_logs_col is not None
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    models  = get_models()
    return _ok({
        "service"     : "SE-GUARD ML Backend",
        "version"     : "1.0.0",
        "models_ready": list(models.keys()),
        "uptime_note" : "All models loaded and serving"
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return _ok({
        "total_detections": len(LOG_STORE),
        "by_endpoint"     : {k: dict(v) for k, v in STATS.items()},
        "recent_log"      : LOG_STORE[-20:]
    })


@app.route("/", methods=["GET"])
def index():
    """Serve the auth/login HTML file as the entry point."""
    return send_from_directory(".", "se_guard_auth.html")


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    """Serve the main dashboard HTML file."""
    return send_from_directory(".", "se-guard-dashboard.html")


@app.route("/se-guard-dashboard.html", methods=["GET"])
def dashboard_html():
    """Serve the dashboard HTML file (for direct file access)."""
    return send_from_directory(".", "se-guard-dashboard.html")


@app.route("/auth", methods=["GET"])
def auth_page():
    """Serve the auth/login HTML file."""
    return send_from_directory(".", "se_guard_auth.html")


@app.route("/api/detect/demo", methods=["GET"])
def demo():
    """Return example payloads for all detectors."""
    return _ok({
        "examples": {
            "profile": {
                "account_age_days": 2, "posts": 0, "completeness": 0.1,
                "email_domain_score": 0.2, "phone_verified": 0,
                "photo_uploaded": 0, "reviews_count": 0,
                "avg_rating": 5.0, "login_frequency": 0.02, "ip_country_mismatch": 1
            },
            "message": {"text": "pay me right now or i will destroy you"},
            "review" : {"text": "best product ever amazing wonderful perfect", "rating": 5},
            "payment": {
                "amount": 175000, "hour_of_day": 2, "retries": 6,
                "new_device": 1, "vpn_flag": 1,
                "amount_vs_history_ratio": 12.0, "time_since_last_txn_min": 0.5
            },
            "product": {
                "price_vs_category_avg_ratio": 0.04, "description_length": 4,
                "image_count": 0, "seller_age_days": 1, "seller_rating": 5.0,
                "seller_total_sales": 0, "discount_pct": 97, "has_contact_info_in_desc": 1
            }
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  SE-GUARD ML Backend  |  Loading models …")
    log.info("=" * 60)
    load_all_models()
    log.info("=" * 60)
    log.info("  Server starting on  http://127.0.0.1:5000")
    log.info("  API docs at         http://127.0.0.1:5000/api/detect/demo")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
