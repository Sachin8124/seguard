"""
se_guard_backend/tests/test_models.py
=======================================
Run with:  python tests/test_models.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.detection_models import (
    FakeProfileDetector, MessageAbuseDetector,
    FakeReviewDetector, SuspiciousPaymentDetector, FakeProductDetector
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
RESULTS = []

def check(name, condition, got):
    status = PASS if condition else FAIL
    RESULTS.append((status, name, got))
    print(f"  {status}  {name}")
    if not condition:
        print(f"         Got: {got}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
section("1. FAKE PROFILE DETECTOR")
pd = FakeProfileDetector()

r1 = pd.predict({
    "account_age_days":1,"posts":0,"completeness":0.05,
    "email_domain_score":0.1,"phone_verified":0,"photo_uploaded":0,
    "reviews_count":0,"avg_rating":5.0,"login_frequency":0.01,"ip_country_mismatch":1
})
check("Obvious fake profile → FAKE",        r1["verdict"]=="FAKE",        r1["verdict"])
check("Obvious fake → HIGH risk",           r1["risk_level"]=="HIGH",     r1["risk_level"])
check("Obvious fake → BLOCK recommendation",r1["recommendation"]=="BLOCK",r1["recommendation"])

r2 = pd.predict({
    "account_age_days":500,"posts":80,"completeness":0.95,
    "email_domain_score":0.9,"phone_verified":1,"photo_uploaded":1,
    "reviews_count":30,"avg_rating":4.5,"login_frequency":0.8,"ip_country_mismatch":0
})
check("Real profile → REAL",   r2["verdict"]=="REAL", r2["verdict"])
check("Real profile → LOW risk",r2["risk_level"]=="LOW", r2["risk_level"])


# ─────────────────────────────────────────────────────────────────────────────
section("2. MESSAGE ABUSE DETECTOR")
md = MessageAbuseDetector()

r3 = md.predict("pay me right now or i will destroy everything")
check("Threat message → flagged",          r3["flagged"]==True,           r3["flagged"])
check("Threat message → BLOCK_MESSAGE",    r3["action"]=="BLOCK_MESSAGE", r3["action"])

r4 = md.predict("send money immediately or else you will regret")
check("Pressure message → flagged",        r4["flagged"]==True,           r4["flagged"])

r5 = md.predict("you are a fucking idiot stop ignoring me")
check("Abusive message → flagged",         r5["flagged"]==True,           r5["flagged"])
check("Abusive message → type ABUSIVE",    r5["type"]=="ABUSIVE",         r5["type"])

r6 = md.predict("Hi, please review the attached invoice and let me know your thoughts.")
check("Clean message → not flagged",       r6["flagged"]==False,          r6["flagged"])
check("Clean message → ALLOW",             r6["action"]=="ALLOW",         r6["action"])


# ─────────────────────────────────────────────────────────────────────────────
section("3. FAKE REVIEW DETECTOR")
rd = FakeReviewDetector()

r7 = rd.predict("best amazing perfect wonderful fantastic excellent product ever!!!", 5)
check("Fake review → FAKE",               r7["verdict"]=="FAKE",         r7["verdict"])
check("Fake review → has flags",          len(r7["flags"])>0,            r7["flags"])

r8 = rd.predict("Decent product but packaging needs improvement. Arrived a day late.", 3)
check("Real review → REAL",               r8["verdict"]=="REAL",         r8["verdict"])


# ─────────────────────────────────────────────────────────────────────────────
section("4. SUSPICIOUS PAYMENT DETECTOR")
spd = SuspiciousPaymentDetector()

r9 = spd.predict({
    "amount":180000,"hour_of_day":2,"retries":7,
    "new_device":1,"vpn_flag":1,"amount_vs_history_ratio":15.0,
    "time_since_last_txn_min":0.2
})
check("Suspicious payment → SUSPICIOUS",    r9["verdict"]=="SUSPICIOUS",  r9["verdict"])
check("Suspicious payment → HIGH risk",     r9["risk_level"]=="HIGH",     r9["risk_level"])
check("Suspicious payment → has flags",     len(r9["flags"])>0,           r9["flags"])

r10 = spd.predict({
    "amount":1200,"hour_of_day":14,"retries":0,
    "new_device":0,"vpn_flag":0,"amount_vs_history_ratio":0.9,
    "time_since_last_txn_min":600
})
check("Normal payment → NORMAL",            r10["verdict"]=="NORMAL",     r10["verdict"])


# ─────────────────────────────────────────────────────────────────────────────
section("5. FAKE PRODUCT DETECTOR")
fpd = FakeProductDetector()

r11 = fpd.predict({
    "price_vs_category_avg_ratio":0.03,"description_length":4,
    "image_count":0,"seller_age_days":1,"seller_rating":5.0,
    "seller_total_sales":0,"discount_pct":97,"has_contact_info_in_desc":1
})
check("Fake listing → FAKE",               r11["verdict"]=="FAKE",        r11["verdict"])
check("Fake listing → HIGH risk",          r11["risk_level"]=="HIGH",     r11["risk_level"])
check("Fake listing → REMOVE action",      "REMOVE" in r11["action"],     r11["action"])

r12 = fpd.predict({
    "price_vs_category_avg_ratio":1.1,"description_length":200,
    "image_count":5,"seller_age_days":300,"seller_rating":4.2,
    "seller_total_sales":80,"discount_pct":15,"has_contact_info_in_desc":0
})
check("Genuine listing → GENUINE",         r12["verdict"]=="GENUINE",     r12["verdict"])


# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
total   = len(RESULTS)
passed  = sum(1 for r in RESULTS if r[0]==PASS)
failed  = total - passed
print(f"\n  Total : {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"\n  {'ALL TESTS PASSED ✅' if failed==0 else f'{failed} TEST(S) FAILED ❌'}")
print()
