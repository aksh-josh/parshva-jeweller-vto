import os
from urllib.parse import quote_plus

# -------------------------------------------------------------------
# DATABASE (MySQL)
# -------------------------------------------------------------------
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'db')
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'Akshat@2004')  # ← YOUR PASSWORD
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'parshva_jewellers')

# URL-encode the password so special characters (@, !, #, etc.) don't break the URL
SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}/{MYSQL_DATABASE}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# -------------------------------------------------------------------
# FLASK
# -------------------------------------------------------------------
SECRET_KEY = os.environ.get('SECRET_KEY', 'parshva-jewellers-secret-key-change-in-production')

# -------------------------------------------------------------------
# TWILIO (SMS OTP) — optional, OTP prints to console without it
# -------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'YOUR_TWILIO_SID_HERE')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'YOUR_TWILIO_TOKEN_HERE')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '+1XXXXXXXXXX')

# -------------------------------------------------------------------
# OTP SETTINGS
# -------------------------------------------------------------------
OTP_EXPIRY_SECONDS = 300  # 5 minutes
OTP_LENGTH = 6