"""
auth.py — FIXED VERSION
========================
FIXES:
  FIX 2a: logout() — properly clears Flask-Login session + server-side session
  FIX 2b: api_login() — OTP is ALWAYS sent on every login, not just first time
  FIX 2c: api_verify_otp() — unified purpose handling so both signup/login work
  FIX 2d: Added /api/auth/logout POST endpoint for AJAX logout from navbar
"""

import random
import string
from datetime import datetime, timedelta

from flask import (Blueprint, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from flask_login import login_user, logout_user, login_required, current_user

from models import db, User, OTPRecord
import config

auth_bp = Blueprint('auth', __name__)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def generate_otp():
    """Generate a random 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=config.OTP_LENGTH))


def send_sms_otp(phone, otp_code):
    """
    Send OTP via Twilio SMS.
    Returns True if sent, False if failed.
    """
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Your Parshva Jewellers verification code is: {otp_code}. Valid for 5 minutes.",
            from_=config.TWILIO_PHONE_NUMBER,
            to=phone
        )
        print(f"[SMS] OTP sent to {phone}: SID {message.sid}")
        return True
    except Exception as e:
        print(f"[SMS ERROR] {e}")
        print(f"[DEV] OTP for {phone}: {otp_code}")
        return False


def create_and_send_otp(phone, purpose):
    """
    Create OTP record in DB and send via SMS.
    FIX 2b: Always invalidates old OTPs and creates a fresh one — no caching.
    Returns (success: bool, message: str, otp_for_dev: str)
    """
    otp_code   = generate_otp()
    expires_at = datetime.utcnow() + timedelta(seconds=config.OTP_EXPIRY_SECONDS)

    # Invalidate ALL previous unused OTPs for this phone (both signup and login)
    OTPRecord.query.filter_by(phone=phone, is_used=False).update({'is_used': True})
    db.session.commit()

    # Save new OTP
    otp_record = OTPRecord(
        phone=phone,
        otp_code=otp_code,
        purpose=purpose,
        expires_at=expires_at
    )
    db.session.add(otp_record)
    db.session.commit()

    sms_sent = send_sms_otp(phone, otp_code)

    if sms_sent:
        return True, "OTP sent to your phone.", otp_code
    else:
        return True, "OTP generated (check console if SMS not configured).", otp_code


def verify_otp(phone, otp_code, purpose):
    """
    FIX 2c: Accept OTP regardless of purpose to handle both login and signup.
    Returns (success: bool, message: str)
    """
    # Try exact purpose match first
    otp_record = OTPRecord.query.filter_by(
        phone=phone,
        otp_code=otp_code,
        is_used=False
    ).order_by(OTPRecord.created_at.desc()).first()

    if not otp_record:
        return False, "Invalid OTP. Please try again."

    if datetime.utcnow() > otp_record.expires_at:
        otp_record.is_used = True
        db.session.commit()
        return False, "OTP has expired. Please request a new one."

    otp_record.is_used = True
    db.session.commit()
    return True, "OTP verified successfully."


# -------------------------------------------------------------------
# ROUTES — PAGES
# -------------------------------------------------------------------

@auth_bp.route('/auth')
def auth_page():
    """Render the sign in / sign up page."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('auth.html')


@auth_bp.route('/profile')
@login_required
def profile_page():
    return render_template('profile.html')


@auth_bp.route('/logout')
def logout():
    """
    FIX 2a: Proper logout — clears Flask-Login + full session wipe.
    Works whether user was logged in or not (no @login_required to avoid redirect loops).
    """
    logout_user()          # clears Flask-Login user from session
    session.clear()        # wipes entire server-side session (remember me cookie etc.)
    # Redirect with cache-control response
    response = redirect(url_for('home'))
    # Delete the session cookie explicitly
    response.delete_cookie(
        app_session_cookie_name(),
        path='/',
        domain=None
    )
    return response


def app_session_cookie_name():
    """Get the session cookie name from Flask config (default 'session')."""
    try:
        from flask import current_app
        return current_app.session_cookie_name
    except Exception:
        return 'session'


# -------------------------------------------------------------------
# API ROUTES — SIGN UP
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """Step 1: Register user + send OTP."""
    data      = request.json or {}
    full_name = data.get('full_name', '').strip()
    phone     = data.get('phone', '').strip()
    email     = (data.get('email') or '').strip() or None

    if not full_name or len(full_name) < 2:
        return jsonify({'success': False, 'message': 'Please enter your full name.'})

    if not phone or len(phone) < 10:
        return jsonify({'success': False, 'message': 'Please enter a valid phone number.'})

    if not phone.startswith('+'):
        phone = '+91' + phone

    existing = User.query.filter_by(phone=phone).first()
    if existing and existing.is_verified:
        return jsonify({'success': False, 'message': 'This phone number is already registered. Please sign in.'})

    if existing:
        existing.full_name = full_name
        existing.email     = email
    else:
        new_user = User(full_name=full_name, phone=phone, email=email)
        db.session.add(new_user)

    db.session.commit()

    success, message, otp = create_and_send_otp(phone, 'signup')

    return jsonify({
        'success':  success,
        'message':  message,
        'dev_otp':  otp
    })


# -------------------------------------------------------------------
# API ROUTES — VERIFY OTP
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/verify-otp', methods=['POST'])
def api_verify_otp():
    """Step 2: Verify OTP and log user in."""
    data     = request.json or {}
    phone    = data.get('phone', '').strip()
    otp_code = data.get('otp', '').strip()
    purpose  = data.get('purpose', 'login')

    if not phone.startswith('+'):
        phone = '+91' + phone

    success, message = verify_otp(phone, otp_code, purpose)

    if not success:
        return jsonify({'success': False, 'message': message})

    user = User.query.filter_by(phone=phone).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found. Please sign up first.'})

    user.is_verified = True
    user.last_login  = datetime.utcnow()
    db.session.commit()

    login_user(user, remember=True)

    return jsonify({
        'success': True,
        'message': 'Welcome, ' + user.full_name + '!',
        'user': {
            'name':  user.full_name,
            'phone': user.phone
        }
    })


# -------------------------------------------------------------------
# API ROUTES — SIGN IN
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/login', methods=['POST'])
def api_login():
    """
    FIX 2b: Always sends a fresh OTP on every login request.
    No skipping, no "already sent" logic.
    """
    data  = request.json or {}
    phone = data.get('phone', '').strip()

    if not phone or len(phone) < 10:
        return jsonify({'success': False, 'message': 'Please enter a valid phone number.'})

    if not phone.startswith('+'):
        phone = '+91' + phone

    user = User.query.filter_by(phone=phone, is_verified=True).first()
    if not user:
        return jsonify({'success': False, 'message': 'No account found with this number. Please sign up.'})

    # Always create and send a fresh OTP
    success, message, otp = create_and_send_otp(phone, 'login')

    return jsonify({
        'success':   success,
        'message':   message,
        'user_name': user.full_name,
        'dev_otp':   otp
    })


# -------------------------------------------------------------------
# API ROUTES — RESEND OTP
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/resend-otp', methods=['POST'])
def api_resend_otp():
    """Resend OTP to phone — always generates a fresh code."""
    data    = request.json or {}
    phone   = data.get('phone', '').strip()
    purpose = data.get('purpose', 'login')

    if not phone.startswith('+'):
        phone = '+91' + phone

    success, message, otp = create_and_send_otp(phone, purpose)

    return jsonify({
        'success': success,
        'message': 'New OTP sent.',
        'dev_otp': otp
    })


# -------------------------------------------------------------------
# API — LOGOUT (AJAX)
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """
    FIX 2a: AJAX logout endpoint — used by navbar logout button.
    Returns JSON so JS can redirect after clearing state.
    """
    logout_user()
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully.'})


# -------------------------------------------------------------------
# API — CHECK AUTH STATUS
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/status')
def api_auth_status():
    """Check if user is logged in (for JS)."""
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'user': {
                'name':  current_user.full_name,
                'phone': current_user.phone
            }
        })
    return jsonify({'logged_in': False})