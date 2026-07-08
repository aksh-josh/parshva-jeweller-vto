from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import os
import time

import pymysql

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    wishlist_items = db.relationship('Wishlist', backref='user', lazy=True, cascade='all, delete-orphan')
    cart_items = db.relationship('Cart', backref='user', lazy=True, cascade='all, delete-orphan')


class OTPRecord(db.Model):
    __tablename__ = 'otp_records'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    phone = db.Column(db.String(15), nullable=False, index=True)
    otp_code = db.Column(db.String(6), nullable=False)
    purpose = db.Column(db.String(20), nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)


class Product(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(255), unique=True, nullable=False)  # "jhumka/earring1.jpg"
    category = db.Column(db.String(50), nullable=False)                   # jhumka, necklace, chain, etc.
    material = db.Column(db.String(50), default='gold')                   # gold, silver, diamond, antique
    price = db.Column(db.Float, default=0)                                # 0 = placeholder
    description = db.Column(db.Text, nullable=True)
    weight = db.Column(db.String(50), nullable=True)                      # e.g. "12.5g"
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def display_price(self):
        if self.price and self.price > 0:
            return f"₹{self.price:,.0f}"
        return "₹ Price on Request"

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'image_path': self.image_path,
            'category': self.category,
            'material': self.material,
            'price': self.price,
            'display_price': self.display_price,
            'description': self.description,
            'weight': self.weight,
            'is_active': self.is_active
        }


class Wishlist(db.Model):
    __tablename__ = 'wishlist'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_image = db.Column(db.String(255), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    product_category = db.Column(db.String(50), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'product_image', name='unique_wishlist_item'),
    )


class Cart(db.Model):
    __tablename__ = 'cart'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_image = db.Column(db.String(255), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    product_category = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'product_image', name='unique_cart_item'),
    )


def import_products_from_static(app):
    """Scan static folders and create Product entries for any new images found."""
    static_path = app.static_folder
    folders = [
        # Main categories
        'necklace', 'mangalsutra', 'rajwadi', 'jhumka', 'chain', 'ring', 'bangles',
        # Collections
        'kundan-stories', 'rajwadi-heritage', 'polki-collection', 'festive-collection',
        # Wedding
        'wedding-necklaces', 'wedding-bangles', 'wedding-earrings', 'wedding-sets', 'bridal-mangalsutra',
        # Gifting
        'for-her', 'for-him', 'for-kids',
    ]

    count = 0
    for folder in folders:
        folder_path = os.path.join(static_path, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            if filename.lower() == 'cover.jpeg':
                continue

            image_path = f"{folder}/{filename}"

            # Skip if already in DB
            existing = Product.query.filter_by(image_path=image_path).first()
            if existing:
                continue

            name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()

            product = Product(
                name=name,
                image_path=image_path,
                category=folder,
                material='gold',  # default — admin can change later
                price=0,          # placeholder — admin sets real price
            )
            db.session.add(product)
            count += 1

    if count > 0:
        db.session.commit()
        print(f"[DB] Imported {count} new products from static folders.")


def _connect_with_retry(host, user, password, port, max_retries=15, retry_delay=2):
    """Try connecting to MySQL, retrying with a short delay if it's not ready yet.

    MySQL containers go through a two-phase startup (temp init server, then
    the real server), so the very first connection attempt(s) after the
    container starts can fail with 'Connection refused' even though the
    container itself is up. Retrying a handful of times avoids hard-failing
    the whole app on that race.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                port=port,
            )
            print(f"[DB] Connected to MySQL on attempt {attempt}/{max_retries}")
            return conn
        except pymysql.err.OperationalError as e:
            last_err = e
            print(f"[DB] MySQL not ready yet (attempt {attempt}/{max_retries}): {e}")
            time.sleep(retry_delay)
    # Ran out of retries — raise the last error so it still fails loudly.
    raise last_err


def init_db(app):
    """Initialize database, create tables, and import products."""
    db.init_app(app)
    with app.app_context():
        import pymysql
        import os
        print("MYSQL_HOST =", os.getenv("MYSQL_HOST"))
        print("MYSQL_USER =", os.getenv("MYSQL_USER"))
        print("MYSQL_PASSWORD =", os.getenv("MYSQL_PASSWORD"))
        print("MYSQL_DATABASE =", os.getenv("MYSQL_DATABASE"))
        conn = _connect_with_retry(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            port=3306,
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {app.config.get('MYSQL_DATABASE', 'parshva_jewellers')}")
        conn.close()

        db.create_all()
        print("[DB] Tables created successfully.")

        # Auto-import product images
        import_products_from_static(app)