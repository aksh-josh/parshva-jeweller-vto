import os
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from functools import wraps
from models import db, Product
from werkzeug.utils import secure_filename

admin_bp = Blueprint('admin', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def admin_required(f):
    """Decorator: user must be logged in AND be admin."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            return "Access denied. Admin only.", 403
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------
# ADMIN PAGE
# -------------------------------------------------------------------

@admin_bp.route('/admin')
@admin_required
def admin_page():
    products = Product.query.order_by(Product.category, Product.name).all()
    categories = db.session.query(Product.category).distinct().all()
    categories = sorted([c[0] for c in categories])
    return render_template('admin.html', products=products, categories=categories)


# -------------------------------------------------------------------
# API: UPDATE PRODUCT
# -------------------------------------------------------------------

@admin_bp.route('/api/admin/product/<int:product_id>', methods=['POST'])
@admin_required
def update_product(product_id):
    product = Product.query.get_or_404(product_id)
    data = request.json

    if 'name' in data:
        product.name = data['name']
    if 'price' in data:
        try:
            product.price = float(data['price'])
        except (ValueError, TypeError):
            product.price = 0
    if 'material' in data:
        product.material = data['material']
    if 'category' in data:
        product.category = data['category']
    if 'description' in data:
        product.description = data['description']
    if 'weight' in data:
        product.weight = data['weight']
    if 'is_active' in data:
        product.is_active = bool(data['is_active'])

    db.session.commit()
    return jsonify({'success': True, 'product': product.to_dict()})


# -------------------------------------------------------------------
# API: BULK UPDATE MATERIAL
# -------------------------------------------------------------------

@admin_bp.route('/api/admin/bulk-material', methods=['POST'])
@admin_required
def bulk_update_material():
    data = request.json
    product_ids = data.get('ids', [])
    material = data.get('material', '')

    if not product_ids or not material:
        return jsonify({'success': False, 'message': 'Missing data'})

    Product.query.filter(Product.id.in_(product_ids)).update(
        {'material': material}, synchronize_session='fetch'
    )
    db.session.commit()
    return jsonify({'success': True, 'updated': len(product_ids)})


# -------------------------------------------------------------------
# API: ADD NEW PRODUCT
# -------------------------------------------------------------------

@admin_bp.route('/api/admin/product/new', methods=['POST'])
@admin_required
def add_product():
    name = request.form.get('name', '').strip()
    category = request.form.get('category', '').strip()
    material = request.form.get('material', 'gold')
    price = request.form.get('price', '0')
    description = request.form.get('description', '')
    weight = request.form.get('weight', '')

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid image file'})

    filename = secure_filename(file.filename)
    folder_path = os.path.join(current_app.static_folder, category)
    os.makedirs(folder_path, exist_ok=True)

    filepath = os.path.join(folder_path, filename)
    file.save(filepath)

    image_path = f"{category}/{filename}"

    # Check duplicate
    if Product.query.filter_by(image_path=image_path).first():
        return jsonify({'success': False, 'message': 'Product with this image already exists'})

    product = Product(
        name=name or filename.rsplit('.', 1)[0].replace('_', ' ').title(),
        image_path=image_path,
        category=category,
        material=material,
        price=float(price) if price else 0,
        description=description,
        weight=weight
    )
    db.session.add(product)
    db.session.commit()

    return jsonify({'success': True, 'product': product.to_dict()})


# -------------------------------------------------------------------
# API: DELETE PRODUCT
# -------------------------------------------------------------------

@admin_bp.route('/api/admin/product/<int:product_id>/delete', methods=['POST'])
@admin_required
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    return jsonify({'success': True})