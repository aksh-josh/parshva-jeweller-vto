from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from models import db, Wishlist, Cart, Product

shop_bp = Blueprint('shop', __name__)


# -------------------------------------------------------------------
# SEARCH API
# -------------------------------------------------------------------

@shop_bp.route('/api/search')
def search_products():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify({'results': []})

    products = Product.query.filter(
        Product.is_active == True,
        db.or_(
            Product.name.ilike(f'%{q}%'),
            Product.category.ilike(f'%{q}%'),
            Product.material.ilike(f'%{q}%'),
            Product.description.ilike(f'%{q}%'),
        )
    ).limit(12).all()

    return jsonify({
        'results': [p.to_dict() for p in products],
        'count': len(products)
    })


@shop_bp.route('/search')
def search_page():
    q = request.args.get('q', '').strip()
    products = []
    if q:
        products = Product.query.filter(
            Product.is_active == True,
            db.or_(
                Product.name.ilike(f'%{q}%'),
                Product.category.ilike(f'%{q}%'),
                Product.material.ilike(f'%{q}%'),
                Product.description.ilike(f'%{q}%'),
            )
        ).order_by(Product.name).all()
    return render_template('search_results.html', products=products, query=q)


# -------------------------------------------------------------------
# HELPER: Get price for a product image
# -------------------------------------------------------------------
def get_product_price(image_path):
    product = Product.query.filter_by(image_path=image_path).first()
    if product and product.price > 0:
        return product.price
    return 0


# -------------------------------------------------------------------
# WISHLIST API
# -------------------------------------------------------------------

@shop_bp.route('/api/wishlist/toggle', methods=['POST'])
@login_required
def toggle_wishlist():
    data = request.json
    image = data.get('product_image', '')
    name = data.get('product_name', '')
    category = data.get('product_category', '')

    if not image:
        return jsonify({'success': False, 'message': 'No product specified'})

    existing = Wishlist.query.filter_by(user_id=current_user.id, product_image=image).first()

    if existing:
        db.session.delete(existing)
        db.session.commit()
        count = Wishlist.query.filter_by(user_id=current_user.id).count()
        return jsonify({'success': True, 'action': 'removed', 'message': 'Removed from wishlist', 'count': count})
    else:
        item = Wishlist(user_id=current_user.id, product_image=image, product_name=name, product_category=category)
        db.session.add(item)
        db.session.commit()
        count = Wishlist.query.filter_by(user_id=current_user.id).count()
        return jsonify({'success': True, 'action': 'added', 'message': 'Added to wishlist', 'count': count})


@shop_bp.route('/api/wishlist', methods=['GET'])
@login_required
def get_wishlist():
    items = Wishlist.query.filter_by(user_id=current_user.id).order_by(Wishlist.added_at.desc()).all()
    return jsonify({
        'success': True,
        'items': [{
            'id': item.id,
            'product_image': item.product_image,
            'product_name': item.product_name,
            'product_category': item.product_category,
            'price': get_product_price(item.product_image)
        } for item in items],
        'count': len(items)
    })


@shop_bp.route('/api/wishlist/remove', methods=['POST'])
@login_required
def remove_wishlist():
    data = request.json
    item_id = data.get('id')
    item = Wishlist.query.filter_by(id=item_id, user_id=current_user.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
        count = Wishlist.query.filter_by(user_id=current_user.id).count()
        return jsonify({'success': True, 'count': count})
    return jsonify({'success': False, 'message': 'Item not found'})


# -------------------------------------------------------------------
# CART API
# -------------------------------------------------------------------

@shop_bp.route('/api/cart/add', methods=['POST'])
@login_required
def add_to_cart():
    data = request.json
    image = data.get('product_image', '')
    name = data.get('product_name', '')
    category = data.get('product_category', '')

    if not image:
        return jsonify({'success': False, 'message': 'No product specified'})

    existing = Cart.query.filter_by(user_id=current_user.id, product_image=image).first()

    if existing:
        existing.quantity += 1
        db.session.commit()
    else:
        item = Cart(user_id=current_user.id, product_image=image, product_name=name, product_category=category)
        db.session.add(item)
        db.session.commit()

    count = db.session.query(db.func.sum(Cart.quantity)).filter_by(user_id=current_user.id).scalar() or 0
    return jsonify({'success': True, 'message': 'Added to cart', 'count': int(count)})


@shop_bp.route('/api/cart', methods=['GET'])
@login_required
def get_cart():
    items = Cart.query.filter_by(user_id=current_user.id).order_by(Cart.added_at.desc()).all()
    total_count = sum(item.quantity for item in items)
    total_price = 0
    item_list = []
    for item in items:
        price = get_product_price(item.product_image)
        item_list.append({
            'id': item.id,
            'product_image': item.product_image,
            'product_name': item.product_name,
            'product_category': item.product_category,
            'quantity': item.quantity,
            'price': price,
            'subtotal': price * item.quantity
        })
        total_price += price * item.quantity

    return jsonify({
        'success': True,
        'items': item_list,
        'count': total_count,
        'total': total_price
    })


@shop_bp.route('/api/cart/update', methods=['POST'])
@login_required
def update_cart():
    data = request.json
    item_id = data.get('id')
    quantity = data.get('quantity', 1)

    item = Cart.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Item not found'})

    if quantity <= 0:
        db.session.delete(item)
    else:
        item.quantity = quantity

    db.session.commit()
    count = db.session.query(db.func.sum(Cart.quantity)).filter_by(user_id=current_user.id).scalar() or 0
    return jsonify({'success': True, 'count': int(count)})


@shop_bp.route('/api/cart/remove', methods=['POST'])
@login_required
def remove_from_cart():
    data = request.json
    item_id = data.get('id')
    item = Cart.query.filter_by(id=item_id, user_id=current_user.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
        count = db.session.query(db.func.sum(Cart.quantity)).filter_by(user_id=current_user.id).scalar() or 0
        return jsonify({'success': True, 'count': int(count)})
    return jsonify({'success': False, 'message': 'Item not found'})


# -------------------------------------------------------------------
# BADGE COUNTS
# -------------------------------------------------------------------

@shop_bp.route('/api/counts')
def get_counts():
    if not current_user.is_authenticated:
        return jsonify({'wishlist': 0, 'cart': 0})

    wishlist_count = Wishlist.query.filter_by(user_id=current_user.id).count()
    cart_count = db.session.query(db.func.sum(Cart.quantity)).filter_by(user_id=current_user.id).scalar() or 0
    return jsonify({'wishlist': wishlist_count, 'cart': int(cart_count)})


# -------------------------------------------------------------------
# PAGES
# -------------------------------------------------------------------

@shop_bp.route('/wishlist')
@login_required
def wishlist_page():
    items = Wishlist.query.filter_by(user_id=current_user.id).order_by(Wishlist.added_at.desc()).all()
    # Attach prices
    for item in items:
        item.price = get_product_price(item.product_image)
    return render_template('wishlist.html', items=items)


@shop_bp.route('/cart')
@login_required
def cart_page():
    items = Cart.query.filter_by(user_id=current_user.id).order_by(Cart.added_at.desc()).all()
    total = 0
    for item in items:
        item.price = get_product_price(item.product_image)
        item.subtotal = item.price * item.quantity
        total += item.subtotal
    return render_template('cart.html', items=items, total=total)


# -------------------------------------------------------------------
# MATERIAL PAGES (Gold, Silver, Diamond, Daily Wear)
# -------------------------------------------------------------------

@shop_bp.route('/material/<material_type>')
def material_page(material_type):
    valid = ['gold', 'silver', 'diamond', 'daily-wear']
    if material_type not in valid:
        return "Material not found", 404

    products = Product.query.filter_by(material=material_type, is_active=True).order_by(Product.category, Product.name).all()

    titles = {
        'gold': 'Gold Collection',
        'silver': 'Silver Collection',
        'diamond': 'Diamond Collection',
        'daily-wear': 'Daily Wear (Anti-Tarnish)'
    }

    return render_template('material.html',
                           products=products,
                           material=material_type,
                           title=titles.get(material_type, material_type.title()))


# -------------------------------------------------------------------
# BROWSE: /browse/<material>/<subcategory> (e.g. /browse/gold/earrings)
# -------------------------------------------------------------------

@shop_bp.route('/browse/<material>/<subcategory>')
def browse(material, subcategory):
    from site_structure import SUBCATEGORY_LABELS, SUBCATEGORY_FOLDER_MAP

    # Map subcategory slug to DB category field
    folder = SUBCATEGORY_FOLDER_MAP.get(subcategory, subcategory)

    products = Product.query.filter_by(
        material=material,
        category=folder,
        is_active=True
    ).order_by(Product.name).all()

    sub_label = SUBCATEGORY_LABELS.get(subcategory, subcategory.title())
    mat_labels = {'gold': 'Gold', 'silver': 'Silver', 'diamond': 'Diamond', 'daily-wear': 'Daily Wear'}
    mat_label = mat_labels.get(material, material.title())

    return render_template('browse.html',
                           products=products,
                           material=material,
                           subcategory=subcategory,
                           title=f"{mat_label} {sub_label}")


# -------------------------------------------------------------------
# COLLECTIONS (e.g. /collections/kundan-stories)
# -------------------------------------------------------------------

@shop_bp.route('/collections/<collection>')
def collection_page(collection):
    from site_structure import COLLECTIONS

    info = COLLECTIONS.get(collection)
    if not info:
        return "Collection not found", 404

    # Products tagged with this collection name in their category or description
    products = Product.query.filter_by(is_active=True).filter(
        db.or_(
            Product.category.like(f'%{collection.replace("-", "%")}%'),
            Product.description.like(f'%{collection}%'),
            Product.category == collection.replace('-', '_'),
            Product.category == collection,
        )
    ).order_by(Product.name).all()

    return render_template('collection.html',
                           products=products,
                           collection=collection,
                           title=info['label'],
                           description=info['description'])


# -------------------------------------------------------------------
# WEDDING (e.g. /wedding/wedding-necklaces)
# -------------------------------------------------------------------

@shop_bp.route('/wedding/<section>')
def wedding_page(section):
    from site_structure import WEDDING

    info = WEDDING.get(section)
    if not info:
        return "Section not found", 404

    products = Product.query.filter_by(is_active=True).filter(
        db.or_(
            Product.category == section,
            Product.category == section.replace('-', '_'),
            Product.category == 'wedding',
        )
    ).order_by(Product.name).all()

    return render_template('collection.html',
                           products=products,
                           collection=section,
                           title=info['label'],
                           description=info['description'])


# -------------------------------------------------------------------
# GIFTING (e.g. /gifting/for-her)
# -------------------------------------------------------------------

@shop_bp.route('/gifting/<section>')
def gifting_page(section):
    from site_structure import GIFTING

    info = GIFTING.get(section)
    if not info:
        return "Section not found", 404

    products = Product.query.filter_by(is_active=True).filter(
        db.or_(
            Product.category == section,
            Product.category == section.replace('-', '_'),
            Product.category == 'gifting',
        )
    ).order_by(Product.name).all()

    return render_template('collection.html',
                           products=products,
                           collection=section,
                           title=info['label'],
                           description=info['description'])