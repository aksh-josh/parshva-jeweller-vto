"""
AI-Powered Jewelry Recommendation Engine
=========================================
Uses image feature embeddings (color histograms, texture, structure)
and product metadata to recommend complementary jewelry pieces.

Example: Viewing a gold necklace → suggests matching gold earrings, bangles, rings.
"""

import os
import numpy as np
import cv2
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Complementary category mapping: if viewing X, suggest from Y categories
COMPLEMENTARY_CATEGORIES = {
    'necklace':    ['jhumka', 'bangles', 'ring', 'mangalsutra'],
    'mangalsutra': ['jhumka', 'bangles', 'ring', 'chain'],
    'jhumka':      ['necklace', 'bangles', 'ring', 'mangalsutra'],
    'chain':       ['jhumka', 'ring', 'bangles'],
    'ring':        ['bangles', 'jhumka', 'necklace', 'chain'],
    'bangles':     ['ring', 'jhumka', 'necklace', 'mangalsutra'],
    'rajwadi':     ['jhumka', 'bangles', 'necklace', 'ring'],
}

# Also handle collection/wedding/gifting folders
COLLECTION_COMPLEMENTS = {
    'kundan-stories':     ['jhumka', 'bangles', 'necklace'],
    'rajwadi-heritage':   ['jhumka', 'bangles', 'necklace'],
    'polki-collection':   ['jhumka', 'bangles', 'ring'],
    'festive-collection': ['jhumka', 'necklace', 'bangles'],
    'wedding-necklaces':  ['wedding-earrings', 'wedding-bangles', 'wedding-sets'],
    'wedding-bangles':    ['wedding-necklaces', 'wedding-earrings', 'wedding-sets'],
    'wedding-earrings':   ['wedding-necklaces', 'wedding-bangles', 'wedding-sets'],
    'wedding-sets':       ['wedding-necklaces', 'wedding-bangles', 'wedding-earrings'],
    'bridal-mangalsutra': ['wedding-necklaces', 'wedding-earrings', 'wedding-bangles'],
    'for-her':            ['jhumka', 'necklace', 'bangles', 'ring'],
    'for-him':            ['chain', 'ring'],
    'for-kids':           ['jhumka', 'bangles', 'ring'],
}

# Merge both maps
ALL_COMPLEMENTS = {**COMPLEMENTARY_CATEGORIES, **COLLECTION_COMPLEMENTS}

# Embedding dimension
EMBEDDING_DIM = 166  # 128 (color hist) + 32 (texture) + 6 (metadata)


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_color_histogram(image, bins=32):
    """
    Extract a normalized color histogram in HSV space.
    HSV is better than RGB for jewelry because it separates
    hue (gold vs silver tone) from brightness.
    Returns a 96-dim vector (32 bins x 3 channels) → compressed to 128 via padding.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []

    # Calculate histogram for each channel
    for i, (upper, name) in enumerate([(180, 'H'), (256, 'S'), (256, 'V')]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, upper])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Pad to 128 dims
    features = features[:128]
    while len(features) < 128:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def extract_texture_features(image, size=64):
    """
    Extract texture features using Gabor filters at multiple orientations.
    Captures the pattern/texture of jewelry (filigree, stone settings, etc.)
    Returns a 32-dim vector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))

    features = []
    # Gabor filters at 8 orientations × 4 frequencies
    for theta_idx in range(8):
        theta = theta_idx * np.pi / 8
        for freq in [0.1, 0.2, 0.3, 0.4]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21), sigma=4.0,
                theta=theta, lambd=1.0/freq,
                gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(filtered.mean())
            features.append(filtered.std())

    # Reduce to 32 dims by taking every other feature
    features = np.array(features, dtype=np.float32)
    if len(features) > 32:
        indices = np.linspace(0, len(features) - 1, 32, dtype=int)
        features = features[indices]

    return features


def encode_metadata(material, category):
    """
    Encode product metadata as a 6-dim feature vector.
    Gives weight to material matching (gold↔gold is preferred).
    """
    material_map = {
        'gold': [1.0, 0.0, 0.0],
        'silver': [0.0, 1.0, 0.0],
        'diamond': [0.0, 0.0, 1.0],
        'daily-wear': [0.3, 0.3, 0.3],
        'antique': [0.5, 0.2, 0.0],
    }

    # Category type encoding (body region)
    category_type = {
        'necklace': [1.0, 0.0, 0.0],
        'mangalsutra': [0.9, 0.0, 0.1],
        'chain': [0.8, 0.0, 0.2],
        'jhumka': [0.0, 1.0, 0.0],
        'ring': [0.0, 0.0, 1.0],
        'bangles': [0.0, 0.3, 0.7],
        'rajwadi': [0.5, 0.5, 0.0],
    }

    mat_vec = material_map.get(material, [0.3, 0.3, 0.3])
    cat_vec = category_type.get(category, [0.3, 0.3, 0.3])

    return np.array(mat_vec + cat_vec, dtype=np.float32)


def extract_embedding(image_path, static_folder, material='gold', category='necklace'):
    """
    Generate a full embedding vector for a product image.
    Combines: color histogram (128) + texture (32) + metadata (6) = 166 dims
    """
    full_path = os.path.join(static_folder, image_path)

    if not os.path.exists(full_path):
        return None

    image = cv2.imread(full_path)
    if image is None:
        return None

    # Resize for consistent processing
    image = cv2.resize(image, (224, 224))

    # Extract features
    color_feat = extract_color_histogram(image)
    texture_feat = extract_texture_features(image)
    meta_feat = encode_metadata(material, category)

    # Concatenate into single embedding
    embedding = np.concatenate([color_feat, texture_feat, meta_feat])

    # L2 normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ---------------------------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

class JewelryRecommender:
    """
    AI recommendation engine that uses product embeddings to find
    visually and stylistically complementary jewelry pieces.
    """

    def __init__(self):
        self.embeddings = {}       # {product_id: embedding_vector}
        self.product_info = {}     # {product_id: {name, image_path, category, material, price}}
        self.is_built = False

    def build_index(self, products, static_folder):
        """
        Build the embedding index from all products in the database.
        Call this on app startup or after adding new products.
        """
        print("[AI Recommender] Building embedding index...")
        count = 0

        for product in products:
            embedding = extract_embedding(
                product.image_path,
                static_folder,
                material=product.material or 'gold',
                category=product.category or 'necklace'
            )

            if embedding is not None:
                self.embeddings[product.id] = embedding
                self.product_info[product.id] = {
                    'id': product.id,
                    'name': product.name,
                    'image_path': product.image_path,
                    'category': product.category,
                    'material': product.material,
                    'price': product.price,
                    'display_price': product.display_price,
                }
                count += 1

        self.is_built = True
        print(f"[AI Recommender] Indexed {count} products with {EMBEDDING_DIM}-dim embeddings.")

    def get_recommendations(self, product_id, max_results=8):
        """
        Get AI-powered recommendations for a given product.

        Strategy:
        1. Find the product's embedding
        2. Look at complementary categories
        3. Rank by cosine similarity (prioritize same material)
        4. Return top N diverse recommendations
        """
        if not self.is_built or product_id not in self.embeddings:
            return []

        query_embedding = self.embeddings[product_id]
        query_info = self.product_info[product_id]
        query_category = query_info['category']
        query_material = query_info['material']

        # Get complementary categories
        complement_cats = ALL_COMPLEMENTS.get(query_category, [])
        if not complement_cats:
            # Fallback: suggest from any category except the same one
            all_cats = set(p['category'] for p in self.product_info.values())
            complement_cats = [c for c in all_cats if c != query_category]

        # Score all candidate products
        candidates = []
        for pid, embedding in self.embeddings.items():
            if pid == product_id:
                continue

            info = self.product_info[pid]

            # Must be from a complementary category
            if info['category'] not in complement_cats:
                continue

            # Cosine similarity (embeddings are already L2-normalized)
            similarity = float(np.dot(query_embedding, embedding))

            # Boost score if same material (strong match signal)
            material_boost = 0.25 if info['material'] == query_material else 0.0

            # Final score
            score = similarity + material_boost

            candidates.append({
                'product': info,
                'score': round(score, 4),
                'similarity': round(similarity, 4),
                'reason': self._get_reason(query_info, info, similarity)
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Ensure diversity: max 3 from any single category
        result = []
        category_counts = defaultdict(int)

        for candidate in candidates:
            cat = candidate['product']['category']
            if category_counts[cat] >= 3:
                continue
            result.append(candidate)
            category_counts[cat] += 1
            if len(result) >= max_results:
                break

        return result

    def _get_reason(self, query_info, rec_info, similarity):
        """Generate a human-readable reason for the recommendation."""
        reasons = []

        if rec_info['material'] == query_info['material']:
            material_name = rec_info['material'].title()
            reasons.append(f"Matching {material_name} tone")

        if similarity > 0.85:
            reasons.append("Very similar style")
        elif similarity > 0.7:
            reasons.append("Complementary design")
        elif similarity > 0.5:
            reasons.append("Goes well together")

        cat_display = rec_info['category'].replace('-', ' ').title()
        reasons.append(f"Pairs with your {query_info['category'].replace('-', ' ')}")

        return " · ".join(reasons[:2]) if reasons else "Recommended for you"

    def get_similar_products(self, product_id, max_results=4):
        """
        Find visually similar products in the SAME category.
        Useful for "You might also like" within the same type.
        """
        if not self.is_built or product_id not in self.embeddings:
            return []

        query_embedding = self.embeddings[product_id]
        query_info = self.product_info[product_id]

        candidates = []
        for pid, embedding in self.embeddings.items():
            if pid == product_id:
                continue

            info = self.product_info[pid]
            if info['category'] != query_info['category']:
                continue

            similarity = float(np.dot(query_embedding, embedding))
            candidates.append({
                'product': info,
                'score': round(similarity, 4),
                'similarity': round(similarity, 4),
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:max_results]


# ---------------------------------------------------------------------------
# SINGLETON INSTANCE
# ---------------------------------------------------------------------------
recommender = JewelryRecommender()