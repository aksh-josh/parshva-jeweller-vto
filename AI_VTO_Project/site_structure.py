# site_structure.py
# Single source of truth for the entire site navigation and category structure.
# Edit this file to add/remove nav items — everything else reads from here.

MATERIALS = {
    'gold': {
        'label': 'Gold',
        'subcategories': ['earrings', 'necklaces', 'mangalsutra', 'bangles', 'rings', 'chains'],
    },
    'silver': {
        'label': 'Silver',
        'subcategories': ['earrings', 'necklaces', 'bangles', 'rings', 'chains'],
    },
    'diamond': {
        'label': 'Diamond',
        'subcategories': ['earrings', 'necklaces', 'mangalsutra', 'bangles', 'rings', 'pendants'],
    },
    'daily-wear': {
        'label': 'Daily Wear (Anti-Tarnish)',
        'subcategories': ['earrings', 'necklaces', 'mangalsutra', 'bangles', 'rings', 'chains'],
    },
}

COLLECTIONS = {
    'kundan-stories': {
        'label': 'Kundan Stories',
        'description': 'Traditional Kundan jewellery with intricate stone settings',
    },
    'rajwadi-heritage': {
        'label': 'Rajwadi Heritage',
        'description': 'Royal Rajwadi pieces inspired by Rajasthani tradition',
    },
    'polki-collection': {
        'label': 'Polki Collection',
        'description': 'Uncut diamond Polki jewellery with timeless charm',
    },
    'festive-collection': {
        'label': 'Festive Collection',
        'description': 'Celebration-ready jewellery for every festival',
    },
}

WEDDING = {
    'wedding-necklaces': {
        'label': 'Wedding Necklaces',
        'description': 'Statement necklaces for the bride',
    },
    'wedding-bangles': {
        'label': 'Wedding Bangles',
        'description': 'Bridal bangles and kadas',
    },
    'wedding-earrings': {
        'label': 'Wedding Earrings',
        'description': 'Stunning bridal earrings',
    },
    'wedding-sets': {
        'label': 'Wedding Sets',
        'description': 'Complete bridal jewellery sets',
    },
    'bridal-mangalsutra': {
        'label': 'Bridal Mangalsutra',
        'description': 'Sacred mangalsutra for the new bride',
    },
}

GIFTING = {
    'for-her': {
        'label': 'For Her',
        'description': 'Beautiful gifts for the special woman in your life',
    },
    'for-him': {
        'label': 'For Him',
        'description': 'Elegant pieces for the modern gentleman',
    },
    'for-kids': {
        'label': 'For Kids',
        'description': 'Adorable jewellery for little ones',
    },
}

# Display names for subcategories
SUBCATEGORY_LABELS = {
    'earrings': 'Earrings',
    'necklaces': 'Necklaces',
    'mangalsutra': 'Mangalsutra',
    'bangles': 'Bangles',
    'rings': 'Rings',
    'chains': 'Chains',
    'pendants': 'Pendants',
    'nose-pins': 'Nose Pins',
    'anklets': 'Anklets',
}

# Map subcategory URL slug → static folder name (for image loading)
SUBCATEGORY_FOLDER_MAP = {
    'earrings': 'jhumka',
    'necklaces': 'necklace',
    'mangalsutra': 'mangalsutra',
    'bangles': 'bangles',
    'rings': 'ring',
    'chains': 'chain',
    'pendants': 'necklace',  # update when you have a separate pendants folder
}

# All collection/wedding/gifting folder names (used for auto-import)
COLLECTION_FOLDERS = ['kundan-stories', 'rajwadi-heritage', 'polki-collection', 'festive-collection']
WEDDING_FOLDERS = ['wedding-necklaces', 'wedding-bangles', 'wedding-earrings', 'wedding-sets', 'bridal-mangalsutra']
GIFTING_FOLDERS = ['for-her', 'for-him', 'for-kids']