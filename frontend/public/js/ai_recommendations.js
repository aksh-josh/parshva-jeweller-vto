/**
 * AI Jewelry Recommendations Widget
 * ==================================
 * Loads AI-powered recommendations for any product and renders
 * a "Complete Your Look" carousel below product cards.
 *
 * Usage: Add this script to base.html, then call:
 *   showRecommendations(productId, containerElementId)
 *
 * Or auto-attach to product cards with data-product-id attribute.
 */

// Fetch recommendations from API
async function fetchRecommendations(productId, limit = 8) {
    try {
        const response = await fetch(`/api/recommendations/${productId}?limit=${limit}`);
        const data = await response.json();
        if (data.success) {
            return data.recommendations;
        }
        return [];
    } catch (err) {
        console.error('[AI Rec] Error fetching recommendations:', err);
        return [];
    }
}

// Fetch similar products from API
async function fetchSimilar(productId, limit = 4) {
    try {
        const response = await fetch(`/api/similar/${productId}?limit=${limit}`);
        const data = await response.json();
        if (data.success) {
            return data.similar;
        }
        return [];
    } catch (err) {
        console.error('[AI Rec] Error fetching similar:', err);
        return [];
    }
}

// Render a recommendation card
function createRecCard(item) {
    const card = document.createElement('div');
    card.className = 'flex-shrink-0 w-44 group cursor-pointer';
    card.innerHTML = `
        <div class="relative h-44 rounded-xl overflow-hidden shadow-sm group-hover:shadow-lg transition-all duration-300 group-hover:-translate-y-1 bg-gray-100 dark:bg-gray-700">
            <img src="/static/${item.image_path}" alt="${item.name}"
                 class="w-full h-full object-cover group-hover:scale-105 transition duration-500" loading="lazy">
            <div class="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition"></div>
            ${item.score ? `<span class="absolute top-2 right-2 bg-yellow-700 text-white text-[9px] px-1.5 py-0.5 rounded-full font-bold">${Math.round(item.score * 100)}% match</span>` : ''}
        </div>
        <div class="mt-2 px-0.5">
            <p class="text-xs font-semibold text-gray-800 dark:text-white truncate">${item.name}</p>
            <p class="text-[10px] text-gray-400 capitalize">${item.category.replace(/-/g, ' ')} · ${item.material}</p>
            ${item.display_price ? `<p class="text-xs font-bold text-yellow-700 dark:text-yellow-500 mt-0.5">${item.display_price}</p>` : ''}
            ${item.reason ? `<p class="text-[10px] text-gray-400 mt-0.5 italic truncate">${item.reason}</p>` : ''}
        </div>
    `;

    card.addEventListener('click', function() {
        // Navigate to browse page for that category
        window.location.href = `/material/${item.material}`;
    });

    return card;
}

// Show recommendations in a container
async function showRecommendations(productId, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const recs = await fetchRecommendations(productId);
    if (recs.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.innerHTML = `
        <div class="mt-8 mb-4">
            <div class="flex items-center justify-between mb-4">
                <div>
                    <h3 class="text-lg font-bold brand-font text-gray-900 dark:text-white">Complete Your Look</h3>
                    <p class="text-xs text-gray-400">AI-powered recommendations based on style matching</p>
                </div>
                <span class="text-[10px] bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-400 px-2 py-1 rounded-full font-semibold">
                    ✨ AI Powered
                </span>
            </div>
            <div id="rec-scroll-${productId}" class="flex gap-4 overflow-x-auto pb-3 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600" style="-webkit-overflow-scrolling: touch; scroll-behavior: smooth;"></div>
        </div>
    `;

    const scrollContainer = document.getElementById(`rec-scroll-${productId}`);
    recs.forEach(item => {
        scrollContainer.appendChild(createRecCard(item));
    });
}

// Auto-initialize: look for elements with data-recommend-for attribute
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('[data-recommend-for]').forEach(function(el) {
        const productId = el.getAttribute('data-recommend-for');
        if (productId) {
            showRecommendations(parseInt(productId), el.id);
        }
    });
});


// =====================================================================
// VTO Accuracy Tracking (frontend)
// =====================================================================

class VTOAccuracyClient {
    constructor() {
        this.isTracking = false;
        this.frameCount = 0;
    }

    async startSession(category) {
        const response = await fetch('/api/vto/accuracy/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: category })
        });
        const data = await response.json();
        this.isTracking = data.success;
        this.frameCount = 0;
        return data;
    }

    async recordFrame(faceDetected, confidence, landmarks) {
        if (!this.isTracking) return;

        this.frameCount++;

        // Only send every 3rd frame to reduce network load
        if (this.frameCount % 3 !== 0) return;

        const landmarkData = landmarks ? landmarks.map(lm => ({
            x: lm.x || 0,
            y: lm.y || 0,
            z: lm.z || 0,
            visibility: lm.visibility || 1.0
        })) : null;

        try {
            await fetch('/api/vto/accuracy/frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    face_detected: faceDetected,
                    confidence: confidence,
                    landmarks: landmarkData
                })
            });
        } catch (err) {
            // Silent fail — don't interrupt VTO experience
        }
    }

    async getReport() {
        const response = await fetch('/api/vto/accuracy/report');
        const data = await response.json();
        return data;
    }

    async saveReport() {
        const response = await fetch('/api/vto/accuracy/save', { method: 'POST' });
        const data = await response.json();
        return data;
    }
}

// Global instance
const vtoAccuracy = new VTOAccuracyClient();