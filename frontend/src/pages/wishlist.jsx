import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export default function Wishlist() {
  // Mock State
  const [items, setItems] = useState([]); // Wiped out the hardcoded data!

  const removeWishlist = (id) => {
    setItems(items.filter(item => item.id !== id));
    // FUTURE: await fetch('/api/wishlist/remove', { ... })
  };

  const moveToCart = (id, name) => {
    // In React, we trigger the remove instantly to make it feel fast, then alert the user
    removeWishlist(id);
    alert(`${name} moved to cart!`);
    // FUTURE: await fetch('/api/cart/add', { ... })
  };

  return (
    <section className="py-16 px-4 min-h-[80vh] bg-gray-50 dark:bg-gray-900 mt-16">
      <div className="container mx-auto max-w-5xl">
        <h1 className="text-4xl brand-font text-center mb-2 dark:text-white">My Wishlist</h1>
        <p className="text-center text-gray-500 dark:text-gray-400 mb-10">
          {items.length > 0 ? `${items.length} ${items.length === 1 ? 'item' : 'items'} saved` : 'Your wishlist is empty'}
        </p>

        {items.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {items.map(item => (
              <div key={item.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden border border-gray-100 dark:border-gray-700">
                <div className="h-64 bg-gray-100 dark:bg-gray-700 overflow-hidden p-4">
                  <img src={item.product_image} alt={item.product_name} className="w-full h-full object-contain hover:scale-105 transition duration-300" />
                </div>
                <div className="p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white brand-font mb-1">{item.product_name}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 capitalize mb-3">{item.product_category}</p>
                  <div className="flex gap-2">
                    <button onClick={() => moveToCart(item.id, item.product_name)} className="flex-1 bg-yellow-700 hover:bg-yellow-800 text-white py-2 rounded-lg text-sm font-semibold transition">
                      Add to Cart
                    </button>
                    <button onClick={() => removeWishlist(item.id)} className="px-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900 transition" title="Remove">
                      <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-20">
            <svg className="w-16 h-16 mx-auto text-gray-300 dark:text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
            </svg>
            <p className="text-gray-400 dark:text-gray-500 text-lg mb-4">No items in your wishlist yet</p>
            <Link to="/" className="inline-block bg-yellow-700 hover:bg-yellow-800 text-white px-8 py-3 rounded-lg font-semibold transition">Start Shopping</Link>
          </div>
        )}
      </div>
    </section>
  );
}