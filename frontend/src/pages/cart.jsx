import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

export default function Cart() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);

  // In a real scenario, you'd fetch this from Flask: fetch('/api/cart')
  // For now, it stays dynamic and empty until the backend is fully wired for the cart.

  const updateQty = (id, newQty) => {
    if (newQty <= 0) { removeCart(id); return; }
    // FUTURE: await fetch('/api/cart/update', { ... })
    setItems(items.map(item => item.id === id ? { ...item, quantity: newQty, subtotal: item.price * newQty } : item));
  };

  const removeCart = (id) => {
    // FUTURE: await fetch('/api/cart/remove', { ... })
    setItems(items.filter(item => item.id !== id));
  };

  return (
    <section className="py-16 px-4 min-h-[80vh] bg-gray-50 dark:bg-gray-900 mt-16">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-4xl brand-font text-center mb-2 dark:text-white">Shopping Cart</h1>
        <p className="text-center text-gray-500 dark:text-gray-400 mb-10">
          {items.length} {items.length === 1 ? 'item' : 'items'}
        </p>

        {items.length > 0 ? (
          <>
            <div className="space-y-4">
              {items.map(item => (
                <div key={item.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-md border border-gray-100 dark:border-gray-700 flex overflow-hidden">
                  <div className="w-32 h-32 sm:w-40 sm:h-40 flex-shrink-0 bg-gray-100 dark:bg-gray-700">
                    <img src={item.product_image} alt={item.product_name} className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1 p-4 flex flex-col justify-between">
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white brand-font text-lg">{item.product_name}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 capitalize">{item.product_category}</p>
                      
                      {/* ✨ FIX: Price on Request Logic */}
                      {item.price > 0 ? (
                        <p className="text-yellow-700 dark:text-yellow-500 font-bold mt-1">₹{item.price.toLocaleString('en-IN')}</p>
                      ) : (
                        <p className="text-gray-400 dark:text-gray-500 text-sm mt-1 font-semibold">Price on request</p>
                      )}
                    </div>
                    
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex items-center gap-2">
                        <button onClick={() => updateQty(item.id, item.quantity - 1)} className="w-8 h-8 flex items-center justify-center rounded-full border hover:bg-gray-100 transition">−</button>
                        <span className="w-8 text-center font-semibold text-gray-800 dark:text-white">{item.quantity}</span>
                        <button onClick={() => updateQty(item.id, item.quantity + 1)} className="w-8 h-8 flex items-center justify-center rounded-full border hover:bg-gray-100 transition">+</button>
                      </div>
                      
                      {item.subtotal > 0 && (
                        <p className="font-bold text-gray-800 dark:text-white">₹{item.subtotal.toLocaleString('en-IN')}</p>
                      )}
                      
                      <button onClick={() => removeCart(item.id)} className="text-red-500 hover:text-red-700 text-sm font-semibold transition">Remove</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Order Summary */}
            <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-md p-6">
              <h2 className="text-lg font-bold dark:text-white mb-4">Order Summary</h2>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between text-gray-600 dark:text-gray-400">
                  <span>Subtotal</span>
                  <span>{total > 0 ? `₹${total.toLocaleString('en-IN')}` : 'Price on request'}</span>
                </div>
                <div className="flex justify-between text-gray-600 dark:text-gray-400">
                  <span>Shipping</span><span className="text-green-600">Free</span>
                </div>
                <hr className="dark:border-gray-700 my-2" />
                <div className="flex justify-between text-lg font-bold dark:text-white pt-2">
                  <span>Total</span>
                  <span className="text-yellow-700">{total > 0 ? `₹${total.toLocaleString('en-IN')}` : 'Price on request'}</span>
                </div>
              </div>
              <button className="w-full mt-6 bg-yellow-700 hover:bg-yellow-800 text-white py-3 rounded-lg font-semibold transition text-lg">
                Proceed to Checkout
              </button>
            </div>
          </>
        ) : (
          <div className="text-center py-20">
            <svg className="w-16 h-16 mx-auto text-gray-300 dark:text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"></path></svg>
            <p className="text-gray-400 text-lg mb-4">Your cart is empty</p>
            <Link to="/" className="inline-block bg-yellow-700 text-white px-8 py-3 rounded-lg font-semibold transition">Start Shopping</Link>
          </div>
        )}
      </div>
    </section>
  );
}