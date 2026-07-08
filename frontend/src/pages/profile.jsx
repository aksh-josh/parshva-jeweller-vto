import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

export default function Profile() {
  const navigate = useNavigate();

  // Mock User Data - In the future, this will be fetched from your Flask API via useEffect
  const [user, setUser] = useState({
    fullName: "Akshat Patel",
    phone: "+91 9876543210",
    email: "akshat@example.com",
    createdAt: "February 2026",
    lastLogin: "06 Jul 2026, 11:41 PM"
  });

  const handleLogout = (e) => {
    e.preventDefault();
    // FUTURE: await fetch('/api/auth/logout', { method: 'POST' });
    navigate('/login');
  };

  return (
    <section className="min-h-[85vh] py-16 px-4 bg-gray-50 dark:bg-gray-900 mt-16">
      <div className="container mx-auto max-w-2xl">
        <h1 className="text-4xl brand-font text-gray-900 dark:text-white mb-8 text-center">
          My Profile
        </h1>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-8 border border-gray-100 dark:border-gray-700">
          
          {/* Avatar */}
          <div className="flex items-center gap-6 mb-8 pb-8 border-b border-gray-100 dark:border-gray-700">
            <div className="w-20 h-20 bg-yellow-100 dark:bg-yellow-900 rounded-full flex items-center justify-center">
              <span className="text-3xl brand-font text-yellow-700 dark:text-yellow-400">
                {user.fullName.charAt(0).toUpperCase()}
              </span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                {user.fullName}
              </h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm">
                Member since {user.createdAt}
              </p>
            </div>
          </div>

          {/* Details */}
          <div className="space-y-5">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Phone
                </p>
                <p className="text-gray-800 dark:text-white mt-1">{user.phone}</p>
              </div>
              <span className="bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-400 px-3 py-1 rounded-full text-xs font-semibold">
                Verified
              </span>
            </div>

            {user.email && (
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Email
                </p>
                <p className="text-gray-800 dark:text-white mt-1">{user.email}</p>
              </div>
            )}

            {user.lastLogin && (
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Last Login
                </p>
                <p className="text-gray-800 dark:text-white mt-1">
                  {user.lastLogin}
                </p>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="mt-8 pt-6 border-t border-gray-100 dark:border-gray-700 flex gap-4">
            <Link
              to="/"
              className="flex-1 text-center bg-yellow-700 hover:bg-yellow-800 text-white py-3 rounded-lg font-semibold transition"
            >
              Continue Shopping
            </Link>
            <button
              onClick={handleLogout}
              className="px-6 text-center border border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 py-3 rounded-lg font-semibold transition cursor-pointer"
            >
              Logout
            </button>
          </div>
          
        </div>
      </div>
    </section>
  );
}