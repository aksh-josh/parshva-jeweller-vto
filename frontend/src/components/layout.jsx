import React, { useState, useEffect } from 'react';
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';

export default function Layout() {
  const [isDark, setIsDark] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [counts, setCounts] = useState({ cart: 0, wishlist: 0 });
  
  // Mock Auth State (You will wire this to your Flask API later)
  const isAuthenticated = true;
  const user = { firstName: "Admin", fullName: "Admin User", phone: "123-456-7890", isAdmin: true };

  const navigate = useNavigate();
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  // Navigation Data Arrays (Replaces Jinja Loops)
  const navMaterials = [
    { key: 'gold', label: 'Gold', subcats: ['earrings','necklaces','mangalsutra','bangles','rings','chains'] },
    { key: 'silver', label: 'Silver', subcats: ['earrings','necklaces','bangles','rings','chains'] },
    { key: 'diamond', label: 'Diamond', subcats: ['earrings','necklaces','mangalsutra','bangles','rings','pendants'] },
    { key: 'daily-wear', label: 'Daily Wear', subcats: ['earrings','necklaces','mangalsutra','bangles','rings','chains'] }
  ];

  const collections = [
    { slug: 'kundan-stories', label: 'Kundan Stories' },
    { slug: 'rajwadi-heritage', label: 'Rajwadi Heritage' },
    { slug: 'polki-collection', label: 'Polki Collection' },
    { slug: 'festive-collection', label: 'Festive Collection' }
  ];

  // Handlers
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 80);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const toggleTheme = () => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.remove("dark");
      setIsDark(false);
    } else {
      root.classList.add("dark");
      setIsDark(true);
    }
  };

  const handleLogout = (e) => {
    e.preventDefault();
    // In the future, this will be: await fetch('/api/auth/logout')
    navigate('/login');
  };

  // Determine Navbar style based on scroll and page
  // Change this section:
  const headerClass = "nav-solid shadow-md fixed w-full top-0 z-50 transition-all duration-500";
  const headerStyle = { marginTop: '0' };

  return (
    <div className="bg-[#fdfbf7] dark:bg-gray-900 text-gray-800 dark:text-gray-200 transition-colors duration-300 min-h-screen flex flex-col">
      
      {/* Top Bar */}
      <div className="bg-gray-900 dark:bg-black text-gray-400 text-[11px] py-1.5 text-center tracking-widest uppercase z-50 relative">
        Free Shipping on Orders Above ₹999 &nbsp;·&nbsp; 100% Certified Jewellery &nbsp;·&nbsp; Lifetime Exchange
      </div>

      {/* Header */}
      <header className={headerClass} style={headerStyle}>
        <div className="container mx-auto px-6 pt-4 pb-2 flex justify-between items-center">
          <Link to="/" className="nav-brand text-4xl font-bold brand-font tracking-wider transition-colors duration-300">
            Parshva Jewellers
          </Link>

          {/* Search Bar */}
          <form onSubmit={(e) => { e.preventDefault(); navigate('/search'); }} className="hidden md:flex items-center flex-1 max-w-md mx-8">
            <div className="relative w-full">
              <input type="text" name="q" placeholder="Search jewellery..." autoComplete="off"
                className="w-full pl-10 pr-4 py-2 rounded-full bg-white/10 backdrop-blur border border-white/20 text-white placeholder-white/50 text-sm outline-none focus:bg-white/20 focus:border-white/40 transition nav-search-input" />
              <svg className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
          </form>

          {/* Icons Right */}
          <div className="flex items-center space-x-6">
            <button onClick={toggleTheme} className="nav-icon theme-toggle transition-colors" title="Toggle theme">
              {isDark ? (
                <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
                </svg>
              )}
            </button>

            <Link to="/wishlist" className="nav-icon relative hover:text-yellow-500 transition-colors">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
              </svg>
              <span className="absolute -top-2 -right-2 bg-yellow-700 text-white text-[10px] font-bold w-4 h-4 rounded-full flex items-center justify-center">{counts.wishlist}</span>
            </Link>

            <Link to="/cart" className="nav-icon relative hover:text-yellow-500 transition-colors">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"></path>
              </svg>
              <span className="absolute -top-2 -right-2 bg-yellow-700 text-white text-[10px] font-bold w-4 h-4 rounded-full flex items-center justify-center">{counts.cart}</span>
            </Link>

            {/* Profile Dropdown */}
            <div className="relative group cursor-pointer py-2">
              <div className="nav-icon flex items-center gap-1 hover:text-yellow-500 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                </svg>
                {isAuthenticated && <span className="text-xs font-medium hidden lg:inline">{user.firstName}</span>}
              </div>

              <ul className="absolute hidden group-hover:block bg-white dark:bg-gray-800 shadow-xl mt-2 w-52 py-2 rounded-lg border border-gray-100 dark:border-gray-700 right-0 z-50">
                {isAuthenticated ? (
                  <>
                    <li className="px-4 py-2 border-b border-gray-50 dark:border-gray-700">
                      <p className="text-sm font-semibold text-gray-800 dark:text-white">{user.fullName}</p>
                      <p className="text-xs text-gray-400">{user.phone}</p>
                    </li>
                    <li><Link to="/profile" className="block px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-yellow-50 dark:hover:bg-gray-700">My Profile</Link></li>
                    {user.isAdmin && (
                      <li><Link to="/admin" className="block px-4 py-2 text-sm text-yellow-700 dark:text-yellow-500 hover:bg-yellow-50 dark:hover:bg-gray-700 font-semibold">Admin Panel</Link></li>
                    )}
                    <li className="border-t border-gray-100 dark:border-gray-700 mt-1 pt-1">
                      <button onClick={handleLogout} className="w-full text-left block px-4 py-2 text-sm text-red-500 hover:bg-red-50 dark:hover:bg-gray-700 font-semibold transition-colors">Sign Out</button>
                    </li>
                  </>
                ) : (
                  <li><Link to="/login" className="block px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-yellow-50 dark:hover:bg-gray-700">Sign In / Register</Link></li>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Mega Menu Links */}
        <nav className="hidden md:block border-t border-white/10">
          <div className="container mx-auto px-6">
            <ul className="flex items-center justify-center space-x-1 text-[11px] font-semibold tracking-[0.2em] uppercase">
              {navMaterials.map((mat) => (
                <li key={mat.key} className="nav-item relative py-3 px-4">
                  <Link to={`/shop/material/${mat.key}`} className="nav-link text-gray-700 dark:text-gray-300 hover:text-yellow-600 dark:hover:text-yellow-500 transition">{mat.label}</Link>
                  <div className="mega-dropdown absolute left-0 top-full bg-white dark:bg-gray-800 shadow-xl rounded-b-lg border border-gray-100 dark:border-gray-700 py-4 px-6 min-w-[200px] z-50">
                    <ul className="space-y-2">
                      {mat.subcats.map((sub) => (
                        <li key={sub}><Link to={`/shop/${mat.key}/${sub}`} className="block text-sm text-gray-600 dark:text-gray-400 hover:text-yellow-700 dark:hover:text-yellow-500 transition capitalize tracking-normal font-normal">{sub}</Link></li>
                      ))}
                    </ul>
                  </div>
                </li>
              ))}

              <li className="nav-divider text-gray-300 dark:text-gray-700 select-none">|</li>

              <li className="nav-item relative py-3 px-4">
                <span className="nav-link text-gray-700 dark:text-gray-300 hover:text-yellow-600 dark:hover:text-yellow-500 transition cursor-pointer">Collections</span>
                <div className="mega-dropdown absolute left-0 top-full bg-white dark:bg-gray-800 shadow-xl rounded-b-lg border border-gray-100 dark:border-gray-700 py-4 px-6 min-w-[220px] z-50">
                  <ul className="space-y-2">
                    {collections.map((col) => (
                      <li key={col.slug}><Link to={`/shop/collection/${col.slug}`} className="block text-sm text-gray-600 dark:text-gray-400 hover:text-yellow-700 dark:hover:text-yellow-500 transition tracking-normal font-normal">{col.label}</Link></li>
                    ))}
                  </ul>
                </div>
              </li>
            </ul>
          </div>
        </nav>
      </header>

      {/* Main Content Replaces Jinja block content */}
      <main className="flex-grow pt-[120px]">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 dark:bg-black text-white py-16 mt-auto">
        <div className="container mx-auto px-6 grid md:grid-cols-4 gap-10 text-sm">
          <div>
            <h3 className="brand-font text-3xl mb-4 brand-shimmer">Parshva Jewellers</h3>
            <p className="text-gray-500 leading-relaxed">Timeless elegance crafted for the modern soul. Every piece tells a story of heritage and artistry.</p>
          </div>
          <div>
            <h4 className="font-bold mb-4 uppercase tracking-[0.15em] text-gray-300 text-xs">Shop</h4>
            <ul className="space-y-2.5 text-gray-500">
              <li><Link to="/shop/material/gold" className="hover:text-yellow-500 transition">Gold</Link></li>
              <li><Link to="/shop/material/silver" className="hover:text-yellow-500 transition">Silver</Link></li>
              <li><Link to="/shop/material/diamond" className="hover:text-yellow-500 transition">Diamond</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 uppercase tracking-[0.15em] text-gray-300 text-xs">Help</h4>
            <ul className="space-y-2.5 text-gray-500">
              <li><Link to="/faq" className="hover:text-yellow-500 transition">FAQ</Link></li>
              <li><Link to="/contact" className="hover:text-yellow-500 transition">Contact Us</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 uppercase tracking-[0.15em] text-gray-300 text-xs">Stay Connected</h4>
            <div className="flex border border-gray-700 rounded overflow-hidden mt-3">
              <input type="email" placeholder="Your email" className="bg-transparent w-full px-3 py-2 outline-none text-white placeholder-gray-600 text-sm"/>
              <button className="bg-yellow-700 hover:bg-yellow-600 px-4 text-white text-xs font-bold uppercase tracking-wider transition whitespace-nowrap">Join</button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}