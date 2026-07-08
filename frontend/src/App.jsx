import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Layout from './components/Layout';
import Home from './pages/Home';
import Catalog from './pages/Catalog';
import Auth from './pages/Auth';
import VirtualTryOn from './pages/VirtualTryOn';
import Cart from './pages/Cart';
import Wishlist from './pages/Wishlist';
import Profile from './pages/Profile'; // New Import
import Admin from './pages/Admin';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="login" element={<Auth />} />
          <Route path="tryon" element={<VirtualTryOn />} />
          
          <Route path="cart" element={<Cart />} />
          <Route path="wishlist" element={<Wishlist />} />
          <Route path="profile" element={<Profile />} /> {/* New Route */}
          <Route path="admin" element={<Admin />} />
          
          <Route path="shop/material/:material" element={<Catalog />} />
          <Route path="shop/:material/:subcategory" element={<Catalog />} />
          <Route path="shop/collection/:collection" element={<Catalog />} />
          <Route path="shop/wedding/:collection" element={<Catalog />} />
          <Route path="shop/gifting/:collection" element={<Catalog />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;