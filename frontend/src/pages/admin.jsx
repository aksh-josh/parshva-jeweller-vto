import React, { useState } from 'react';

export default function Admin() {
  // Mock Data - Will be replaced by your Flask API
  const categories = ['earrings', 'necklaces', 'rings', 'bangles', 'chains', 'mangalsutra'];
  
  const [products, setProducts] = useState([
    { id: 1, name: "Kundan Choker", category: "necklaces", material: "gold", price: 45000, weight: "45g", is_active: true, image_path: "/necklace/1.png" },
    { id: 2, name: "Diamond Solitaire", category: "rings", material: "diamond", price: 85000, weight: "5g", is_active: true, image_path: "/ring/1.png" },
    { id: 3, name: "Silver Jhumki", category: "earrings", material: "silver", price: 3500, weight: "12g", is_active: false, image_path: "/jhumka/1.png" }
  ]);

  const [filterCat, setFilterCat] = useState('all');
  const [filterMat, setFilterMat] = useState('all');
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Form State for New Product
  const [newProduct, setNewProduct] = useState({ name: '', category: categories[0], material: 'gold', price: '', weight: '', image: null });

  // --- Handlers ---
  const updateField = (id, field, value) => {
    setProducts(products.map(p => p.id === id ? { ...p, [field]: value } : p));
    // FUTURE: await fetch(`/api/admin/product/${id}`, { method: 'POST', body: JSON.stringify({ [field]: value }) });
  };

  const deleteProduct = (id) => {
    if (window.confirm('Delete this product?')) {
      setProducts(products.filter(p => p.id !== id));
      // FUTURE: await fetch(`/api/admin/product/${id}/delete`, { method: 'POST' });
    }
  };

  const handleAddSubmit = (e) => {
    e.preventDefault();
    alert(`Product ${newProduct.name} added! (Mock)`);
    setIsModalOpen(false);
    // FUTURE: Use FormData to send the actual image file to Flask
  };

  // --- Filtering Logic ---
  const filteredProducts = products.filter(p => {
    const matchCat = filterCat === 'all' || p.category === filterCat;
    const matchMat = filterMat === 'all' || p.material === filterMat;
    return matchCat && matchMat;
  });

  return (
    <section className="py-10 px-4 min-h-[85vh] bg-gray-50 dark:bg-gray-900 mt-16">
      <div className="container mx-auto max-w-7xl">
        
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl brand-font dark:text-white">Admin Panel</h1>
            <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">{filteredProducts.length} products displayed</p>
          </div>
          <button 
            onClick={() => setIsModalOpen(true)}
            className="bg-yellow-700 hover:bg-yellow-800 text-white px-6 py-2 rounded-lg font-semibold text-sm transition"
          >
            + Add Product
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button onClick={() => { setFilterCat('all'); setFilterMat('all'); }} className={`px-4 py-2 rounded-full text-sm font-semibold transition ${filterCat === 'all' && filterMat === 'all' ? 'bg-yellow-700 text-white' : 'bg-gray-200 text-gray-700'}`}>Reset Filters</button>
          <span className="mx-2 text-gray-300 dark:text-gray-600">|</span>
          
          {categories.map(cat => (
            <button key={cat} onClick={() => setFilterCat(cat)} className={`px-4 py-2 rounded-full text-sm font-semibold transition capitalize ${filterCat === cat ? 'bg-yellow-700 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}>{cat}</button>
          ))}
          
          <span className="mx-2 text-gray-300 dark:text-gray-600">|</span>
          
          {['gold', 'silver', 'diamond', 'antique'].map(mat => (
            <button key={mat} onClick={() => setFilterMat(mat)} className={`px-4 py-2 rounded-full text-sm font-semibold transition capitalize ${filterMat === mat ? 'bg-yellow-700 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}>{mat}</button>
          ))}
        </div>

        {/* Product Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {filteredProducts.map(p => (
            <div key={p.id} className="bg-white dark:bg-gray-800 rounded-xl shadow border border-gray-100 dark:border-gray-700 overflow-hidden">
              <div className="h-48 bg-gray-100 dark:bg-gray-700 relative">
                <img src={p.image_path} alt={p.name} className="w-full h-full object-cover" />
                <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-yellow-200 text-yellow-800">{p.material}</span>
                {!p.is_active && (
                  <span className="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-bold bg-red-200 text-red-700">HIDDEN</span>
                )}
              </div>
              <div className="p-3 space-y-2">
                <p className="text-xs text-gray-400 dark:text-gray-500 capitalize">{p.category}</p>
                <input 
                  type="text" 
                  value={p.name} 
                  onChange={(e) => updateField(p.id, 'name', e.target.value)}
                  className="w-full font-semibold text-sm text-gray-900 dark:text-white bg-transparent border-b border-transparent hover:border-gray-300 focus:border-yellow-600 outline-none transition"
                />
                <div className="flex items-center gap-1">
                  <span className="text-yellow-700 dark:text-yellow-500 font-bold">₹</span>
                  <input 
                    type="number" 
                    value={p.price} 
                    onChange={(e) => updateField(p.id, 'price', e.target.value)}
                    className="w-24 font-bold text-yellow-700 dark:text-yellow-500 bg-transparent border-b border-transparent hover:border-gray-300 focus:border-yellow-600 outline-none transition text-sm"
                  />
                </div>
                <div className="flex gap-2">
                  <select 
                    value={p.material} 
                    onChange={(e) => updateField(p.id, 'material', e.target.value)}
                    className="text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 text-gray-700 dark:text-gray-300 flex-1"
                  >
                    <option value="gold">Gold</option>
                    <option value="silver">Silver</option>
                    <option value="diamond">Diamond</option>
                    <option value="antique">Antique</option>
                  </select>
                  <input 
                    type="text" 
                    value={p.weight} 
                    onChange={(e) => updateField(p.id, 'weight', e.target.value)}
                    className="text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 rounded px-2 py-1 text-gray-700 dark:text-gray-300 w-20"
                  />
                </div>
                <div className="flex justify-between items-center pt-2">
                  <label className="flex items-center gap-1 text-xs text-gray-500 cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={p.is_active} 
                      onChange={(e) => updateField(p.id, 'is_active', e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    Visible
                  </label>
                  <button onClick={() => deleteProduct(p.id)} className="text-xs text-red-400 hover:text-red-600 transition">Delete</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Add Product Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-md p-6">
            <h2 className="text-xl font-bold dark:text-white mb-4">Add New Product</h2>
            <form onSubmit={handleAddSubmit} className="space-y-3">
              <input required value={newProduct.name} onChange={e => setNewProduct({...newProduct, name: e.target.value})} placeholder="Product name" className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white outline-none focus:border-yellow-600" />
              
              <select value={newProduct.category} onChange={e => setNewProduct({...newProduct, category: e.target.value})} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white">
                {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
              </select>
              
              <select value={newProduct.material} onChange={e => setNewProduct({...newProduct, material: e.target.value})} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white">
                <option value="gold">Gold</option>
                <option value="silver">Silver</option>
                <option value="diamond">Diamond</option>
                <option value="antique">Antique</option>
              </select>
              
              <input required type="number" value={newProduct.price} onChange={e => setNewProduct({...newProduct, price: e.target.value})} placeholder="Price (₹)" className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white outline-none focus:border-yellow-600" />
              <input value={newProduct.weight} onChange={e => setNewProduct({...newProduct, weight: e.target.value})} placeholder="Weight (e.g. 12.5g)" className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white outline-none focus:border-yellow-600" />
              <input required type="file" accept="image/*" className="w-full text-sm text-gray-500 dark:text-gray-400" />
              
              <div className="flex gap-3 pt-2">
                <button type="submit" className="flex-1 bg-yellow-700 hover:bg-yellow-800 text-white py-2 rounded-lg font-semibold transition">Add Product</button>
                <button type="button" onClick={() => setIsModalOpen(false)} className="px-4 py-2 border dark:border-gray-600 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  );
}