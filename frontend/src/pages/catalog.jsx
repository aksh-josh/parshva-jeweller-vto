import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';

export default function Catalog() {
  const { material, subcategory, collection } = useParams();
  const navigate = useNavigate();
  
  // ✨ FIX: State now holds full product objects, not just strings!
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);

  const activeCategory = subcategory || collection || 'necklaces';

  useEffect(() => {
    async function loadProducts() {
      setLoading(true);
      try {
        let targetCategory = activeCategory.toLowerCase();
        if (targetCategory.endsWith('s')) targetCategory = targetCategory.slice(0, -1);
        if (targetCategory.includes('earring')) targetCategory = 'jhumka';

        const response = await fetch(`/api/products?category=${targetCategory}`);
        const data = await response.json();
        
        // ✨ ADD THIS LINE:
        console.log("API Response:", data); 

        if (data.success) {
          setProducts(data.products);
        }
      } catch (error) {
        console.error("Error loading products:", error);
      } finally {
        setLoading(false);
      }
    }

    loadProducts();
  }, [activeCategory]);

  return (
    <section className="pt-24 pb-16 px-4 min-h-screen bg-gray-50">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl brand-font text-gray-900 capitalize mb-4">
            {material ? `${material} ${activeCategory}` : activeCategory.replace('-', ' ')}
          </h1>
          <p className="text-gray-500">Discover our exquisite handcrafted pieces</p>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-700"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
            
            {/* ✨ FIX: We now loop over the 'products' array and pull the dynamic name/category */}
            {products.map((product) => (
              <div key={product.id} className="bg-white rounded-2xl p-4 shadow-lg border border-gray-100 flex flex-col h-full group hover:shadow-xl transition">
                <div className="h-64 bg-gray-50 rounded-xl overflow-hidden mb-4 relative flex items-center justify-center">
                  
                  <img
                    src={`/static/${product.image_path}`}
                    alt={product.name}
                    className="w-3/4 h-3/4 object-contain group-hover:scale-110 transition duration-500"
                  />
                  
                  {/* Try On Button Overlay */}
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <button
                      onClick={() => navigate(`/tryon?category=${product.folder}&file=${product.image_path.split('/').pop()}`)}
                      className="bg-yellow-700 hover:bg-yellow-600 text-white px-6 py-2 rounded-full font-semibold transform translate-y-4 group-hover:translate-y-0 transition-all duration-300"
                    >
                      Virtual Try-On
                    </button>
                  </div>
                </div>

                <div className="flex flex-col flex-grow">
                  {/* Real Dynamic Name from Backend! */}
                  <h3 className="font-bold text-gray-900 brand-font text-lg leading-tight">{product.name}</h3>
                  <p className="text-xs text-gray-500 capitalize mt-1">{product.category.replace('-', ' ')}</p>

                  <div className="mt-auto pt-4 flex items-center justify-between">
                    <p className="text-yellow-700 font-bold text-lg">₹{product.price.toLocaleString('en-IN')}</p>
                    <button className="bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg text-xs font-semibold transition shadow-md">
                      Add to Cart
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && products.length === 0 && (
          <div className="text-center py-20">
            <p className="text-gray-500 text-lg">No products found in this category.</p>
          </div>
        )}
      </div>
    </section>
  );
}