import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

export default function Home() {
  // --- Carousel State & Logic ---
  const [currentSlide, setCurrentSlide] = useState(0);
  const totalSlides = 3;

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % totalSlides);
    }, 7000);
    return () => clearInterval(timer);
  }, []);

  const nextSlide = () => setCurrentSlide((prev) => (prev + 1) % totalSlides);
  const prevSlide = () => setCurrentSlide((prev) => (prev - 1 + totalSlides) % totalSlides);

  // --- Scroll Reveal Animation Logic ---
  useEffect(() => {
    const revealEls = document.querySelectorAll(".reveal, .reveal-left, .reveal-right, .reveal-scale");
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("revealed");
      });
    }, { threshold: 0.1, rootMargin: "0px 0px -50px 0px" });
    
    revealEls.forEach((el) => revealObserver.observe(el));
    return () => revealObserver.disconnect();
  }, []);

  // --- Data Arrays (Replaces Jinja Blocks) ---
  const collections = [
    { slug: 'kundan-stories', label: 'Kundan Stories', desc: 'Intricate stone settings', img: 'https://blingbag.co.in/cdn/shop/files/EmeraldMansiKundanJewellerySet_1.jpg?v=1753870981&width=1080' },
    { slug: 'rajwadi-heritage', label: 'Rajwadi Heritage', desc: 'Royal Rajasthani tradition', img: 'https://ishhaara.com/cdn/shop/files/ishhaara-pink-kundan-rajwadi-pendent-jewellery-set-30509247594539.jpg?v=1710636673&width=1920' },
    { slug: 'polki-collection', label: 'Polki Collection', desc: 'Uncut diamond beauty', img: 'https://www.sanvijewels.com/cdn/shop/files/IMG_20240527_194828.jpg?v=1717754516&width=2569' },
    { slug: 'festive-collection', label: 'Festive Collection', desc: 'Celebration-ready pieces', img: 'https://ishhaara.com/cdn/shop/files/ishhaara-jadau-kundan-leaf-ear-chain-39720070935637.jpg?v=1767780078&width=533' }
  ];

  const weddingItems = [
    { slug: 'wedding-necklaces', label: 'Wedding Necklaces', img: 'https://images.unsplash.com/photo-1721807644561-9efcabee5c42?w=500&h=500&fit=crop' },
    { slug: 'wedding-bangles', label: 'Wedding Bangles', img: 'https://staticimg.tanishq.co.in/microsite/gold-page/assets/images/collection/51O5B1VOI2AP3.jpg' },
    { slug: 'wedding-earrings', label: 'Wedding Earrings', img: 'https://images.unsplash.com/photo-1589095053205-8fc842336f4a?w=500&h=500&fit=crop' },
    { slug: 'wedding-sets', label: 'Wedding Sets', img: 'https://images.unsplash.com/photo-1601121141461-9d6647bca1ed?w=500&h=500&fit=crop' },
    { slug: 'bridal-mangalsutra', label: 'Bridal Mangalsutra', img: 'https://sonchafa.com/cdn/shop/files/CA4E8182-E913-413E-90F5-382E22C540B5.jpg?v=1709899390&width=1946' }
  ];

  const gifts = [
    { slug: 'for-her', label: 'For Her', desc: 'Beautiful gifts for women', img: 'https://assets.ajio.com/medias/sys_master/root/20240628/rtI6/667ebcd61d763220fa6a472a/-473Wx593H-463282653-white-MODEL.jpg' },
    { slug: 'for-him', label: 'For Him', desc: 'Elegant pieces for men', img: 'https://starkle.in/cdn/shop/products/JPEGimage-A221EC61FE5C-1.jpg?v=1641142495&width=1897' },
    { slug: 'for-kids', label: 'For Kids', desc: 'Adorable little treasures', img: 'https://i.pinimg.com/236x/64/a1/03/64a1034fbd92590d1e0254d1c2e8c47a.jpg' }
  ];

  const promises = [
    { icon: 'M5 13l4 4L19 7', title: 'Certified Quality', desc: 'Every piece comes with a certificate of authenticity.' },
    { icon: 'M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z', title: 'Best Pricing', desc: 'Direct manufacturer pricing for the best value.' },
    { icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z', title: 'Artisan Crafted', desc: 'Handcrafted by skilled artisans preserving tradition.' },
    { icon: 'M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z', title: '24/7 Support', desc: 'Dedicated concierge service for all your needs.' }
  ];

  return (
    <>
      {/* Full-screen Hero Carousel with Ken Burns */}
      <section id="hero-section" className="relative h-screen w-full overflow-hidden mt-[-120px]">
        <div className="h-full w-full relative">
          
          {/* Slide 1 */}
          <div className={`slide absolute inset-0 transition-opacity duration-[2000ms] ${currentSlide === 0 ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
            <div className={`absolute inset-0 bg-cover bg-center ${currentSlide === 0 ? 'ken-burns' : ''}`} style={{ backgroundImage: "url('/backgrounds/slide_7.webp')" }}></div>
            <div className="absolute inset-0 bg-black/20"></div>
            <div className="absolute inset-0 flex flex-col justify-center items-center text-white text-center px-4 z-10">
              <h1 className="text-5xl md:text-7xl mb-4 brand-font font-bold">Timeless Elegance</h1>
              <p className="text-xl md:text-2xl mb-8 font-light tracking-wide">Discover our curated collections</p>
              <a href="#collections" className="bg-white hover:bg-yellow-600 text-gray-900 hover:text-white px-10 py-3.5 uppercase tracking-widest text-sm transition shadow-lg font-semibold">Explore Collection</a>
            </div>
          </div>

          {/* Slide 2 */}
          <div className={`slide absolute inset-0 transition-opacity duration-[2000ms] ${currentSlide === 1 ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
            <div className={`absolute inset-0 bg-cover bg-center ${currentSlide === 1 ? 'ken-burns' : ''}`} style={{ backgroundImage: "url('/backgrounds/slide_9.webp')" }}></div>
            <div className="absolute inset-0 bg-black/20"></div>
            <div className="absolute inset-0 flex flex-col justify-center items-center text-white text-center px-4 z-10">
              <h1 className="text-5xl md:text-7xl mb-4 brand-font font-bold">Golden Heritage</h1>
              <p className="text-xl md:text-2xl mb-8 font-light tracking-wide">Crafted for royalty</p>
              <Link to="/shop/material/gold" className="bg-white hover:bg-yellow-600 text-gray-900 hover:text-white px-10 py-3.5 uppercase tracking-widest text-sm transition shadow-lg font-semibold">Shop Gold</Link>
            </div>
          </div>

          {/* Slide 3 */}
          <div className={`slide absolute inset-0 transition-opacity duration-[2000ms] ${currentSlide === 2 ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
            <div className={`absolute inset-0 bg-cover bg-center ${currentSlide === 2 ? 'ken-burns' : ''}`} style={{ backgroundImage: "url('/backgrounds/slide_10.avif')" }}></div>
            <div className="absolute inset-0 bg-black/20"></div>
            <div className="absolute inset-0 flex flex-col justify-center items-center text-white text-center px-4 z-10">
              <h1 className="text-5xl md:text-7xl mb-4 brand-font font-bold">Wedding Season</h1>
              <p className="text-xl md:text-2xl mb-8 font-light tracking-wide">Shine on your big day</p>
              <a href="#wedding" className="bg-white hover:bg-yellow-600 text-gray-900 hover:text-white px-10 py-3.5 uppercase tracking-widest text-sm transition shadow-lg font-semibold">View Collection</a>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 animate-bounce">
          <svg className="w-6 h-6 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
        </div>

        {/* Arrows */}
        <button onClick={prevSlide} className="absolute left-6 top-1/2 -translate-y-1/2 z-20 w-12 h-12 rounded-full bg-white/20 hover:bg-white/40 backdrop-blur text-white flex items-center justify-center transition">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path></svg>
        </button>
        <button onClick={nextSlide} className="absolute right-6 top-1/2 -translate-y-1/2 z-20 w-12 h-12 rounded-full bg-white/20 hover:bg-white/40 backdrop-blur text-white flex items-center justify-center transition">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path></svg>
        </button>
      </section>

      {/* COLLECTIONS */}
      <section id="collections" className="py-24 bg-white dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 reveal">
            <p className="text-xs uppercase tracking-[0.3em] text-yellow-700 dark:text-yellow-500 mb-3 font-semibold">Curated For You</p>
            <h2 className="text-5xl brand-font text-gray-900 dark:text-white mb-5">Our Collections</h2>
            <div className="w-16 h-[2px] bg-yellow-600 mx-auto"></div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {collections.map((col, index) => (
              <Link key={col.slug} to={`/shop/collection/${col.slug}`} className="reveal-scale group relative h-[420px] overflow-hidden rounded-2xl shadow-lg cursor-pointer" style={{ transitionDelay: `${index * 0.1}s` }}>
                <img src={col.img} alt={col.label} className="w-full h-full object-cover group-hover:scale-110 transition duration-[800ms]" loading="lazy" />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent opacity-70 group-hover:opacity-90 transition duration-500"></div>
                <div className="absolute bottom-8 left-8 right-8 text-white">
                  <h3 className="text-2xl mb-2 brand-font">{col.label}</h3>
                  <p className="text-gray-300 text-sm flex items-center gap-2">{col.desc} <span className="group-hover:translate-x-1 transition">→</span></p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* WEDDING */}
      <section id="wedding" className="py-24 bg-[#faf8f5] dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 reveal">
            <p className="text-xs uppercase tracking-[0.3em] text-yellow-700 dark:text-yellow-500 mb-3 font-semibold">For Your Special Day</p>
            <h2 className="text-5xl brand-font text-gray-900 dark:text-white mb-5">Wedding Jewellery</h2>
            <div className="w-16 h-[2px] bg-yellow-600 mx-auto"></div>
          </div>

          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-7">
            {weddingItems.map((item, index) => (
              <Link key={item.slug} to={`/shop/wedding/${item.slug}`} className="reveal group text-center" style={{ transitionDelay: `${index * 0.08}s` }}>
                <div className="h-56 rounded-2xl overflow-hidden shadow-md group-hover:shadow-2xl transition-all duration-500 group-hover:-translate-y-2">
                  <img src={item.img} alt={item.label} className="w-full h-full object-cover group-hover:scale-110 transition duration-700" loading="lazy" />
                </div>
                <h3 className="mt-5 font-semibold text-gray-800 dark:text-white brand-font text-base">{item.label}</h3>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* GIFTING */}
      <section id="gifting" className="py-24 bg-white dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 reveal">
            <p className="text-xs uppercase tracking-[0.3em] text-yellow-700 dark:text-yellow-500 mb-3 font-semibold">Tokens of Love</p>
            <h2 className="text-5xl brand-font text-gray-900 dark:text-white mb-5">Gifting</h2>
            <div className="w-16 h-[2px] bg-yellow-600 mx-auto"></div>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {gifts.map((gift, index) => (
              <Link key={gift.slug} to={`/shop/gifting/${gift.slug}`} className="reveal-scale group relative h-[450px] overflow-hidden rounded-2xl shadow-lg cursor-pointer" style={{ transitionDelay: `${index * 0.15}s` }}>
                <img src={gift.img} alt={gift.label} className="w-full h-full object-cover group-hover:scale-110 transition duration-[800ms]" loading="lazy" />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition duration-500"></div>
                <div className="absolute bottom-10 left-0 right-0 text-center text-white">
                  <h3 className="text-3xl mb-2 brand-font">{gift.label}</h3>
                  <p className="text-gray-300 text-sm">{gift.desc} →</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ABOUT / WHY US */}
      <section id="about" className="py-24 bg-[#faf8f5] dark:bg-gray-800">
        <div className="container mx-auto px-6 text-center">
          <div className="reveal">
            <p className="text-xs uppercase tracking-[0.3em] text-yellow-700 dark:text-yellow-500 mb-3 font-semibold">Why Choose Us</p>
            <h2 className="text-4xl brand-font text-gray-900 dark:text-white mb-16">The Parshva Promise</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10">
            {promises.map((promise, index) => (
              <div key={index} className="reveal p-6" style={{ transitionDelay: `${index * 0.1}s` }}>
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-6 mx-auto bg-yellow-100/80 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-400">
                  <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d={promise.icon}></path>
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2 brand-font dark:text-white">{promise.title}</h3>
                <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed">{promise.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Newsletter */}
      <section className="bg-gray-900 dark:bg-black py-20">
        <div className="container mx-auto px-6 text-center text-white reveal">
          <h2 className="text-4xl brand-font mb-4">Join Our Exclusive Circle</h2>
          <p className="text-gray-500 mb-10 max-w-lg mx-auto">Be the first to know about new arrivals, limited editions, and special events.</p>
          <div className="max-w-md mx-auto flex">
            <input type="email" placeholder="Enter your email" className="flex-1 px-5 py-3.5 bg-gray-800 border border-gray-700 rounded-l focus:outline-none focus:border-yellow-600 text-white placeholder-gray-600" />
            <button className="bg-yellow-700 hover:bg-yellow-600 text-white px-8 py-3.5 font-semibold transition uppercase tracking-wider text-sm rounded-r">Subscribe</button>
          </div>
        </div>
      </section>
    </>
  );
}