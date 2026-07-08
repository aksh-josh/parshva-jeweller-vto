import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';

export default function VirtualTryOn() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  
  const category = searchParams.get('category') || 'necklace';
  const filename = searchParams.get('file') || '1.png';

  // --- UI State ---
  const [isRunning, setIsRunning] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [status, setStatus] = useState('Camera offline. Click Start to begin.');
  const [zoom, setZoom] = useState(1.0);
  
  const [activeItems, setActiveItems] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [primaryProductId, setPrimaryProductId] = useState(null);
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');

  // --- High-Performance Architectural Refs (Ported from tryon_live.html) ---
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const renderLoopRef = useRef(null);
  
  // The exact 480x360 network canvas from your original code
  const networkCanvasRef = useRef(document.createElement('canvas')); 
  
  const overlayImgRef = useRef(new Image()); 
  const overlayReadyRef = useRef(false);
  const pendingServerResponseRef = useRef(null);
  
  const lastProcessTimeRef = useRef(0);
  const isSendingRef = useRef(false);
  
  const activeItemsRef = useRef([]); 
  const isRunningRef = useRef(false);
  const zoomRef = useRef(1.0);
  const isStartingRef = useRef(false);
  const metadataListenerRef = useRef(null);

  // Keep refs synced with React state
  useEffect(() => { activeItemsRef.current = activeItems; }, [activeItems]);
  useEffect(() => { isRunningRef.current = isRunning; }, [isRunning]);

  // 1. INITIALIZE ITEM
  useEffect(() => {
    async function loadPrimaryItem() {
      try {
        let targetCategory = category.toLowerCase();
        if (targetCategory.endsWith('s')) targetCategory = targetCategory.slice(0, -1);
        if (targetCategory.includes('earring')) targetCategory = 'jhumka';
        
        const prodRes = await fetch(`/api/products?category=${targetCategory}`);
        const prodData = await prodRes.json();
        
        const filenameBase = filename.split('.')[0]; 
        const foundProduct = prodData.products?.find(p => 
          p.image_path.startsWith(`${targetCategory}/${filenameBase}.`)
        );
        
        if (foundProduct) {
          setPrimaryProductId(foundProduct.id);
          setActiveItems([{ 
            id: Number(foundProduct.id), 
            name: foundProduct.name, 
            category: foundProduct.category, 
            image: `/static/${foundProduct.image_path}`, 
            isPrimary: true 
          }]);
        }
      } catch (err) {
        console.error("Failed to load item:", err);
      }
    }
    loadPrimaryItem();
  }, [category, filename]);

  const fetchRecommendations = async (productId) => {
    if (!productId) return;
    try {
      const recRes = await fetch(`/api/recommendations/${productId}`);
      const recData = await recRes.json();
      if (recData.success) {
        setRecommendations(recData.recommendations);
      }
    } catch (err) {
      console.error("Rec error:", err);
    }
  };

  // Hard-release any existing camera stream and detach it from the video
  // element. Always call this before requesting a new stream — leaving a
  // stale track alive is a common reason a *second* getUserMedia() call
  // silently fails or grabs the wrong device.
  const releaseStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    if (metadataListenerRef.current && videoRef.current) {
      videoRef.current.removeEventListener('loadedmetadata', metadataListenerRef.current);
      metadataListenerRef.current = null;
    }
  };

  // ════════════════════════════════════════════════════════════════════════
  // CAMERA ENUMERATION — ported directly from tryon_live.html's
  // enumerateCameraDevices()/detectCameras(). Runs once on mount so the
  // dropdown is populated before the user ever clicks Start, exactly like
  // the original page did on DOMContentLoaded.
  //
  // Note: the original HTML also had a server-side /api/camera/list path
  // using cv2.VideoCapture on the machine running Flask. That only works
  // when Flask runs directly on the same PC as the webcam. Now that the
  // backend runs inside a Linux Docker container, it has no access to the
  // host's webcam hardware, so that path is intentionally NOT used here —
  // camera selection is 100% client-side (navigator.mediaDevices), which
  // is what actually worked in your original build for the browser camera.
  // ════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    async function detectCameras() {
      try {
        let tempStream = null;
        try {
          tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (e) {
          // Permission not granted yet — device labels will just be blank
          // until the user starts the camera for the first time.
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === 'videoinput');
        setVideoDevices(cams);

        if (tempStream) tempStream.getTracks().forEach(t => t.stop());
      } catch (e) {
        console.error('[Camera] Enumeration error:', e);
      }
    }
    detectCameras();
  }, []);

  // Refresh the device list if the user plugs/unplugs a camera mid-session.
  useEffect(() => {
    const handleDeviceChange = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setVideoDevices(devices.filter(d => d.kind === 'videoinput'));
      } catch (e) { /* ignore */ }
    };
    navigator.mediaDevices?.addEventListener?.('devicechange', handleDeviceChange);
    return () => navigator.mediaDevices?.removeEventListener?.('devicechange', handleDeviceChange);
  }, []);

  // ════════════════════════════════════════════════════════════════════════
  // CAMERA START — ported directly from tryon_live.html's startTryOn().
  // Default (no manual selection) uses facingMode:'user' with NO deviceId,
  // exactly like camIdx===0 in the original working version. A deviceId is
  // only forced when the user explicitly picks a specific camera from the
  // dropdown — this is deliberately simple and avoids second-guessing which
  // device the browser considers "the" front camera.
  // ════════════════════════════════════════════════════════════════════════
  const startCamera = async () => {
    if (isStartingRef.current) return; // ignore double-clicks / re-entrancy
    isStartingRef.current = true;

    try {
      setStatus('Requesting camera access...');
      releaseStream(); // make sure nothing stale is holding the device

      const videoConstraint = selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraint,
        audio: false
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Video element not ready');

      video.srcObject = stream;

      await new Promise((resolve, reject) => {
        const onLoaded = () => {
          video.removeEventListener('loadedmetadata', onLoaded);
          metadataListenerRef.current = null;
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
          video.play().then(resolve).catch(reject);
        };
        metadataListenerRef.current = onLoaded;
        video.addEventListener('loadedmetadata', onLoaded);
      });

      // Refresh device labels now that permission is definitely granted.
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setVideoDevices(devices.filter(d => d.kind === 'videoinput'));
      } catch (e) { /* non-fatal */ }

      networkCanvasRef.current.width = 480;
      networkCanvasRef.current.height = 360;

      setIsRunning(true);
      setIsScanning(true);
      setStatus('AI Face Mesh Scanning...');

      setTimeout(() => {
        setIsScanning(false);
        setStatus('AI Active & Tracking');
        fetchRecommendations(primaryProductId);
      }, 2000);

      lastProcessTimeRef.current = performance.now();
      startRenderLoop();

    } catch (err) {
      setStatus(`Camera error: ${err.message}. Try selecting a different camera.`);
      console.error('[Camera] Start error:', err);
    } finally {
      isStartingRef.current = false;
    }
  };

  const stopCamera = () => {
    setIsRunning(false);
    setIsScanning(false);
    releaseStream();
    if (renderLoopRef.current) cancelAnimationFrame(renderLoopRef.current);

    overlayImgRef.current.src = '';
    overlayReadyRef.current = false;
    pendingServerResponseRef.current = null;

    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setStatus('Camera stopped.');
    setRecommendations([]);
  };

  // ════════════════════════════════════════════════════════════════════════
  // THE MASTER RENDER LOOP (Exactly ported from tryon_live.html)
  // ════════════════════════════════════════════════════════════════════════
  const startRenderLoop = () => {
    const render = (timestamp) => {
      if (!isRunningRef.current || !videoRef.current || !canvasRef.current) return;
      
      const ctx = canvasRef.current.getContext('2d');
      const video = videoRef.current;

      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        // 1. Draw the correct frame to the canvas
        if (overlayReadyRef.current && activeItemsRef.current.length > 0 && !isScanning) {
          ctx.drawImage(overlayImgRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        } else {
          ctx.drawImage(video, 0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        // 2. Decode the background buffer if we received a response
        if (pendingServerResponseRef.current) {
          const src = pendingServerResponseRef.current;
          pendingServerResponseRef.current = null;
          
          overlayImgRef.current.onload = () => {
            overlayReadyRef.current = true;
          };
          overlayImgRef.current.onerror = () => {
            overlayReadyRef.current = false;
          };
          overlayImgRef.current.src = src;
        }
      }

      // 3. Trigger the network payload every 300ms
      if (timestamp - lastProcessTimeRef.current >= 300 && !isSendingRef.current && !isScanning) {
        lastProcessTimeRef.current = timestamp;
        processFrame();
      }

      renderLoopRef.current = requestAnimationFrame(render);
    };
    renderLoopRef.current = requestAnimationFrame(render);
  };

  // ════════════════════════════════════════════════════════════════════════
  // THE NETWORK PAYLOAD (Exactly ported from tryon_live.html)
  // ════════════════════════════════════════════════════════════════════════
  const processFrame = async () => {
    if (!videoRef.current || activeItemsRef.current.length === 0) return;
    
    isSendingRef.current = true;

    // Draw to 480x360 canvas and compress to 0.40 quality for lightning speed
    const sendCtx = networkCanvasRef.current.getContext('2d');
    sendCtx.drawImage(videoRef.current, 0, 0, 480, 360);
    const frameData = networkCanvasRef.current.toDataURL('image/jpeg', 0.40); 

    try {
      const ids = activeItemsRef.current.map(i => Number(i.id));

      const res = await fetch('/api/jewelry-tryon', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'process_frame',
          data: { 
            frame: frameData, 
            jewelry_id: ids[0],      // 🔥 Sent exactly as old HTML did!
            jewelry_ids: ids,        // 🔥 Array sent exactly as old HTML did!
            zoom_factor: zoomRef.current,
            camera_is_usb: false
          }
        })
      });
      
      const data = await res.json();
      
      if (data.processed_frame) {
        pendingServerResponseRef.current = data.processed_frame;
      }
      
    } catch (err) {
      console.log('Network skip');
    } finally {
      isSendingRef.current = false;
    }
  };

  useEffect(() => { return () => stopCamera(); }, []);

  const adjustZoom = (delta) => {
    const newZoom = Math.max(0.5, Math.min(2.0, Math.round((zoomRef.current + delta) * 10) / 10));
    setZoom(newZoom);
    zoomRef.current = newZoom;
  };
  
  const removeJewelry = (id) => setActiveItems(prev => prev.filter(item => item.id !== id || item.isPrimary));
  
  const addRecommendationToTryOn = (rec) => {
    if (activeItems.find(item => item.id === rec.id)) return;
    setActiveItems(prev => [...prev, {
      id: Number(rec.id), 
      name: rec.name, 
      category: rec.category,
      image: `/static/${rec.image_path}`, 
      isPrimary: false
    }]);
  };

  return (
    <div className="bg-[#0f172a] min-h-screen pt-24 pb-12 font-sans text-gray-100">
      <div className="container mx-auto px-4 max-w-7xl">
        
        <div className="flex justify-between items-center mb-6">
          <button onClick={() => navigate(-1)} className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-yellow-500 transition">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path></svg>
            Back to Studio
          </button>
          <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-md flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-xs font-semibold tracking-wider text-gray-300 uppercase">{status}</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-12 gap-8">
          
          {/* LEFT: Camera Viewport */}
          <div className="lg:col-span-8">
            <div className="rounded-3xl overflow-hidden shadow-2xl relative border border-white/10 bg-black">
              
              <div className="absolute top-4 left-4 z-20 flex flex-wrap gap-2">
                {activeItems.map(item => (
                  <div key={item.id} className="flex items-center gap-3 bg-black/40 backdrop-blur-md rounded-full pr-4 p-1 shadow-lg border border-white/10">
                    <img src={item.image} alt={item.name} className={`w-8 h-8 rounded-full object-cover border-2 ${item.isPrimary ? 'border-yellow-500' : 'border-blue-400'}`} />
                    <div>
                      <p className="text-white text-[11px] font-semibold">{item.name}</p>
                      <p className="text-gray-400 text-[9px] capitalize">{item.category}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="relative w-full bg-black" style={{ aspectRatio: '16/9' }}>
                <video ref={videoRef} playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-0" style={{ transform: 'scaleX(-1)' }} />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} />
                
                {isScanning && (
                  <div className="absolute inset-0 z-30 pointer-events-none flex flex-col items-center justify-center bg-black/20">
                    <div className="w-full h-1 bg-green-400 opacity-50 shadow-[0_0_15px_#4ade80] animate-[scan_2s_ease-in-out_infinite] absolute top-0"></div>
                    <svg className="w-24 h-24 text-green-400 animate-pulse opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M3 8v-2a2 2 0 0 1 2-2h2m10 0h2a2 2 0 0 1 2 2v2m0 10v2a2 2 0 0 1-2 2h-2m-10 0h-2a2 2 0 0 1-2-2v-2"></path></svg>
                    <p className="mt-4 font-mono text-green-400 text-sm tracking-widest uppercase animate-pulse shadow-black drop-shadow-md">Mapping Facial Topology...</p>
                  </div>
                )}

                {!isRunning && (
                  <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-gray-900/80 backdrop-blur-sm">
                    <div className="w-20 h-20 bg-yellow-500/20 rounded-full flex items-center justify-center mb-6">
                      <span className="text-4xl">{status.startsWith('Camera error') ? '⚠️' : '✨'}</span>
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">AR Fitting Room</h2>
                    {status.startsWith('Camera error') ? (
                      <p className="text-red-400 mb-8 max-w-sm text-center text-sm">{status}</p>
                    ) : (
                      <p className="text-gray-400 mb-8 max-w-sm text-center text-sm">Experience our jewelry in real-time. Allow camera access to begin the AI scan.</p>
                    )}
                    {videoDevices.length > 1 && (
                      <select
                        value={selectedDeviceId}
                        onChange={(e) => setSelectedDeviceId(e.target.value)}
                        className="mb-4 bg-black/40 border border-white/10 text-gray-300 text-xs rounded-lg px-3 py-2"
                      >
                        <option value="">Built-in webcam (default)</option>
                        {videoDevices.map((d, i) => (
                          <option key={d.deviceId} value={d.deviceId}>
                            {d.label || `Camera ${i + 1}`}
                          </option>
                        ))}
                      </select>
                    )}
                    <button onClick={startCamera} className="bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 text-white px-10 py-3.5 rounded-full font-bold shadow-[0_0_20px_rgba(234,179,8,0.3)] transition transform hover:-translate-y-1">
                      {status.startsWith('Camera error') ? 'Retry Camera' : 'Initialize Camera'}
                    </button>
                  </div>
                )}
              </div>

              <div className="absolute bottom-4 left-4 right-4 z-20 flex items-center justify-between bg-black/50 backdrop-blur-xl border border-white/10 rounded-2xl p-3">
                {isRunning ? (
                  <button onClick={stopCamera} className="bg-red-500/20 hover:bg-red-500/40 text-red-400 border border-red-500/30 px-6 py-2 rounded-xl text-sm font-semibold transition">End Session</button>
                ) : (
                  <div className="px-6 py-2"></div>
                )}
                
                <div className="flex items-center gap-3 bg-white/5 rounded-xl p-1 border border-white/10">
                  <button onClick={() => adjustZoom(-0.1)} className="w-8 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition">-</button>
                  <span className="text-xs font-mono w-10 text-center text-yellow-500">{zoom.toFixed(1)}x</span>
                  <button onClick={() => adjustZoom(0.1)} className="w-8 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition">+</button>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: AI Styling Engine */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white/5 border border-white/10 rounded-3xl overflow-hidden backdrop-blur-xl shadow-2xl flex flex-col h-[500px]">
              <div className="p-5 border-b border-white/10 bg-gradient-to-r from-yellow-500/10 to-transparent">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold text-white text-lg">AI Stylist</h3>
                    <p className="text-[11px] text-gray-400 mt-1">Multi-item layer testing</p>
                  </div>
                  <span className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-2 py-1 rounded text-[10px] font-bold tracking-widest">LIVE</span>
                </div>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                {!isRunning ? (
                  <div className="h-full flex flex-col items-center justify-center text-center opacity-50">
                    <svg className="w-12 h-12 mb-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
                    <p className="text-sm font-semibold">Engine Locked</p>
                    <p className="text-[11px] text-gray-400 mt-1">Start camera to generate matches</p>
                  </div>
                ) : isScanning ? (
                  <div className="h-full flex flex-col items-center justify-center">
                    <div className="w-6 h-6 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin mb-3"></div>
                    <p className="text-xs text-yellow-500 animate-pulse">Running Neural Match...</p>
                  </div>
                ) : recommendations.length === 0 ? (
                  <p className="text-xs text-gray-500 text-center py-10">No matches found.</p>
                ) : (
                  recommendations.map(rec => (
                    <div key={rec.id} className="group relative flex items-center gap-3 p-2.5 rounded-2xl bg-white/5 hover:bg-white/10 border border-transparent hover:border-white/10 transition cursor-pointer">
                      <div className="w-16 h-16 rounded-xl overflow-hidden bg-gray-800 flex-shrink-0">
                        <img src={`/static/${rec.image_path}`} alt={rec.name} className="w-full h-full object-cover group-hover:scale-110 transition duration-500" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-gray-200 truncate">{rec.name}</p>
                        <p className="text-[10px] text-gray-500 capitalize">{rec.category}</p>
                        <div className="mt-1.5 flex items-center gap-1.5">
                          <div className="w-full bg-gray-700 h-1 rounded-full overflow-hidden">
                            <div className="bg-green-400 h-full" style={{ width: `${Math.round((rec.score || rec.similarity || 0) * 100)}%` }}></div>
                          </div>
                          <span className="text-[9px] text-green-400 font-mono">{Math.round((rec.score || rec.similarity || 0) * 100)}%</span>
                        </div>
                      </div>
                      
                      <button 
                        onClick={() => addRecommendationToTryOn(rec)}
                        className={`absolute right-3 w-8 h-8 rounded-full flex items-center justify-center transition shadow-lg ${activeItems.find(i => i.id === Number(rec.id)) ? 'bg-green-500 text-white' : 'bg-white/10 hover:bg-yellow-500 text-white'}`}
                      >
                        {activeItems.find(i => i.id === Number(rec.id)) ? '✓' : '+'}
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-3xl p-5 backdrop-blur-xl shadow-lg">
              <h3 className="font-semibold text-sm text-gray-300 mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                Active Layers
              </h3>
              <div className="space-y-2">
                {activeItems.map(item => (
                  <div key={item.id} className="flex items-center justify-between p-2 rounded-xl bg-black/40 border border-white/5">
                    <div className="flex items-center gap-3">
                      <img src={item.image} alt={item.name} className="w-10 h-10 rounded-lg object-cover" />
                      <div>
                        <p className="text-xs font-semibold text-gray-200">{item.name}</p>
                        <p className="text-[9px] text-gray-500 capitalize">{item.category}</p>
                      </div>
                    </div>
                    {!item.isPrimary && (
                      <button onClick={() => removeJewelry(item.id)} className="w-7 h-7 flex items-center justify-center rounded-full hover:bg-red-500/20 text-red-400 transition">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>

      <style>{`
        @keyframes scan {
          0% { top: 0; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.02); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }
      `}</style>
    </div>
  );
}