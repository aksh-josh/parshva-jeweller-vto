// static/js/virtual_tryon.js
(() => {
  // Elements
  const vtModal = document.getElementById('vt-modal');
  const vtVideo = document.getElementById('vt-video');
  const vtProcessed = document.getElementById('vt-processed');
  const vtStatus = document.getElementById('vt-status');
  const vtPosition = document.getElementById('vt-position');
  const vtPersonDetected = document.getElementById('vt-person-detected');
  const vtSelectedName = document.getElementById('vt-selected-name');
  const vtCloseBtn = document.getElementById('vt-close');
  const vtToggleBtn = document.getElementById('vt-toggle-camera');
  const vtFPSInput = document.getElementById('vt-fps');
  const vtSnapBtn = document.getElementById('vt-snap');
  const vtStopSessionBtn = document.getElementById('vt-stop-session');
  const vtLast = document.getElementById('vt-last');

  // State
  let stream = null;
  let captureIntervalId = null;
  let sessionId = null;
  let selectedJewelryId = null;
  let selectedJewelryName = null;
  let imagePathToId = {}; // mapping filled from server
  let lastProcessedDataUrl = null;

  // Fetch mapping from server
  async function fetchJewelryCategories() {
    try {
      const res = await fetch('/api/jewelry/categories');
      const j = await res.json();
      if (j.status === 'success') {
        const categories = j.categories;
        Object.keys(categories).forEach(type => {
          categories[type].forEach(item => {
            imagePathToId[item.image_path] = item.id;
          });
        });
      } else {
        console.warn('Failed to load categories mapping:', j);
      }
    } catch (e) {
      console.warn('Error fetching categories:', e);
    }
  }

  // Utility: find <img> sibling in product card and get data-image-path
  function getImagePathFromTrigger(trigger) {
    // trigger is the <a> or button clicked
    // look for closest parent .group then find img.gallery-image inside
    const parent = trigger.closest('.group');
    if (!parent) {
      // fallback: data-image-path on trigger
      return trigger.getAttribute('data-image-path') || null;
    }
    const img = parent.querySelector('.gallery-image');
    if (img) return img.getAttribute('data-image-path') || img.src.split('/static/').pop();
    return null;
  }

  // Start session on server
  async function startTryOnServer(jId, jName) {
    try {
      const resp = await fetch('/api/jewelry-tryon', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          action: 'start_tryon',
          data: {
            jewelry_id: jId,
            jewelry_name: jName || ''
          }
        })
      });
      const data = await resp.json();
      if (data.status === 'success') {
        sessionId = data.session_id;
        console.log('session started', sessionId);
        return true;
      } else {
        console.error('start_tryon failed', data);
        return false;
      }
    } catch (e) {
      console.error('start_tryon error', e);
      return false;
    }
  }

  // Stop session on server
  async function stopTryOnServer() {
    if (!sessionId) return;
    try {
      await fetch('/api/jewelry-tryon', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          action: 'stop_tryon',
          data: { session_id: sessionId }
        })
      });
    } catch (e) {
      console.warn('Error stopping session on server', e);
    } finally {
      sessionId = null;
    }
  }

  // Process one frame: send to server and update processed image
  async function processFrame(frameBase64) {
    if (!selectedJewelryId) return;
    try {
      const resp = await fetch('/api/jewelry-tryon', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          action: 'process_frame',
          data: {
            frame: frameBase64,
            jewelry_id: selectedJewelryId,
            session_id: sessionId
          }
        })
      });
      const data = await resp.json();
      if (data.processed_frame) {
        vtProcessed.src = data.processed_frame;
        lastProcessedDataUrl = data.processed_frame;
      }
      vtPosition.innerText = `Position: ${data.position_status || '—'}`;
      vtPersonDetected.innerText = `Person: ${data.person_detected ? 'Yes' : 'No'}`;
      vtLast.innerText = `Last: ${new Date().toLocaleTimeString()}`;
      vtStatus.innerText = 'Frame processed';
    } catch (e) {
      console.warn('processFrame error', e);
      vtStatus.innerText = 'Error processing frame';
    }
  }

  // Capture current frame from video as base64 jpeg
  function captureFrameBase64() {
    if (!vtVideo || vtVideo.readyState < 2) return null;
    const canvas = document.createElement('canvas');
    canvas.width = vtVideo.videoWidth;
    canvas.height = vtVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(vtVideo, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.75);
  }

  // Start camera and processing loop
  async function startCamera(fps = 6) {
    if (stream) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      vtVideo.srcObject = stream;
      await vtVideo.play();
      vtStatus.innerText = 'Camera started';

      // start session with server if not started
      if (!sessionId) {
        const ok = await startTryOnServer(selectedJewelryId, selectedJewelryName);
        if (!ok) {
          vtStatus.innerText = 'Failed to start session on server';
          stopCamera();
          return;
        }
      }

      const intervalMs = Math.max(50, Math.round(1000 / fps));
      captureIntervalId = setInterval(async () => {
        const frame = captureFrameBase64();
        if (frame) {
          await processFrame(frame);
        }
      }, intervalMs);

      vtToggleBtn.innerText = 'Stop Camera';
    } catch (e) {
      console.error('startCamera error', e);
      vtStatus.innerText = 'Camera error: ' + (e.message || e);
      stopCamera();
    }
  }

  function stopCamera() {
    if (captureIntervalId) {
      clearInterval(captureIntervalId);
      captureIntervalId = null;
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (vtVideo) {
      vtVideo.pause();
      vtVideo.srcObject = null;
    }
    vtToggleBtn.innerText = 'Start Camera';
    vtStatus.innerText = 'Camera stopped';
  }

  function snapshotDownload() {
    const data = lastProcessedDataUrl;
    if (!data) {
      alert('No processed image yet.');
      return;
    }
    const a = document.createElement('a');
    a.href = data;
    a.download = `${selectedJewelryName || 'tryon'}_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // Open try-on modal when "Virtual Try-On" button clicked
  window.openTryOn = async function (event, triggerEl) {
    event.preventDefault();
    // ensure mapping loaded
    await fetchJewelryCategories();

    const imagePath = getImagePathFromTrigger(triggerEl);
    if (imagePath && imagePathToId[imagePath]) {
      selectedJewelryId = imagePathToId[imagePath];
      selectedJewelryName = imagePath.split('/').pop().split('.')[0].replace('_',' ');
      vtSelectedName.innerText = `${selectedJewelryName} (id:${selectedJewelryId})`;
    } else {
      // fallback: try attribute on trigger
      const fallback = triggerEl.getAttribute('data-image-path') || null;
      if (fallback && imagePathToId[fallback]) {
        selectedJewelryId = imagePathToId[fallback];
        selectedJewelryName = fallback.split('/').pop().split('.')[0].replace('_',' ');
        vtSelectedName.innerText = `${selectedJewelryName} (id:${selectedJewelryId})`;
      } else {
        // Not found: alert and allow user to continue but warn
        selectedJewelryId = null;
        selectedJewelryName = fallback || 'Unknown';
        vtSelectedName.innerText = `${selectedJewelryName} (no id)`;
        vtStatus.innerText = 'Warning: item id not found. Try-on may not apply overlays.';
      }
    }

    // show modal
    vtModal.classList.add('show');
    vtModal.setAttribute('aria-hidden', 'false');
    // reset processed image
    vtProcessed.src = '';
    vtPosition.innerText = 'Position: —';
    vtPersonDetected.innerText = 'Person: —';
    vtLast.innerText = '—';
    lastProcessedDataUrl = null;
  };

  // Close & cleanup
  async function closeModal() {
    stopCamera();
    await stopTryOnServer();
    vtModal.classList.remove('show');
    vtModal.setAttribute('aria-hidden', 'true');
    vtSelectedName.innerText = '—';
    selectedJewelryId = null;
    selectedJewelryName = null;
    vtProcessed.src = '';
    vtStatus.innerText = 'Camera stopped';
  }

  // Helpers
  function getImagePathFromTrigger(trigger) {
    const parent = trigger.closest('.group');
    if (!parent) {
      return trigger.getAttribute('data-image-path') || null;
    }
    const img = parent.querySelector('.gallery-image');
    if (img) return img.getAttribute('data-image-path') || null;
    return null;
  }

  // Bind UI events
  vtToggleBtn.addEventListener('click', () => {
    if (!stream) {
      const fps = parseInt(vtFPSInput.value || '6', 10);
      if (!selectedJewelryId) {
        if (!confirm('This item does not have a mapped id. Continue anyway?')) return;
      }
      startCamera(fps);
    } else {
      stopCamera();
    }
  });

  vtCloseBtn.addEventListener('click', closeModal);
  vtSnapBtn.addEventListener('click', snapshotDownload);
  vtStopSessionBtn.addEventListener('click', async () => {
    await stopTryOnServer();
    vtStatus.innerText = 'Session stopped';
  });

  // When user presses Esc, close modal
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && vtModal.classList.contains('show')) {
      closeModal();
    }
  });

  // Preload mapping on script load
  fetchJewelryCategories();

})();
