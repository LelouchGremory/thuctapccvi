document.addEventListener('DOMContentLoaded', () => {
    const cameraToggleBtn = document.getElementById('cameraToggleBtn');
    const mirrorToggleBtn = document.getElementById('mirrorToggleBtn');
    const streamCanvas = document.getElementById('streamCanvas');
    const webcamVideo = document.getElementById('webcamVideo');
    const connectionOverlay = document.getElementById('connectionOverlay');
    const statusMessage = document.getElementById('statusMessage');
    const activeComboBadge = document.getElementById('activeComboBadge');
    
    const fpsVal = document.getElementById('fpsVal');
    const facesVal = document.getElementById('facesVal');
    const wsStatusVal = document.getElementById('wsStatusVal');
    
    const backlogTableBody = document.getElementById('backlogTableBody');
    const recTableBody = document.getElementById('recTableBody');
    
    const ctx = streamCanvas.getContext('2d');

    let socket = null;
    let isConnected = false;
    let isCameraActive = false;
    let isMirrored = false;
    let webcamStream = null;
    let captureInterval = null;
    let frameCount = 0;
    let lastTime = performance.now();

    // Mirror Camera Toggle Button
    if (mirrorToggleBtn) {
        mirrorToggleBtn.addEventListener('click', () => {
            isMirrored = !isMirrored;
            if (isMirrored) {
                mirrorToggleBtn.className = 'btn btn-primary';
                mirrorToggleBtn.textContent = '🪞 Mirroring: ON';
            } else {
                mirrorToggleBtn.className = 'btn btn-secondary';
                mirrorToggleBtn.textContent = '🪞 Mirror Camera';
            }
        });
    }

    // Single Camera Toggle Button
    if (cameraToggleBtn) {
        cameraToggleBtn.addEventListener('click', () => {
            if (isCameraActive) {
                stopCamera();
            } else {
                startCamera();
            }
        });
    }

    // Tab Switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            e.target.classList.add('active');
            const targetTab = e.target.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    async function startCamera() {
        statusMessage.textContent = 'Opening camera stream...';
        connectionOverlay.style.display = 'flex';
        
        // Try accessing Browser Webcam first
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } }
            });
            webcamVideo.srcObject = webcamStream;
            await webcamVideo.play();
        } catch (err) {
            console.warn('Browser webcam unavailable, falling back to server stream:', err);
        }

        connectWebSocket();
        isCameraActive = true;

        if (cameraToggleBtn) {
            cameraToggleBtn.textContent = '📹 Tắt Camera';
            cameraToggleBtn.className = 'btn btn-secondary';
        }
    }

    function stopCamera() {
        isCameraActive = false;
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        if (socket) {
            socket.close();
            socket = null;
        }
        isConnected = false;
        wsStatusVal.textContent = 'Disconnected';
        wsStatusVal.className = 'stat-value text-warning';
        connectionOverlay.style.display = 'flex';
        statusMessage.textContent = 'Camera is stopped. Click "📷 Bật Camera" to resume.';

        if (cameraToggleBtn) {
            cameraToggleBtn.textContent = '📷 Bật Camera';
            cameraToggleBtn.className = 'btn btn-primary';
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            isConnected = true;
            wsStatusVal.textContent = 'Connected';
            wsStatusVal.className = 'stat-value text-success';
            connectionOverlay.style.display = 'none';

            if (webcamStream) {
                startFrameStreaming();
            }
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.combo) {
                activeComboBadge.textContent = data.combo;
            }
            if (data.faces) {
                facesVal.textContent = data.faces.length;
            }
            
            renderFrame(data.image, data.faces);
            updateFPS();
        };

        socket.onerror = (err) => {
            console.error('WebSocket Error:', err);
            statusMessage.textContent = 'Connection error. Retrying...';
        };

        socket.onclose = () => {
            if (isCameraActive) {
                wsStatusVal.textContent = 'Reconnecting...';
                wsStatusVal.className = 'stat-value text-warning';
            }
        };
    }

    function startFrameStreaming() {
        if (captureInterval) clearInterval(captureInterval);
        const offCanvas = document.createElement('canvas');
        const offCtx = offCanvas.getContext('2d');

        captureInterval = setInterval(() => {
            if (socket && socket.readyState === WebSocket.OPEN && webcamVideo.videoWidth > 0 && isCameraActive) {
                offCanvas.width = webcamVideo.videoWidth;
                offCanvas.height = webcamVideo.videoHeight;
                offCtx.drawImage(webcamVideo, 0, 0, offCanvas.width, offCanvas.height);
                const base64Data = offCanvas.toDataURL('image/jpeg', 0.6);
                socket.send(base64Data);
            }
        }, 100);
    }

    function renderFrame(base64Image, faces) {
        const img = new Image();
        img.onload = () => {
            streamCanvas.width = img.width;
            streamCanvas.height = img.height;

            if (isMirrored) {
                ctx.save();
                ctx.scale(-1, 1);
                ctx.translate(-img.width, 0);
                ctx.drawImage(img, 0, 0);
                ctx.restore();
            } else {
                ctx.drawImage(img, 0, 0);
            }

            if (faces && faces.length > 0) {
                faces.forEach(face => {
                    const [x, y, w, h] = face.bbox;
                    const drawX = isMirrored ? (img.width - x - w) : x;

                    let strokeColor = '#ef4444'; // REJECTED / UNKNOWN (Red)

                    if (face.status === 'RECOGNIZED') {
                        strokeColor = '#10b981'; // RECOGNIZED (Green)
                    } else if (face.status === 'UNCERTAIN') {
                        strokeColor = '#f59e0b'; // UNCERTAIN (Yellow)
                    }

                    ctx.strokeStyle = strokeColor;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(drawX, y, w, h);

                    ctx.fillStyle = strokeColor;
                    const text = face.label || face.status;
                    ctx.font = '14px Inter, sans-serif';
                    const textWidth = ctx.measureText(text).width;
                    ctx.fillRect(drawX, Math.max(0, y - 24), textWidth + 12, 24);

                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(text, drawX + 6, Math.max(14, y - 7));
                });
            }
        };
        img.src = base64Image;
    }

    function updateFPS() {
        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
            fpsVal.textContent = frameCount;
            frameCount = 0;
            lastTime = now;
        }
    }

    function fetchLogs() {
        fetch('/api/v1/logs/backlog?limit=20')
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    backlogTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No error logs recorded.</td></tr>';
                    return;
                }
                backlogTableBody.innerHTML = data.map(row => `
                    <tr>
                        <td><span class="text-danger font-semibold">${row.failure_stage}</span></td>
                        <td>${row.failure_reason}</td>
                        <td>${row.active_ai_combo || 'N/A'}</td>
                        <td>${row.track_id || 'N/A'}</td>
                    </tr>
                `).join('');
            }).catch(() => {});

        fetch('/api/v1/logs/recognition?limit=20')
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    recTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No recognition logs recorded.</td></tr>';
                    return;
                }
                recTableBody.innerHTML = data.map(row => `
                    <tr>
                        <td><span class="${row.status === 'RECOGNIZED' ? 'text-success' : (row.status === 'UNCERTAIN' ? 'text-warning' : 'text-danger')}">${row.status}</span></td>
                        <td>${row.similarity ? row.similarity.toFixed(2) : '0.00'}</td>
                        <td>${row.track_id || 'N/A'}</td>
                        <td>${row.active_ai_combo || 'N/A'}</td>
                    </tr>
                `).join('');
            }).catch(() => {});
    }

    setInterval(fetchLogs, 3000);
    fetchLogs();

    // Auto-start camera on page load
    startCamera();
});
