import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import './App.css';

const WS_URL = 'ws://localhost:8000/ws/live';
const API_BASE = 'http://localhost:8000';

const CAMERA_COLORS = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bff6b', '#ff6bff', '#ff9f43'];

/* Error Boundary to catch React rendering errors */
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error('React Error Boundary caught:', error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, color: '#ff4d4d', background: '#071428', minHeight: '100vh', fontFamily: 'monospace' }}>
          <h2>⚠️ Dashboard Error</h2>
          <p>{this.state.error?.message || 'Unknown error'}</p>
          <button onClick={() => { this.setState({ hasError: false }); window.location.reload(); }}
            style={{ padding: '8px 16px', background: '#00d4ff', color: '#000', border: 'none', cursor: 'pointer', borderRadius: 4 }}>
            🔄 Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const PIPELINE_STAGES = [
  { key: 'cam', icon: '📷', label: 'CAMERA' },
  { key: 'snn', icon: '🧠', label: 'SNN GATE' },
  { key: 'yolo', icon: '🎯', label: 'YOLOv8' },
  { key: 'anomaly', icon: '🔍', label: 'ANOMALY' },
  { key: 'score', icon: '📊', label: 'SCORING' },
  { key: 'compress', icon: '🗜️', label: 'COMPRESS' },
  { key: 'db', icon: '🗄️', label: 'FORENSIC' },
  { key: 'dash', icon: '📡', label: 'DASHBOARD' },
];

/* ============================================================
   CameraFeedPanel — Renders one camera's live feed + overlays
   ============================================================ */
function CameraFeedPanel({ camId, camData, onStop }) {
  const imgRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const score = camData?.score || 0;
  const category = camData?.category || 'IDLE';
  const detections = camData?.detections || [];
  const snnSpike = camData?.snn_spike || false;
  const snnMembrane = camData?.snn_membrane || 0;
  const fps = camData?.fps || 0;

  const getScoreColor = (s) => {
    if (s > 60) return '#00ff88';
    if (s > 30) return '#ffaa00';
    return '#ff4d4d';
  };

  const getCategoryLabel = (cat) => {
    if (cat === 'EVENT') return '🔴 EVENT';
    if (cat === 'NORMAL') return '🟡 NORMAL';
    return '🟢 IDLE';
  };

  // Close fullscreen on Escape
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setIsFullscreen(false); };
    if (isFullscreen) window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isFullscreen]);

  return (
    <div className={`cam-feed-panel ${category === 'EVENT' ? 'cam-event' : category === 'NORMAL' ? 'cam-normal' : ''} ${isFullscreen ? 'cam-fullscreen' : ''}`}>
      <div className="cam-feed-header">
        <div className="cam-feed-title">
          <span className="live-dot"></span>
          <span>{camId.toUpperCase().replace('_', ' ')}</span>
          <span className="cam-fps-badge">{fps} FPS</span>
          <span className={`cam-spike-badge ${snnSpike ? 'spike-active' : ''}`}>
            {snnSpike ? '⚡ SPIKE' : '— SKIP'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="cam-fullscreen-btn" onClick={() => setIsFullscreen(f => !f)} title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}>
            {isFullscreen ? '⊡' : '⛶'}
          </button>
          <button className="cam-stop-btn" onClick={() => onStop(camId)}>⏹ STOP</button>
        </div>
      </div>

      <div className={`cam-feed-container ${category === 'IDLE' ? 'feed-idle' : category === 'EVENT' ? 'feed-event' : ''}`}>
        {camData?.frame ? (
          <img
            ref={imgRef}
            src={`data:image/jpeg;base64,${camData.frame}`}
            alt={`${camId} Feed`}
            className={`live-feed-img ${category === 'IDLE' ? 'feed-blur' : ''}`}
          />
        ) : (
          <div className="feed-placeholder">⏳ Connecting to {camId}...</div>
        )}

        {/* Score overlay */}
        <div className="score-overlay" style={{ borderColor: getScoreColor(score) }}>
          <div className="score-big" style={{ color: getScoreColor(score) }}>{score}</div>
          <div className="score-label-small">/100</div>
          <div className="score-cat">{getCategoryLabel(category)}</div>
        </div>

        {/* SNN membrane bar */}
        <div className="membrane-bar-container">
          <div className="membrane-bar-label">SNN</div>
          <div className="membrane-bar-track">
            <div
              className={`membrane-bar-fill ${snnSpike ? 'spiked' : ''}`}
              style={{ width: `${Math.min(snnMembrane / (camData?.snn_threshold || 0.15) * 100, 100)}%` }}
            />
          </div>
          <div className="membrane-bar-value">{snnMembrane.toFixed(3)}</div>
        </div>
      </div>

      {/* Detection badges */}
      {detections.length > 0 && (
        <div className="cam-det-badges">
          {detections.slice(0, 5).map((det, i) => (
            <span key={i} className={`cam-det-badge ${det.is_person ? 'det-person' : 'det-object'}`}>
              {det.class} {(det.conf * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}


function App() {
  const [data, setData] = useState(null);
  const [camerasData, setCamerasData] = useState({});
  const [scoreHistory, setScoreHistory] = useState({});
  const [events, setEvents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [clips, setClips] = useState([]);
  const [prebufferClips, setPrebufferClips] = useState([]);
  const [clipDateFilter, setClipDateFilter] = useState('');
  const [videoModal, setVideoModal] = useState(null); // { url, title, type }
  const [availableCameras, setAvailableCameras] = useState([]);
  const [activeCamIds, setActiveCamIds] = useState([]);
  const [cameraLoading, setCameraLoading] = useState({});
  const [detecting, setDetecting] = useState(false);
  const [backendAlive, setBackendAlive] = useState(false);
  const [uptime, setUptime] = useState(0);
  const [sessionName, setSessionName] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date());
  const uptimeRef = useRef(null);
  const pollPauseRef = useRef(false);
  const fpsCounterRef = useRef(0);
  const [clientFps, setClientFps] = useState(0);

  const { lastJsonMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: () => true,
    reconnectInterval: 2000,
  });

  const wsConnected = readyState === ReadyState.OPEN;

  // ---- Client-side FPS counter ----
  useEffect(() => {
    const t = setInterval(() => {
      setClientFps(fpsCounterRef.current);
      fpsCounterRef.current = 0;
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // ---- Live clock ----
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // ---- Uptime counter ----
  useEffect(() => {
    if (activeCamIds.length > 0) {
      uptimeRef.current = setInterval(() => setUptime(u => u + 1), 1000);
    } else {
      clearInterval(uptimeRef.current);
      setUptime(0);
    }
    return () => clearInterval(uptimeRef.current);
  }, [activeCamIds.length]);

  const formatUptime = (s) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  const formatDate = (d) =>
    d.toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });

  const formatTime = (d) =>
    d.toLocaleTimeString('en-IN', { hour12: false });

  // ---- Poll camera status ----
  useEffect(() => {
    const fetchStatus = async () => {
      if (pollPauseRef.current) return;
      try {
        const res = await fetch(`${API_BASE}/api/camera/status`);
        const json = await res.json();
        const running = json.cameras
          ? Object.entries(json.cameras).filter(([, v]) => v.running).map(([k]) => k)
          : [];
        setActiveCamIds(running);
        setBackendAlive(true);
      } catch (e) {
        setBackendAlive(false);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // ---- Detect available cameras ----
  const detectCameras = useCallback(async () => {
    setDetecting(true);
    try {
      const res = await fetch(`${API_BASE}/api/cameras/detect`);
      const json = await res.json();
      setAvailableCameras(json.cameras || []);
    } catch (e) { }
    setDetecting(false);
  }, []);

  // Auto-detect on mount
  useEffect(() => {
    if (backendAlive) detectCameras();
  }, [backendAlive, detectCameras]);

  // ---- Start a camera ----
  const startCamera = useCallback(async (source) => {
    setCameraLoading(prev => ({ ...prev, [source]: true }));
    pollPauseRef.current = true;
    setTimeout(() => { pollPauseRef.current = false; }, 5000);
    try {
      const name = sessionName || `Demo_${new Date().toLocaleTimeString('en-IN', { hour12: false }).replace(/:/g, '')}`;
      const res = await fetch(`${API_BASE}/api/camera/start?source=${source}&session_name=${encodeURIComponent(name)}`, { method: 'POST' });
      const json = await res.json();
      if (json.status === 'started' || json.status === 'already_running') {
        setActiveCamIds(prev => [...new Set([...prev, json.cam_id])]);
        setBackendAlive(true);
      }
    } catch (e) { }
    setCameraLoading(prev => ({ ...prev, [source]: false }));
  }, [sessionName]);

  // ---- Stop a camera ----
  const stopCamera = useCallback(async (camId) => {
    pollPauseRef.current = true;
    setTimeout(() => { pollPauseRef.current = false; }, 5000);
    try {
      await fetch(`${API_BASE}/api/camera/stop?cam_id=${camId}`, { method: 'POST' });
      setActiveCamIds(prev => prev.filter(id => id !== camId));
    } catch (e) { }
  }, []);

  // ---- Stop all ----
  const stopAll = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/camera/stop_all`, { method: 'POST' });
      setActiveCamIds([]);
    } catch (e) { }
  }, []);

  // ---- Clear session ----
  const clearSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/session/clear`, { method: 'POST' });
      setScoreHistory({});
      setEvents([]);
      setAlerts([]);
      setClips([]);
    } catch (e) { }
  }, []);

  // ---- Fetch events ----
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/events?limit=25`);
        const json = await res.json();
        setEvents(json.events || []);
      } catch (e) { }
    };
    const interval = setInterval(fetchEvents, 3000);
    fetchEvents();
    return () => clearInterval(interval);
  }, []);

  // ---- Fetch alerts ----
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/alerts?unacknowledged_only=true`);
        const json = await res.json();
        setAlerts(json.alerts || []);
      } catch (e) { }
    };
    const interval = setInterval(fetchAlerts, 2000);
    fetchAlerts();
    return () => clearInterval(interval);
  }, []);

  // ---- Fetch clips ----
  useEffect(() => {
    const fetchClips = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/clips`, { cache: 'no-store' });
        const json = await res.json();
        setClips(json.clips || []);
      } catch (e) { }
    };
    const interval = setInterval(fetchClips, 5000);
    fetchClips();
    return () => clearInterval(interval);
  }, []);

  // ---- Fetch prebuffer recordings ----
  useEffect(() => {
    const fetchPB = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/prebuffer`, { cache: 'no-store' });
        const json = await res.json();
        setPrebufferClips(json.prebuffer || []);
      } catch (e) { }
    };
    const interval = setInterval(fetchPB, 5000);
    fetchPB();
    return () => clearInterval(interval);
  }, []);

  // ---- Process WebSocket data ----
  useEffect(() => {
    if (lastJsonMessage) {
      fpsCounterRef.current += 1;
      setData(lastJsonMessage);
      setBackendAlive(true);

      // Store per-camera data
      if (lastJsonMessage.cameras) {
        setCamerasData(lastJsonMessage.cameras);

        // Update per-camera score history
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-IN', { hour12: false });
        setScoreHistory(prev => {
          const updated = { ...prev };
          for (const [camId, cd] of Object.entries(lastJsonMessage.cameras)) {
            const camHist = [...(updated[camId] || []), {
              time: timeStr,
              score: cd.score || 0,
              spike: cd.snn_spike ? 100 : 0,
            }];
            updated[camId] = camHist.slice(-120);
          }
          return updated;
        });
      }
    }
  }, [lastJsonMessage]);

  // ---- Helpers ----
  const getScoreColor = (score) => {
    if (score > 60) return '#00ff88';
    if (score > 30) return '#ffaa00';
    return '#ff4d4d';
  };

  const getCategoryLabel = (cat) => {
    if (cat === 'EVENT') return '🔴 EVENT';
    if (cat === 'NORMAL') return '🟡 NORMAL';
    return '🟢 IDLE';
  };

  const getCategoryClass = (cat) => {
    if (cat === 'EVENT') return 'severity-critical';
    if (cat === 'NORMAL') return 'severity-high';
    return 'severity-low';
  };

  // Aggregate stats from first camera for backward compat
  const firstCamId = activeCamIds[0] || '';
  const firstCamData = (firstCamId && camerasData[firstCamId]) || data || {};
  const detections = firstCamData.detections || [];
  const snnSpike = firstCamData.snn_spike || false;
  const activeCount = data?.active_count || 0;
  const displayFps = (firstCamData.fps && firstCamData.fps > 0) ? firstCamData.fps : clientFps;

  // Merge all cameras into a single chart dataset
  const chartCamIds = activeCamIds.length > 0 ? activeCamIds : Object.keys(scoreHistory);
  const mergedChartData = useMemo(() => {
    if (chartCamIds.length === 0) return [];
    let maxLen = 0;
    for (const id of chartCamIds) {
      maxLen = Math.max(maxLen, (scoreHistory[id] || []).length);
    }
    const merged = [];
    for (let i = 0; i < maxLen; i++) {
      const entry = {};
      for (const camId of chartCamIds) {
        const hist = scoreHistory[camId] || [];
        if (hist[i]) {
          entry.time = entry.time || hist[i].time;
          entry[`score_${camId}`] = hist[i].score;
        }
      }
      if (entry.time) merged.push(entry);
    }
    return merged;
  }, [scoreHistory, chartCamIds]);

  return (
    <div className="app">
      {/* ======== HEADER ======== */}
      <header className="header">
        <div className="header-top-row">
          <div className="header-badge">HACKARENA'26 // TEAM SPECTRUM</div>
          <div className="header-clock">
            <div className="clock-date">{formatDate(currentTime)}</div>
            <div className="clock-time">{formatTime(currentTime)}</div>
          </div>
        </div>
        <h1>EDGEVID <span className="accent">LOWBAND</span></h1>
        <p className="subtitle">NEUROMORPHIC EDGE-AI DVR — MULTI-CAMERA SURVEILLANCE</p>
      </header>

      {/* ======== PIPELINE INDICATOR ======== */}
      <div className="pipeline-bar">
        {PIPELINE_STAGES.map((stage, i) => {
          let stepClass = 'idle';
          if (activeCamIds.length > 0) {
            if (stage.key === 'snn' && snnSpike) stepClass = 'active spiking';
            else if (stage.key === 'yolo' && snnSpike && detections.length > 0) stepClass = 'active detecting';
            else if (stage.key === 'anomaly' && data?.active_tracks > 0) stepClass = 'active tracking';
            else stepClass = 'active';
          }
          return (
            <React.Fragment key={stage.key}>
              <div className={`pipeline-step ${stepClass}`}>
                <span>{stage.icon}</span>
                <span>{stage.label}</span>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <span className={`pipeline-arrow ${activeCamIds.length > 0 ? 'flowing' : ''}`}>→</span>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* ======== TOP METRICS ======== */}
      <div className="metrics-bar">
        <div className="metric-item">
          <div className="metric-value cyan">{activeCount} <span style={{fontSize: 11, opacity: 0.6}}>CAM{activeCount !== 1 ? 'S' : ''}</span></div>
          <div className="metric-label">📷 ACTIVE</div>
        </div>
        <div className="metric-item">
          <div className={`metric-value ${snnSpike ? 'green pulse' : 'red'}`}>
            {snnSpike ? '⚡ SPIKE' : '— SKIP'}
          </div>
          <div className="metric-label">🧠 SNN GATE</div>
        </div>
        <div className="metric-item">
          <div className="metric-value cyan">{displayFps} fps</div>
          <div className="metric-label">🎬 FPS</div>
        </div>
        <div className="metric-item">
          <div className="metric-value orange">{data?.spike_rate || 0}%</div>
          <div className="metric-label">📡 SPIKE RATE</div>
        </div>
        <div className="metric-item">
          <div className="metric-value red">{data?.alerts || 0}</div>
          <div className="metric-label">🚨 ALERTS</div>
        </div>
        <div className="metric-item">
          <div className="metric-value green">{data?.frame_count || 0}</div>
          <div className="metric-label">🎞️ FRAMES</div>
        </div>
        {activeCamIds.length > 0 && (
          <div className="metric-item">
            <div className="metric-value cyan">{formatUptime(uptime)}</div>
            <div className="metric-label">⏱️ UPTIME</div>
          </div>
        )}
      </div>

      {/* ======== CAMERA MANAGER ======== */}
      <div className="panel camera-manager-panel">
        <div className="panel-header">
          <span>📷 CAMERA MANAGER — MULTI-CAM CONTROL</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div className={`connection-status ${wsConnected || backendAlive ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              {wsConnected || backendAlive ? 'LIVE' : 'OFFLINE'}
            </div>
            <button className="detect-btn" onClick={detectCameras} disabled={detecting || !backendAlive}>
              {detecting ? '🔍 Scanning...' : '🔍 DETECT CAMERAS'}
            </button>
            {activeCamIds.length > 1 && (
              <button className="cam-stop-all-btn" onClick={stopAll}>⏹ STOP ALL</button>
            )}
          </div>
        </div>

        <div className="camera-grid-manager">
          {availableCameras.length === 0 && !detecting && (
            <div className="no-cameras-msg">
              {backendAlive
                ? 'Click "DETECT CAMERAS" to scan for connected cameras'
                : '⛔ Backend offline — start the server first'}
            </div>
          )}
          {availableCameras.map((cam) => {
            const camId = cam.cam_id || `cam_${cam.index}`;
            const isActive = activeCamIds.includes(camId);
            const isLoading = cameraLoading[cam.index];
            return (
              <div key={cam.index} className={`camera-card ${isActive ? 'cam-active' : ''}`}>
                <div className="camera-card-icon">
                  {cam.index === 0 ? '💻' : '📹'}
                </div>
                <div className="camera-card-info">
                  <div className="camera-card-name">{cam.name}</div>
                  <div className="camera-card-res">{cam.resolution} • Index {cam.index}</div>
                </div>
                <button
                  className={`camera-card-btn ${isActive ? 'active' : ''}`}
                  onClick={() => isActive ? stopCamera(camId) : startCamera(cam.index)}
                  disabled={isLoading}
                >
                  {isLoading ? '⏳' : isActive ? '⏹ STOP' : '▶ START'}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* ======== MULTI-CAMERA FEEDS ======== */}
      {activeCamIds.length > 0 && (
        <div className={`multi-cam-grid cam-count-${Math.min(activeCamIds.length, 4)}`}>
          {activeCamIds.map(camId => (
            <CameraFeedPanel
              key={camId}
              camId={camId}
              camData={camerasData[camId]}
              onStop={stopCamera}
            />
          ))}
        </div>
      )}

      {/* No cameras active placeholder */}
      {activeCamIds.length === 0 && (
        <div className="panel feed-panel">
          <div className="panel-header">
            <span className="live-dot off"></span>
            <span>LIVE FEED</span>
          </div>
          <div className="feed-container">
            <div className="feed-placeholder">
              {!backendAlive
                ? '⛔ Backend offline — start the server first'
                : '📷 No cameras active — Use Camera Manager above to start'}
            </div>
          </div>
        </div>
      )}

      {/* ======== MAIN GRID ======== */}
      <div className="main-grid">

        {/* ---- Score Chart ---- */}
        <div className="panel">
          <div className="panel-header">📊 FRAME INTELLIGENCE SCORE {chartCamIds.length === 1 ? `— ${chartCamIds[0].toUpperCase().replace('_', ' ')}` : chartCamIds.length > 1 ? '— ALL CAMERAS' : ''}</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={mergedChartData}>
              <defs>
                {chartCamIds.map((camId, idx) => (
                  <linearGradient key={camId} id={`scoreGrad_${camId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CAMERA_COLORS[idx % CAMERA_COLORS.length]} stopOpacity={0.15}/>
                    <stop offset="95%" stopColor={CAMERA_COLORS[idx % CAMERA_COLORS.length]} stopOpacity={0}/>
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#0d2d50" />
              <XAxis dataKey="time" stroke="#4a7a9b" tick={false} />
              <YAxis domain={[0, 100]} stroke="#4a7a9b" fontSize={10} />
              <Tooltip
                contentStyle={{
                  background: '#071428',
                  border: '1px solid #0d2d50',
                  borderRadius: 8,
                  fontSize: 12,
                  fontFamily: 'JetBrains Mono, monospace'
                }}
                labelStyle={{ color: '#4a7a9b' }}
              />
              {chartCamIds.length > 1 && (
                <Legend
                  verticalAlign="top"
                  height={28}
                  iconType="line"
                  wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#7ec8e3' }}
                />
              )}
              {chartCamIds.map((camId, idx) => (
                <Area
                  key={camId}
                  type="monotone"
                  dataKey={`score_${camId}`}
                  name={camId.replace(/_/g, ' ').toUpperCase()}
                  stroke={CAMERA_COLORS[idx % CAMERA_COLORS.length]}
                  strokeWidth={2}
                  fill={`url(#scoreGrad_${camId})`}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={800}
                  animationEasing="ease-in-out"
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
          <div className="score-bar">
            <div className="zone zone-idle">IDLE 0–30</div>
            <div className="zone zone-normal">NORMAL 30–60</div>
            <div className="zone zone-event">EVENT 60–100</div>
          </div>
        </div>

        {/* ---- Alert Panel ---- */}
        <div className="panel alert-panel">
          <div className="panel-header">
            <span>🚨 REAL-TIME ANOMALY ALERTS</span>
            <span style={{ fontSize: 10, color: '#4a7a9b' }}>{alerts.length} active</span>
          </div>
          <div className="alert-list">
            {alerts.length === 0 && !data?.last_alert ? (
              <div className="no-alerts">✅ No active alerts — system nominal</div>
            ) : (
              <>
                {data?.last_alert && (
                  <div className="alert-item latest">
                    <div className="alert-meta-row">
                      <div className="alert-type">{data.last_alert.type}</div>
                      <div className="alert-severity">CRITICAL</div>
                    </div>
                    <div className="alert-msg">{data.last_alert.message}</div>
                    <div className="alert-time-row">
                      <span className="alert-live-badge">⚡ LIVE</span>
                    </div>
                  </div>
                )}
                {alerts.map((alert, i) => (
                  <div key={i} className="alert-item">
                    <div className="alert-meta-row">
                      <div className="alert-type">{alert.alert_type}</div>
                      <div className="alert-severity">{alert.severity}</div>
                    </div>
                    <div className="alert-msg">{alert.message}</div>
                    <div className="alert-time-row">
                      <span className="alert-timestamp">
                        {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString('en-IN', { hour12: false }) : ''}
                      </span>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>

        {/* ---- Video Player Modal ---- */}
        {videoModal && (
          <div className="video-modal-overlay" onClick={() => setVideoModal(null)}>
            <div className="video-modal" onClick={e => e.stopPropagation()}>
              <div className="video-modal-header">
                <span>{videoModal.title}</span>
                <button className="video-modal-close" onClick={() => setVideoModal(null)}>✕</button>
              </div>
              {videoModal.type === 'avi' ? (
                <div className="video-modal-avi-notice">
                  <div style={{ fontSize: 32, marginBottom: 10 }}>📹</div>
                  <div style={{ fontWeight: 700, marginBottom: 6, color: 'var(--orange)' }}>AVI / XVID — Not supported in browser</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 16 }}>Browsers cannot play AVI files natively.<br/>Download and open with VLC or Windows Media Player.</div>
                  <a className="video-modal-dl" style={{ padding: '8px 20px', background: 'rgba(0,212,255,0.12)', borderRadius: 6, border: '1px solid rgba(0,212,255,0.25)' }} href={videoModal.url} download target="_blank" rel="noopener noreferrer">⬇ DOWNLOAD FILE</a>
                </div>
              ) : (
                <video
                  className="video-modal-player"
                  src={videoModal.url}
                  controls
                  autoPlay
                  onError={e => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
              )}
              {videoModal.type !== 'avi' && (
                <div className="video-modal-avi-notice" style={{ display: 'none' }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>⚠️</div>
                  <div style={{ fontWeight: 700, marginBottom: 6, color: 'var(--orange)' }}>Cannot play in browser</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 12 }}>This video codec is not supported natively.<br/>Download and open with VLC.</div>
                  <a className="video-modal-dl" href={videoModal.url} download target="_blank" rel="noopener noreferrer">⬇ DOWNLOAD FILE</a>
                </div>
              )}
              <div className="video-modal-footer">
                <a className="video-modal-dl" href={videoModal.url} download target="_blank" rel="noopener noreferrer">⬇ DOWNLOAD</a>
              </div>
            </div>
          </div>
        )}

        {/* ---- Event Clips with Date Filter ---- */}
        <div className="panel clips-panel">
          <div className="panel-header">
            <span>🎬 RECORDED CLIPS</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="date"
                className="clip-date-input"
                value={clipDateFilter}
                onChange={e => setClipDateFilter(e.target.value)}
                title="Filter clips by date"
              />
              {clipDateFilter && (
                <button className="clip-date-clear" onClick={() => setClipDateFilter('')}>✕</button>
              )}
              <span style={{ fontSize: 10, color: '#4a7a9b' }}>
                {(() => {
                  const filtered = clipDateFilter
                    ? clips.filter(c => c.start_time && c.start_time.startsWith(clipDateFilter))
                    : clips;
                  return `${filtered.length} clip${filtered.length !== 1 ? 's' : ''}`;
                })()}
              </span>
            </div>
          </div>
          <div className="clips-table-wrapper">
            <table className="clips-table">
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>TIME</th>
                  <th>CAM</th>
                  <th>TYPE</th>
                  <th>QUALITY</th>
                  <th>DURATION</th>
                  <th>SIZE</th>
                  <th>FPS</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const filtered = clipDateFilter
                    ? clips.filter(c => c.start_time && c.start_time.startsWith(clipDateFilter))
                    : clips;
                  if (filtered.length === 0) {
                    return (
                      <tr>
                        <td colSpan={9} style={{ textAlign: 'center', padding: 30, color: '#4a7a9b' }}>
                          {clips.length === 0 ? 'No clips yet — events auto-record when persons detected' : 'No clips for selected date'}
                        </td>
                      </tr>
                    );
                  }
                  return filtered.map((clip, i) => {
                    const ts = clip.start_time ? new Date(clip.start_time) : null;
                    const dateStr = ts ? ts.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '-';
                    const timeStr = ts ? ts.toLocaleTimeString('en-IN', { hour12: false }) : '-';
                    const catClass = clip.category === 'EVENT' ? 'clip-event' : clip.category === 'NORMAL' ? 'clip-normal' : 'clip-idle';
                    const qualityLabel = clip.quality || 'HD';
                    const clipUrl = `${API_BASE}/api/clips/${clip.filename}`;
                    return (
                      <tr key={i} className={catClass}>
                        <td>{dateStr}</td>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>{timeStr}</td>
                        <td>{clip.camera || 'cam_0'}</td>
                        <td className={`clip-cat-cell ${catClass}`}>{clip.category}</td>
                        <td>{qualityLabel}</td>
                        <td>{clip.duration_sec ? `${clip.duration_sec}s` : '-'}</td>
                        <td>{clip.size_kb ? `${clip.size_kb.toFixed(0)} KB` : '-'}</td>
                        <td>{clip.fps || 15}</td>
                        <td style={{ display: 'flex', gap: 4 }}>
                          <button className="clip-play-btn" onClick={() => setVideoModal({ url: clipUrl, title: clip.filename, type: 'mp4' })} title="Play">▶</button>
                          <a className="clip-dl-btn" href={clipUrl} download target="_blank" rel="noopener noreferrer" title="Download">⬇</a>
                        </td>
                      </tr>
                    );
                  });
                })()}
              </tbody>
            </table>
          </div>
        </div>

        {/* ---- Pre-Buffer Recordings ---- */}
        <div className="panel clips-panel prebuffer-panel">
          <div className="panel-header">
            <span>⏪ PRE-BUFFER RECORDINGS (30s before event)</span>
            <span style={{ fontSize: 10, color: '#4a7a9b' }}>{prebufferClips.length} recording{prebufferClips.length !== 1 ? 's' : ''}</span>
          </div>
          <div className="clips-table-wrapper">
            <table className="clips-table">
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>TIME</th>
                  <th>EVENT TYPE</th>
                  <th>DURATION</th>
                  <th>SIZE</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {prebufferClips.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: 30, color: '#4a7a9b' }}>
                      No pre-buffer recordings yet — these save 30s before loitering/anomaly alerts
                    </td>
                  </tr>
                ) : (
                  prebufferClips.map((pb, i) => {
                    const ts = pb.start_time ? new Date(pb.start_time) : null;
                    const dateStr = ts ? ts.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '-';
                    const timeStr = ts ? ts.toLocaleTimeString('en-IN', { hour12: false }) : '-';
                    const pbUrl = `${API_BASE}/api/prebuffer/${pb.filename}`;
                    return (
                      <tr key={i} className="clip-event">
                        <td>{dateStr}</td>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>{timeStr}</td>
                        <td style={{ color: 'var(--orange)', fontWeight: 700 }}>{pb.event_type}</td>
                        <td>{pb.duration_sec ? `${pb.duration_sec}s` : '-'}</td>
                        <td>{pb.size_kb ? `${pb.size_kb.toFixed(0)} KB` : '-'}</td>
                        <td style={{ display: 'flex', gap: 4 }}>
                          <button className="clip-play-btn" onClick={() => setVideoModal({ url: pbUrl, title: pb.filename, type: 'avi' })} title="Play">▶</button>
                          <a className="clip-dl-btn" href={pbUrl} download target="_blank" rel="noopener noreferrer" title="Download">⬇</a>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ---- Forensic Event Log ---- */}
        <div className="panel log-panel">
          <div className="panel-header">
            <span>📋 FORENSIC EVENT LOG — SMART CCTV DIARY</span>
            <span style={{ fontSize: 10, color: '#4a7a9b' }}>{events.length} records</span>
          </div>
          <div className="event-table-wrapper">
            <table className="event-table">
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>TIME</th>
                  <th>CAM</th>
                  <th>FRAME</th>
                  <th>SCORE</th>
                  <th>CATEGORY</th>
                  <th>TYPE</th>
                  <th>PERSONS</th>
                  <th>SEVERITY</th>
                </tr>
              </thead>
              <tbody>
                {events.length === 0 ? (
                  <tr>
                    <td colSpan={9} style={{ textAlign: 'center', padding: 30, color: '#4a7a9b' }}>
                      No events recorded yet — start a camera to begin
                    </td>
                  </tr>
                ) : (
                  events.map((event, i) => {
                    const ts = event.timestamp ? new Date(event.timestamp) : null;
                    return (
                      <tr key={i}>
                        <td>{ts ? ts.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) : '-'}</td>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                          {ts ? ts.toLocaleTimeString('en-IN', { hour12: false }) : '-'}
                        </td>
                        <td>{event.camera_id || 'CAM_01'}</td>
                        <td>{event.frame_number}</td>
                        <td style={{ color: getScoreColor(event.score), fontWeight: 700 }}>{event.score}</td>
                        <td className={getCategoryClass(event.category)}>{getCategoryLabel(event.category)}</td>
                        <td>{event.event_type}</td>
                        <td>{event.person_count}</td>
                        <td className={`severity-${event.severity?.toLowerCase()}`}>{event.severity}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ======== SESSION CONTROLS ======== */}
      <div className="session-bar">
        <div className="session-controls">
          <input
            className="session-input"
            type="text"
            placeholder="Session name (e.g. Demo_Loitering)"
            value={sessionName}
            onChange={e => setSessionName(e.target.value)}
          />
          <button className="session-btn clear-btn" onClick={clearSession} disabled={activeCamIds.length > 0}>
            🗑️ CLEAR DATA
          </button>
          <a className="session-btn export-btn" href={`${API_BASE}/api/export`} target="_blank" rel="noopener noreferrer">
            📥 EXPORT CSV
          </a>
        </div>
      </div>

      {/* ======== FOOTER ======== */}
      <footer className="footer">
        <div className="quote">
          "Traditional CCTV records everything. EdgeVid LowBand remembers what matters."
        </div>
        <div className="footer-info">
          TEAM SPECTRUM // HACKARENA'26 // PS-04 // No cloud. No GPU. Just intelligence.
        </div>
      </footer>
    </div>
  );
}

function WrappedApp() {
  return <ErrorBoundary><App /></ErrorBoundary>;
}

export default WrappedApp;
