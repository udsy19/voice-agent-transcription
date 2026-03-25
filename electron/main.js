const { app, BrowserWindow, Tray, Menu, screen, nativeImage, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');

const PORT = 8528;

// In packaged app: Python files are in Resources/python/
// In dev mode: Python files are in the parent directory
const IS_PACKAGED = app.isPackaged;
const PROJECT_DIR = IS_PACKAGED
  ? path.join(process.resourcesPath, 'python')
  : path.join(__dirname, '..');

// Data directory (persistent, survives updates)
const DATA_DIR = path.join(app.getPath('userData'));

// Find Python binary
const PYTHON_CANDIDATES = [
  '/usr/local/bin/python3',
  '/opt/homebrew/bin/python3',
  '/usr/bin/python3',
];
const PYTHON = PYTHON_CANDIDATES.find(p => fs.existsSync(p)) || 'python3';

let mainWindow = null;
let pillWindow = null;
let tray = null;
let pythonProcess = null;
let ws = null;
let restartCount = 0;
let currentStatus = 'loading';

// ── Tray Icon (must be a real image, not empty) ─────────────────────────────

function createTrayIcon(label) {
  // Create a 32x32 canvas-drawn icon since we can't use emoji as tray icon
  // Use a simple colored circle: green=ready, red=recording, amber=processing
  const size = 32;
  const canvas = Buffer.alloc(size * size * 4); // RGBA

  const colors = {
    '🎙': [56, 161, 105, 255],   // green
    '🔴': [229, 62, 62, 255],    // red
    '⏳': [204, 138, 36, 255],   // amber
    '🟢': [56, 161, 105, 255],   // green
    '⏳loading': [181, 164, 148, 255], // light gray
  };
  const c = colors[label] || colors['🎙'];

  // Draw a filled circle
  const cx = size / 2, cy = size / 2, r = 6;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - cx, dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const idx = (y * size + x) * 4;
      if (dist <= r) {
        canvas[idx] = c[0]; canvas[idx+1] = c[1]; canvas[idx+2] = c[2]; canvas[idx+3] = c[3];
      } else if (dist <= r + 1) {
        // Anti-alias edge
        const alpha = Math.max(0, 1 - (dist - r));
        canvas[idx] = c[0]; canvas[idx+1] = c[1]; canvas[idx+2] = c[2]; canvas[idx+3] = Math.round(alpha * c[3]);
      }
    }
  }

  return nativeImage.createFromBuffer(canvas, { width: size, height: size, scaleFactor: 2 });
}

// ── Python Process ──────────────────────────────────────────────────────────

function loadEnv() {
  const env = { ...process.env, PYTHONPATH: PROJECT_DIR };
  const envPath = path.join(PROJECT_DIR, '.env');
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
      const m = line.match(/^([^#=]+)=(.*)$/);
      if (m) env[m[1].trim()] = m[2].trim();
    }
  }
  return env;
}

function startPython() {
  // Check if backend is already running (started by start.sh or another instance)
  if (process.env.VOICE_AGENT_EXTERNAL_BACKEND) {
    console.log('[main] External backend mode — skipping Python spawn');
    pollForReady();
    return;
  }

  // Also check if port is already in use (another instance running)
  const checkReq = http.get(`http://127.0.0.1:${PORT}/api/health`, (res) => {
    let body = '';
    res.on('data', c => body += c);
    res.on('end', () => {
      try {
        if (JSON.parse(body).ok) {
          console.log('[main] Backend already running on port — connecting');
          onBackendReady();
          return;
        }
      } catch {}
      actuallyStartPython();
    });
  });
  checkReq.on('error', () => actuallyStartPython());
  checkReq.setTimeout(1000, () => { checkReq.destroy(); actuallyStartPython(); });
}

function actuallyStartPython() {

  console.log(`[main] Starting Python (${IS_PACKAGED ? 'packaged' : 'dev'}) from ${PROJECT_DIR}`);
  const env = loadEnv();
  env.PYTHONPATH = PROJECT_DIR;
  env.VOICE_AGENT_DATA_DIR = DATA_DIR;

  pythonProcess = spawn(PYTHON, [path.join(PROJECT_DIR, 'app.py'), '--no-browser'], {
    cwd: '/tmp',
    env,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.on('data', d => console.log('[py]', d.toString().trim()));
  pythonProcess.stderr.on('data', d => {
    const s = d.toString().trim();
    if (s && !s.includes('RuntimeWarning')) console.log('[py:err]', s);
  });

  pythonProcess.on('exit', (code) => {
    console.log(`[main] Python exited (${code})`);
    pythonProcess = null;
    if (!app.isQuitting && restartCount < 3) {
      restartCount++;
      setTimeout(startPython, 2000);
    }
  });

  pollForReady();
}

function pollForReady() {
  const check = () => {
    http.get(`http://127.0.0.1:${PORT}/api/health`, res => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try {
          if (JSON.parse(body).ok) {
            console.log('[main] Backend ready');
            restartCount = 0;
            connectWS();
            // Tell renderer backend is up
            sendToMain('backend-ready', {});
            showPill(); // Show idle capsule immediately
            return;
          }
        } catch {}
        setTimeout(check, 500);
      });
    }).on('error', () => setTimeout(check, 500));
  };
  check();
}

// ── WebSocket (main process → controls pill + forwards to renderers) ────────

function connectWS() {
  if (ws) try { ws.close(); } catch {}

  ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);
  ws.on('open', () => console.log('[ws] connected'));

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === 'status') {
        currentStatus = msg.status;
        updateTrayIcon(msg.status);
        // Pill is always visible — just changes state (idle=small capsule, recording=expanded)
        showPill();
      }

      // Forward ALL messages to main window and pill
      sendToMain('ws-message', msg);
      sendToPill('ws-message', msg);

    } catch (e) {
      console.log('[ws] parse error:', e.message);
    }
  });

  ws.on('close', () => {
    if (!app.isQuitting) setTimeout(connectWS, 2000);
  });
  ws.on('error', () => {});
}

function sendToMain(channel, data) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    try { mainWindow.webContents.send(channel, data); } catch {}
  }
}

function sendToPill(channel, data) {
  if (pillWindow && !pillWindow.isDestroyed()) {
    try { pillWindow.webContents.send(channel, data); } catch {}
  }
}

function killPython() {
  if (!pythonProcess) return;
  pythonProcess.kill('SIGTERM');
  setTimeout(() => { try { if (pythonProcess) pythonProcess.kill('SIGKILL'); } catch {} }, 3000);
}

// ── Windows ─────────────────────────────────────────────────────────────────

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 750,
    minWidth: 700,
    minHeight: 500,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#FAF6F0',
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Load LOCAL file — not localhost. The HTML connects to localhost:8528 via JS.
  mainWindow.loadFile(path.join(__dirname, 'ui', 'app.html'));
  mainWindow.once('ready-to-show', () => mainWindow.show());

  mainWindow.on('close', (e) => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
}

function createPillWindow() {
  const { width } = screen.getPrimaryDisplay().workAreaSize;

  pillWindow = new BrowserWindow({
    width: 180,
    height: 36,
    x: Math.round((width - 180) / 2),
    y: 4,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    hasShadow: false,
    skipTaskbar: true,
    resizable: false,
    focusable: false,
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  pillWindow.loadFile(path.join(__dirname, 'ui', 'pill.html'));
  pillWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
}

function showPill() {
  if (pillWindow && !pillWindow.isDestroyed()) {
    pillWindow.showInactive();
  }
}

function hidePill() {
  if (pillWindow && !pillWindow.isDestroyed() && pillWindow.isVisible()) {
    pillWindow.hide();
  }
}

// ── Tray ────────────────────────────────────────────────────────────────────

function createTray() {
  tray = new Tray(createTrayIcon('⏳loading'));
  updateTrayMenu();

  tray.on('click', () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

function updateTrayIcon(status) {
  if (!tray) return;
  const map = { idle: '🎙', recording: '🔴', processing: '⏳', hands_free: '🟢', loading: '⏳loading' };
  tray.setImage(createTrayIcon(map[status] || '🎙'));
  updateTrayMenu();
}

function updateTrayMenu() {
  if (!tray) return;
  const labels = { idle: 'Ready', recording: 'Recording...', processing: 'Processing...', hands_free: 'Hands-Free', loading: 'Loading...' };
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: `Voice Agent — ${labels[currentStatus] || currentStatus}`, enabled: false },
    { type: 'separator' },
    { label: 'Show Window', click: () => { if (mainWindow) { mainWindow.show(); mainWindow.focus(); } } },
    { type: 'separator' },
    { label: 'Quit Voice Agent', click: () => { app.isQuitting = true; app.quit(); } },
  ]));
}

// ── IPC from renderers ──────────────────────────────────────────────────────

ipcMain.handle('get-port', () => PORT);

// ── App Lifecycle ───────────────────────────────────────────────────────────

app.on('ready', () => {
  createTray();
  createMainWindow();
  createPillWindow();
  startPython();
});

app.on('activate', () => {
  if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
});

app.on('before-quit', () => {
  app.isQuitting = true;
  killPython();
});

app.on('window-all-closed', () => { /* stay in tray */ });
