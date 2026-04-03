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

// ── Tray Icon (loaded from pre-generated assets) ────────────────────────────

const ICON_SIZE = 32;
const ICON_ASSETS = {};

function loadTrayIcons() {
  const assetsDir = path.join(__dirname, 'assets');
  const names = ['idle', 'recording', 'processing', 'handsfree', 'loading'];
  for (const name of names) {
    const file = path.join(assetsDir, `tray-${name}.rgba`);
    if (fs.existsSync(file)) {
      ICON_ASSETS[name] = nativeImage.createFromBuffer(
        fs.readFileSync(file),
        { width: ICON_SIZE, height: ICON_SIZE, scaleFactor: 2 }
      );
    }
  }
}

function getTrayIcon(status) {
  const map = { idle: 'idle', recording: 'recording', processing: 'processing', hands_free: 'handsfree', loading: 'loading' };
  return ICON_ASSETS[map[status] || 'idle'] || ICON_ASSETS['idle'];
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

let wsRetryDelay = 1000;

function connectWS() {
  if (ws) try { ws.close(); } catch {}

  ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);

  ws.on('open', () => {
    console.log('[ws] connected');
    wsRetryDelay = 1000; // Reset backoff on success
  });

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === 'status') {
        currentStatus = msg.status;
        updateTrayIcon(msg.status);
        showPill();
      }

      // Close OAuth window when auth completes
      if (msg.type === 'oauth_complete' && oauthWindow && !oauthWindow.isDestroyed()) {
        oauthWindow.close();
      }

      sendToMain('ws-message', msg);
      sendToPill('ws-message', msg);

    } catch (e) {
      console.warn('[ws] parse error:', e.message);
    }
  });

  ws.on('close', () => {
    if (!app.isQuitting) {
      wsRetryDelay = Math.min(wsRetryDelay * 1.5, 15000); // Exponential backoff, cap 15s
      setTimeout(connectWS, wsRetryDelay);
    }
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

const PILL_W = 320, PILL_H = 200;

function getPillPosition() {
  const display = screen.getPrimaryDisplay();
  const { width: screenW } = display.size;
  const workArea = display.workArea;
  const x = Math.round((screenW - PILL_W) / 2);
  // Top of work area (just below menu bar / notch)
  const y = workArea.y;
  return { x, y };
}

function createPillWindow() {
  const { x, y } = getPillPosition();

  pillWindow = new BrowserWindow({
    width: PILL_W,
    height: PILL_H,
    x, y,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    hasShadow: false,
    skipTaskbar: true,
    resizable: false,
    movable: false,
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

  // Reposition when display changes (dock resize, external monitor, fullscreen toggle)
  screen.on('display-metrics-changed', repositionPill);
}

function repositionPill() {
  if (!pillWindow || pillWindow.isDestroyed()) return;
  const { x, y } = getPillPosition();
  pillWindow.setPosition(x, y, false);
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
  loadTrayIcons();
  tray = new Tray(getTrayIcon('loading'));
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
  const icon = getTrayIcon(status);
  if (icon) tray.setImage(icon);
  updateTrayMenu();
}

function updateTrayMenu() {
  if (!tray) return;
  const labels = { idle: 'Ready', recording: 'Recording...', processing: 'Processing...', hands_free: 'Hands-Free', loading: 'Loading...' };
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: `Muse — ${labels[currentStatus] || currentStatus}`, enabled: false },
    { type: 'separator' },
    { label: 'Show Window', click: () => { if (mainWindow) { mainWindow.show(); mainWindow.focus(); } } },
    { type: 'separator' },
    { label: 'Quit Muse', click: () => { app.isQuitting = true; app.quit(); } },
  ]));
}

// ── OAuth Window ───────────────────────────────────────────────────────────

let oauthWindow = null;

function createOAuthWindow(url) {
  if (oauthWindow && !oauthWindow.isDestroyed()) {
    oauthWindow.focus();
    return;
  }
  oauthWindow = new BrowserWindow({
    width: 600, height: 700,
    title: 'Connect Account',
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  oauthWindow.loadURL(url);
  oauthWindow.on('closed', () => { oauthWindow = null; });
}

// ── IPC from renderers ──────────────────────────────────────────────────────

ipcMain.handle('get-port', () => PORT);
ipcMain.handle('open-oauth', (_, url) => { createOAuthWindow(url); return true; });

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
  if (ws) try { ws.close(); } catch {}
  screen.removeListener('display-metrics-changed', repositionPill);
  killPython();
});

app.on('window-all-closed', () => { /* stay in tray */ });
