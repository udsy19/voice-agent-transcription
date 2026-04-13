const { contextBridge, ipcRenderer } = require('electron');

// Track listeners so we can clean them up on window unload
const listeners = [];

function addListener(channel, handler) {
  ipcRenderer.on(channel, handler);
  listeners.push({ channel, handler });
}

contextBridge.exposeInMainWorld('electronAPI', {
  onBackendReady: (cb) => addListener('backend-ready', () => cb()),
  onWsMessage: (cb) => addListener('ws-message', (_, msg) => cb(msg)),
  onStatus: (cb) => addListener('status', (_, msg) => cb(msg)),
  getPort: () => ipcRenderer.invoke('get-port'),
  openOAuth: (url) => ipcRenderer.invoke('open-oauth', url),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  onCommandPalette: (cb) => addListener('open-command-palette', () => cb()),
  onThemeChange: (cb) => addListener('theme-change', (_, msg) => cb(msg)),
  setTheme: (theme) => ipcRenderer.invoke('set-theme', theme),
  setIgnoreMouseEvents: (ignore, opts) => ipcRenderer.invoke('set-ignore-mouse', ignore, opts),
});

// Prevent listener leaks on window reload
window.addEventListener('beforeunload', () => {
  for (const { channel, handler } of listeners) {
    ipcRenderer.removeListener(channel, handler);
  }
  listeners.length = 0;
});
