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
});

// Prevent listener leaks on window reload
window.addEventListener('beforeunload', () => {
  for (const { channel, handler } of listeners) {
    ipcRenderer.removeListener(channel, handler);
  }
  listeners.length = 0;
});
