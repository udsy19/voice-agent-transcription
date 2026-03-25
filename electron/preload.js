const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onBackendReady: (cb) => ipcRenderer.on('backend-ready', () => cb()),
  onWsMessage: (cb) => ipcRenderer.on('ws-message', (_, msg) => cb(msg)),
  onStatus: (cb) => ipcRenderer.on('status', (_, msg) => cb(msg)),
  getPort: () => ipcRenderer.invoke('get-port'),
});
