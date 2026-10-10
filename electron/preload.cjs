// Minimal, safe bridge. The frontend is a normal web app and needs no Node
// access; we only expose a couple of read-only hints so UI can adapt if it
// wants to know it's running inside the desktop shell.

const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('djmate', {
  isDesktop: true,
  platform: process.platform,
});
