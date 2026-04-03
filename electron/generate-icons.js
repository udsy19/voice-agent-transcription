#!/usr/bin/env node
/**
 * Generate tray icon PNGs for Muse.
 * Run once: node generate-icons.js
 * Creates 32x32 RGBA PNGs in assets/
 */

const fs = require('fs');
const path = require('path');

const SIZE = 32;  // 32x32 @2x = 16x16 tray icon on macOS

function createIcon(color, innerColor) {
  const buf = Buffer.alloc(SIZE * SIZE * 4);
  const cx = SIZE / 2, cy = SIZE / 2, r = 12, ir = 5;

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - cx, dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const idx = (y * SIZE + x) * 4;

      if (innerColor && dist <= ir) {
        // Inner circle
        buf[idx] = innerColor[0]; buf[idx+1] = innerColor[1];
        buf[idx+2] = innerColor[2]; buf[idx+3] = innerColor[3];
      } else if (dist <= r) {
        // Outer circle
        buf[idx] = color[0]; buf[idx+1] = color[1];
        buf[idx+2] = color[2]; buf[idx+3] = color[3];
      } else if (dist <= r + 1) {
        // Anti-alias
        const alpha = Math.max(0, 1 - (dist - r));
        buf[idx] = color[0]; buf[idx+1] = color[1];
        buf[idx+2] = color[2]; buf[idx+3] = Math.round(alpha * color[3]);
      }
    }
  }
  return buf;
}

// Write raw RGBA — electron's nativeImage.createFromBuffer can read this
const icons = {
  'tray-idle.rgba':       createIcon([56, 161, 105, 255], [255, 255, 255, 200]),   // green + white center
  'tray-recording.rgba':  createIcon([229, 62, 62, 255], [255, 255, 255, 220]),    // red + white center
  'tray-processing.rgba': createIcon([204, 138, 36, 255], [255, 255, 255, 200]),   // amber + white center
  'tray-handsfree.rgba':  createIcon([56, 161, 105, 255], [56, 161, 105, 255]),    // solid green
  'tray-loading.rgba':    createIcon([181, 164, 148, 255], null),                  // grey, no center
};

const outDir = path.join(__dirname, 'assets');
for (const [name, buf] of Object.entries(icons)) {
  fs.writeFileSync(path.join(outDir, name), buf);
  console.log(`  wrote ${name}`);
}
console.log('Done — icons in electron/assets/');
