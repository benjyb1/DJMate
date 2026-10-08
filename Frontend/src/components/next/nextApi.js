// API client for the Next Track screen. Talks only to the local backend's
// /next routes, so it works without signing in to DJMate.

export const API_BASE = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');

async function request(path, opts) {
  const res = await fetch(`${API_BASE}/next${path}`, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.detail || res.statusText || 'Request failed');
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export const nextApi = {
  status: () => request('/status'),
  now: () => request('/now'),
  build: () => request('/index/build', { method: 'POST' }),
  suggest: (id, exclude = []) =>
    request(`/suggest?id=${encodeURIComponent(id)}&exclude=${encodeURIComponent(exclude.join(','))}`),
  track: (id) => request(`/track/${encodeURIComponent(id)}`),
  search: (q) => request(`/search?q=${encodeURIComponent(q)}`),
  reveal: (id) => request(`/reveal/${encodeURIComponent(id)}`, { method: 'POST' }),
  analyse: (blob, filename, kind) => {
    const fd = new FormData();
    fd.append('file', blob, filename);
    fd.append('kind', kind);
    return request('/analyse', { method: 'POST', body: fd });
  },
  artUrl: (id) => `${API_BASE}/next/art/${encodeURIComponent(id)}`,
};

// ── Microphone capture → 16-bit mono WAV ─────────────────────────────────────

function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buf = new ArrayBuffer(44 + length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, length * 2, true);
  let o = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, c[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/**
 * Record `seconds` of audio from the microphone. Calls onLevel(0..1) and
 * onProgress(0..1) while recording; resolves to a WAV Blob. Returns a
 * { promise, cancel } pair.
 */
export function recordClip(seconds, { onLevel, onProgress } = {}) {
  let cancelled = false;
  let cleanup = () => {};
  const promise = (async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const sink = ctx.createGain();
    sink.gain.value = 0;
    const chunks = [];
    const target = seconds * ctx.sampleRate;
    let got = 0;
    cleanup = () => {
      try { proc.disconnect(); src.disconnect(); sink.disconnect(); } catch { /* already gone */ }
      stream.getTracks().forEach(t => t.stop());
      ctx.close().catch(() => {});
    };
    return await new Promise((resolve, reject) => {
      proc.onaudioprocess = (e) => {
        if (cancelled) { cleanup(); reject(new Error('cancelled')); return; }
        const data = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(data);
        got += data.length;
        let peak = 0;
        for (let i = 0; i < data.length; i += 16) peak = Math.max(peak, Math.abs(data[i]));
        onLevel?.(Math.min(1, peak * 1.6));
        onProgress?.(Math.min(1, got / target));
        if (got >= target) {
          cleanup();
          resolve(encodeWav(chunks, ctx.sampleRate));
        }
      };
      src.connect(proc);
      proc.connect(sink);
      sink.connect(ctx.destination);
    });
  })();
  return { promise, cancel: () => { cancelled = true; cleanup(); } };
}
