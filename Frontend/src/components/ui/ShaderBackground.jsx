import React, { useEffect, useRef } from 'react';

/**
 * Full-screen WebGL shader background — cosmic nebula with drifting gas clouds
 * and faint twinkling stars. Slow organic movement that complements the 3D
 * point-cloud galaxy without competing with it.
 */
export default function ShaderBackground() {
  const canvasRef = useRef(null);

  const vsSource = `
    attribute vec4 aVertexPosition;
    void main() {
      gl_Position = aVertexPosition;
    }
  `;

  const fsSource = `
    precision highp float;
    uniform vec2 iResolution;
    uniform float iTime;

    /* ── noise primitives ── */
    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }

    float hash21(vec2 p) {
      p = fract(p * vec2(234.34, 435.345));
      p += dot(p, p + 34.23);
      return fract(p.x * p.y);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    /* fractal brownian motion — 5 octaves with rotation per octave */
    float fbm(vec2 p) {
      float v = 0.0;
      float a = 0.5;
      vec2 shift = vec2(100.0);
      mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
      for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + shift;
        a *= 0.5;
      }
      return v;
    }

    /* ── main ── */
    void main() {
      vec2 uv = gl_FragCoord.xy / iResolution.xy;
      float aspect = iResolution.x / iResolution.y;
      vec2 st = vec2(uv.x * aspect, uv.y) * 2.5;

      float t = iTime * 0.04;

      /* domain-warped FBM for organic nebula shapes */
      float n1 = fbm(st + vec2(t * 0.3, t * 0.15));
      float n2 = fbm(st + vec2(5.2 + t * 0.2, 1.3 - t * 0.1));
      float n3 = fbm(st + vec2(n1, n2) * 1.8 + t * 0.1);

      /* second warp layer for depth */
      float n4 = fbm(st * 1.5 + vec2(n3 * 0.8, n1 * 0.6) + vec2(-t * 0.15, t * 0.08));

      /* base: very dark space */
      vec3 color = vec3(0.012, 0.012, 0.025);

      /* nebula cloud layers — purples and deep blues */
      vec3 purpleDeep  = vec3(0.10, 0.02, 0.20);
      vec3 purpleMid   = vec3(0.18, 0.06, 0.35);
      vec3 blueDeep    = vec3(0.02, 0.06, 0.18);
      vec3 cyanHint    = vec3(0.02, 0.12, 0.20);

      /* blend nebula colors using noise layers */
      float cloud = smoothstep(0.3, 0.75, n3);
      float wisps = smoothstep(0.4, 0.8, n4) * 0.6;
      float glow  = smoothstep(0.5, 0.9, n3 * n4) * 0.4;

      color += purpleDeep * cloud * 0.8;
      color += blueDeep * wisps;
      color += purpleMid * glow;
      color += cyanHint * smoothstep(0.55, 0.85, n2) * 0.2;

      /* soft vignette — darker at edges, subtle focus center */
      vec2 vigUv = uv - 0.5;
      float vig = 1.0 - dot(vigUv, vigUv) * 1.6;
      vig = smoothstep(0.0, 1.0, vig);
      color *= 0.5 + vig * 0.5;

      /* ── star field ── */
      /* layer 1: faint background stars */
      vec2 starGrid1 = floor(gl_FragCoord.xy * 0.4);
      float starVal1 = hash21(starGrid1);
      float star1 = step(0.993, starVal1);
      float twinkle1 = sin(iTime * (1.0 + starVal1 * 2.0) + starVal1 * 6.28) * 0.4 + 0.6;
      float starBright1 = star1 * twinkle1 * (0.15 + starVal1 * 0.15);

      /* layer 2: sparse brighter stars */
      vec2 starGrid2 = floor(gl_FragCoord.xy * 0.15);
      float starVal2 = hash21(starGrid2 + 73.1);
      float star2 = step(0.997, starVal2);
      float twinkle2 = sin(iTime * 0.8 + starVal2 * 6.28) * 0.3 + 0.7;
      float starBright2 = star2 * twinkle2 * (0.3 + starVal2 * 0.2);

      /* star colors — mostly white with subtle purple/blue tints */
      vec3 starColor1 = mix(vec3(0.7, 0.7, 0.85), vec3(0.6, 0.5, 0.9), starVal1);
      vec3 starColor2 = mix(vec3(0.85, 0.85, 1.0), vec3(0.7, 0.65, 1.0), starVal2);

      color += starColor1 * starBright1;
      color += starColor2 * starBright2;

      gl_FragColor = vec4(color, 1.0);
    }
  `;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl');
    if (!gl) return;

    const compileShader = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vs = compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return;
    }

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(program, 'aVertexPosition');
    const resLoc = gl.getUniformLocation(program, 'iResolution');
    const timeLoc = gl.getUniformLocation(program, 'iTime');

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    window.addEventListener('resize', resize);
    resize();

    let raf;
    const start = Date.now();
    const render = () => {
      const t = (Date.now() - start) / 1000;
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);
      gl.uniform2f(resLoc, canvas.width, canvas.height);
      gl.uniform1f(timeLoc, t);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(posLoc);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 0.6,
      }}
    />
  );
}
