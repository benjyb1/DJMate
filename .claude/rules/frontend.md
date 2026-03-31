---
paths:
  - "Frontend/**/*.jsx"
  - "Frontend/**/*.tsx"
  - "Frontend/**/*.js"
  - "Frontend/**/*.ts"
  - "Frontend/**/*.css"
---

# Frontend Rules

## Stack
- React 18 with Vite (port 5173)
- Framer Motion (`m` components, `AnimatePresence`, spring physics)
- CSS custom properties in `Frontend/src/index.css` — use tokens, not hardcoded values
- Inter for UI text, JetBrains Mono for data/monospace

## Design System
- Glassmorphism: `var(--glass-bg)` + `backdrop-filter: blur(24px)` + `var(--glass-border)`
- Colours: purple `#7c3aed / #a855f7`, cyan `#00d4ff`, white `#e2e8f0`, bg `#0a0a14`
- No warm colours ever
- Border radius: 8–20px (`--radius-sm` to `--radius-xl`), pills use `--radius-pill`
- Shadows: use `--shadow-panel`, `--shadow-card`, `--shadow-float` tokens
- No emojis in UI — SVG icons only

## Component Patterns
- Buttons: `m.button` with `whileHover={{ scale: 1.02 }}`, `whileTap={{ scale: 0.97 }}`
- Tags/badges: pill shape with `--radius-pill`
- Panels: use `<GlassPanel>` component
- Accessibility: check `useReducedMotion` — disable shader bg + particles when true

## Album Art
- Fallback chain: Supabase storage → iTunes API → generative canvas
- All art uses `borderRadius: var(--radius-sm)`
- Shared utility: `Frontend/src/utils/coverUrl.js`
