import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { LazyMotion, domAnimation } from 'framer-motion'
import './index.css'
// The Supabase-backed app (App.jsx: sign-in, 3D map, playlists, tagging) is
// switched off for now: its Supabase project is gone. DJMate is Next Track,
// which runs entirely on this Mac. To bring the old app back, render <App />
// from './App.jsx' here instead.
import NextTrackStandalone from './components/next/NextTrackStandalone'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LazyMotion features={domAnimation}>
      <NextTrackStandalone />
    </LazyMotion>
  </StrictMode>,
)
