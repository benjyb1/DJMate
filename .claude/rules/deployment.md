---
paths:
  - "render.yaml"
  - "vercel.json"
  - "Dockerfile"
  - ".env*"
  - "Procfile"
---

# Deployment Rules

## Architecture
- Frontend: Vercel (auto-deploys on push to main)
- Backend: Render web service (auto-deploys on push to main)
- Database: Supabase hosted Postgres

## Always push to main
User deploys via Vercel + Render auto-deploy. Always merge and push to main — no long-lived branches or PRs unless explicitly requested.

## Common Pitfalls
- Default API URL must point to Render in production builds (not localhost)
- DB init must be deferred to lifespan handler so Render's port binding isn't blocked
- Environment variables: check both Render dashboard and `.env` are in sync
