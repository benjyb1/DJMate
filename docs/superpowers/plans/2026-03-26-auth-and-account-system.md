# Auth & Account System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user accounts with Supabase Auth on the central djmate-db, persist BYOS credentials per-user, replace text folder input with native folder picker, add Profile tab, fix layout bugs.

**Architecture:** Supabase Auth (email/password) on djmate-db with a `profiles` table storing username, BYOS Supabase credentials, and `has_imported_library` flag. Frontend gates on auth session instead of raw credential check. New `authStore.js` manages session state. Profile tab added to nav. Folder import uses `showDirectoryPicker()` (Chromium-only) with toast fallback for other browsers.

**Tech Stack:** Supabase Auth JS (`@supabase/supabase-js` already installed), React, Framer Motion, Zustand (already installed)

---

## File Structure

### New Files
- `Frontend/src/stores/authStore.js` — Zustand store for auth session, user profile, login/signup/logout
- `Frontend/src/utils/centralSupabase.js` — Supabase client pointing at djmate-db (the central auth DB, distinct from user's BYOS client)
- `Frontend/src/components/AuthScreen.jsx` — Login/Signup screen (replaces SetupScreen as gate)
- `Frontend/src/components/ProfileTab.jsx` — Profile tab content (username, Supabase linking modal, re-scan, logout)
- `Frontend/src/components/ui/SupabaseLinkModal.jsx` — Reusable modal for entering/updating BYOS Supabase credentials
- `Frontend/src/components/ui/Toast.jsx` — Simple toast notification component (for Chromium warning, etc.)

### Modified Files
- `Frontend/src/App.jsx` — Replace SetupScreen gate with AuthScreen gate; add PROFILE to NAV_TABS; pass auth state down; add "[username]'s Space" header; fix nav padding
- `Frontend/src/stores/credentialStore.js` — Add `hasImportedLibrary()` / `setImportedLibrary()` using localStorage (local cache, synced from profile)
- `Frontend/src/components/SetupScreen.jsx` — DELETE (replaced by AuthScreen + SupabaseLinkModal)
- `Frontend/src/components/LiveMode.jsx` — Remove redundant "LIVE SESSION" / "SET TRACKER" header; add top padding for nav; conditionally show "Upload Library" if not imported
- `Frontend/src/components/playlist/PlaylistSidebar.jsx` — Replace text folder input with `showDirectoryPicker()` button; change "Import Music Folder" to "Re-scan Library" if already imported
- `Frontend/src/components/playlist/PlaylistOrganiser.jsx` — Add top padding for nav clearance

### Database Migration (djmate-db)
- `profiles` table: `id` (uuid, FK auth.users), `username` (text, unique), `supabase_url` (text), `supabase_key` (text), `has_imported_library` (boolean, default false), `created_at` (timestamptz)
- RLS policies: users can only read/update their own row
- Trigger: auto-create profile row on auth.users insert

---

## Task 1: Database — Create profiles table on djmate-db

**Files:**
- Database migration on djmate-db Supabase

- [ ] **Step 1: Create profiles table with RLS**

Apply migration via `mcp__djmate-db__apply_migration`:

```sql
-- Create profiles table
CREATE TABLE public.profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username text UNIQUE NOT NULL,
  supabase_url text,
  supabase_key text,
  has_imported_library boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Users can read their own profile
CREATE POLICY "Users can read own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Users can insert their own profile (for signup)
CREATE POLICY "Users can insert own profile"
  ON public.profiles FOR INSERT
  WITH CHECK (auth.uid() = id);
```

- [ ] **Step 2: Create auto-profile trigger**

Apply migration:

```sql
-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.profiles (id, username)
  VALUES (NEW.id, NEW.raw_user_meta_data->>'username');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

- [ ] **Step 3: Verify table exists**

Query: `SELECT * FROM public.profiles LIMIT 1;` — should return empty result set, no error.

---

## Task 2: Central Supabase client + Auth store

**Files:**
- Create: `Frontend/src/utils/centralSupabase.js`
- Create: `Frontend/src/stores/authStore.js`
- Modify: `Frontend/src/stores/credentialStore.js`

- [ ] **Step 1: Create centralSupabase.js**

This client connects to YOUR djmate-db project (not the user's BYOS). We need the project URL and anon key from the djmate-db MCP. Use `mcp__djmate-db__get_project_url` and `mcp__djmate-db__get_publishable_keys` to get these values.

```javascript
// Frontend/src/utils/centralSupabase.js
import { createClient } from '@supabase/supabase-js';

const CENTRAL_URL = import.meta.env.VITE_CENTRAL_SUPABASE_URL || '<from get_project_url>';
const CENTRAL_KEY = import.meta.env.VITE_CENTRAL_SUPABASE_KEY || '<from get_publishable_keys>';

export const centralSupabase = createClient(CENTRAL_URL, CENTRAL_KEY);
```

- [ ] **Step 2: Create authStore.js (Zustand)**

```javascript
// Frontend/src/stores/authStore.js
import { create } from 'zustand';
import { centralSupabase } from '../utils/centralSupabase';

export const useAuthStore = create((set, get) => ({
  session: null,
  profile: null,
  loading: true,

  // Initialise — call once on app mount
  init: async () => {
    const { data: { session } } = await centralSupabase.auth.getSession();
    if (session) {
      const profile = await get().fetchProfile(session.user.id);
      set({ session, profile, loading: false });
    } else {
      set({ loading: false });
    }

    // Listen for auth changes (login/logout/token refresh)
    centralSupabase.auth.onAuthStateChange(async (_event, session) => {
      if (session) {
        const profile = await get().fetchProfile(session.user.id);
        set({ session, profile });
      } else {
        set({ session: null, profile: null });
      }
    });
  },

  fetchProfile: async (userId) => {
    const { data } = await centralSupabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();
    return data;
  },

  signUp: async (email, password, username) => {
    const { data, error } = await centralSupabase.auth.signUp({
      email,
      password,
      options: { data: { username } },
    });
    if (error) throw error;
    return data;
  },

  signIn: async (email, password) => {
    const { data, error } = await centralSupabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw error;
    return data;
  },

  signOut: async () => {
    await centralSupabase.auth.signOut();
    set({ session: null, profile: null });
  },

  updateUsername: async (username) => {
    const userId = get().session?.user?.id;
    if (!userId) return;
    const { error } = await centralSupabase
      .from('profiles')
      .update({ username })
      .eq('id', userId);
    if (error) throw error;
    set(s => ({ profile: { ...s.profile, username } }));
  },

  linkSupabase: async (url, key) => {
    const userId = get().session?.user?.id;
    if (!userId) return;
    const { error } = await centralSupabase
      .from('profiles')
      .update({ supabase_url: url, supabase_key: key })
      .eq('id', userId);
    if (error) throw error;
    set(s => ({ profile: { ...s.profile, supabase_url: url, supabase_key: key } }));
  },

  setImportedLibrary: async () => {
    const userId = get().session?.user?.id;
    if (!userId) return;
    await centralSupabase
      .from('profiles')
      .update({ has_imported_library: true })
      .eq('id', userId);
    set(s => ({ profile: { ...s.profile, has_imported_library: true } }));
  },

  // Derived getters
  get isLinked() { return !!(get().profile?.supabase_url); },
  get hasImported() { return !!(get().profile?.has_imported_library); },
  get username() { return get().profile?.username || ''; },
}));
```

- [ ] **Step 3: Update credentialStore.js**

The credential store should now read from the auth store's profile instead of standalone localStorage. But we keep the localStorage layer as a cache so the apiClient and supabaseClient work without changes:

```javascript
// Add to credentialStore.js — sync from profile
export function syncFromProfile(profile) {
  if (profile?.supabase_url && profile?.supabase_key) {
    saveCredentials(profile.supabase_url, profile.supabase_key);
  } else {
    clearCredentials();
  }
}
```

- [ ] **Step 4: Verify imports compile**

Run: `cd Frontend && npx vite build --mode development 2>&1 | head -30`
Expected: No import errors for new files.

- [ ] **Step 5: Commit**

```bash
git add Frontend/src/utils/centralSupabase.js Frontend/src/stores/authStore.js Frontend/src/stores/credentialStore.js
git commit -m "feat: add central Supabase client and auth store"
```

---

## Task 3: Auth Screen (Login/Signup)

**Files:**
- Create: `Frontend/src/components/AuthScreen.jsx`

- [ ] **Step 1: Create AuthScreen component**

Two-mode form (login / signup) with glassmorphism styling matching existing SetupScreen aesthetic. Fields: email, password, username (signup only). Uses `useAuthStore` for signUp/signIn.

Key behaviour:
- Toggle between "Sign In" and "Sign Up" modes
- On signup: collect username + email + password, call `authStore.signUp()`
- On login: email + password, call `authStore.signIn()`
- Show error messages inline
- Match existing glass/purple/cyan design system
- Centre on page with DJMate logo above

- [ ] **Step 2: Verify it renders standalone**

Temporarily import in App.jsx, confirm it renders without errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/src/components/AuthScreen.jsx
git commit -m "feat: add AuthScreen login/signup component"
```

---

## Task 4: Toast component + Chromium browser warning

**Files:**
- Create: `Frontend/src/components/ui/Toast.jsx`

- [ ] **Step 1: Create Toast component**

Simple animated toast notification using Framer Motion. Auto-dismisses after 4 seconds. Positioned bottom-centre. Glass styling.

```javascript
// Usage: <Toast message="..." visible={bool} onDismiss={fn} />
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/src/components/ui/Toast.jsx
git commit -m "feat: add Toast notification component"
```

---

## Task 5: Supabase Link Modal

**Files:**
- Create: `Frontend/src/components/ui/SupabaseLinkModal.jsx`

- [ ] **Step 1: Create SupabaseLinkModal**

Reusable modal for entering/updating BYOS Supabase credentials. Extracted from the old SetupScreen step 0 logic. Features:
- URL + anon key inputs
- "Test Connection" button (queries tracks table)
- On success: calls `authStore.linkSupabase(url, key)` + `syncFromProfile()`
- Glass panel styling, overlay backdrop
- Close button (X)

Used by: ProfileTab and the "Link Supabase" button on empty homepage.

- [ ] **Step 2: Commit**

```bash
git add Frontend/src/components/ui/SupabaseLinkModal.jsx
git commit -m "feat: add Supabase credentials linking modal"
```

---

## Task 6: Profile Tab

**Files:**
- Create: `Frontend/src/components/ProfileTab.jsx`

- [ ] **Step 1: Create ProfileTab component**

Layout (top padding for nav clearance ~70px):
- "[username]'s Space" — already shown in nav area, so Profile just shows settings
- **Username section:** display current username, "Change" button → inline edit
- **Supabase Connection:** status indicator (linked/not linked), "Update Credentials" button → opens SupabaseLinkModal
- **Library:** "Re-scan Library" button → opens `showDirectoryPicker()`, starts ingest
- **Account:** "Sign Out" button → calls `authStore.signOut()`

All in glass panels, matching existing design system.

- [ ] **Step 2: Commit**

```bash
git add Frontend/src/components/ProfileTab.jsx
git commit -m "feat: add Profile tab component"
```

---

## Task 7: Wire up App.jsx — auth gate, nav changes, username header

**Files:**
- Modify: `Frontend/src/App.jsx`
- Delete: `Frontend/src/components/SetupScreen.jsx`

- [ ] **Step 1: Replace SetupScreen gate with auth gate**

In App.jsx `App()` function (lines 199-214):
- Import `useAuthStore`
- Replace `hasCredentials` check with auth session check
- Show `<AuthScreen />` if no session
- After login, sync BYOS credentials from profile to credentialStore + supabaseClient
- Show "Link Supabase" centred button on Discovery if no BYOS creds linked
- Show "Upload Library" button below "Link Supabase" if linked but not yet imported

- [ ] **Step 2: Add PROFILE to NAV_TABS**

Change line 196:
```javascript
const NAV_TABS = ['DISCOVERY', 'LIVE', 'PLAYLISTS', 'PROFILE'];
const NAV_TABS_SHORT = { DISCOVERY: 'DISCOVER', LIVE: 'LIVE', PLAYLISTS: 'LISTS', PROFILE: 'PROFILE' };
```

Add rendering for `activeTab === 'PROFILE'` → `<ProfileTab />`

- [ ] **Step 3: Add "[username]'s Space" top-centre header**

Below the floating nav pill, add a centred text element:
```
[username]'s Space
```
Style: subtle, small (fontSize 11), semi-transparent, JetBrains Mono, letter-spacing. Positioned top-centre, below the nav bar (about 60px from top). Visible on all pages.

- [ ] **Step 4: Show conditional buttons on Discovery page when empty**

When Discovery tab is active and user hasn't linked Supabase yet:
- Centre a "Link Supabase" glass button that opens SupabaseLinkModal

When linked but hasn't imported:
- Show "Upload Library" button that triggers `showDirectoryPicker()` → ingest flow
- On non-Chromium: show Toast "Local folder access requires Chrome, Edge, or Arc"

After first successful import: these buttons never appear again (check `profile.has_imported_library`).

- [ ] **Step 5: Delete SetupScreen.jsx**

```bash
git rm Frontend/src/components/SetupScreen.jsx
```

- [ ] **Step 6: Commit**

```bash
git add Frontend/src/App.jsx
git commit -m "feat: wire auth gate, Profile nav tab, username header, conditional onboarding"
```

---

## Task 8: Fix Live page layout — remove redundant header, add nav clearance

**Files:**
- Modify: `Frontend/src/components/LiveMode.jsx`

- [ ] **Step 1: Remove redundant "LIVE SESSION" label**

At line 982-984, remove the "LIVE SESSION" label div entirely. Keep just "SET TRACKER" / "CRATE BUILDER" as the header title (the nav already says LIVE).

- [ ] **Step 2: Add top padding for nav clearance**

The floating nav bar is ~50px tall and positioned at the top. The Live page content starts directly at the top with no clearance. Add `paddingTop: 60` (or similar) to the outer container so content doesn't hide behind the nav.

Look at the outer div at approximately line 968-971 and add appropriate top padding.

- [ ] **Step 3: Add conditional "Upload Library" on Live page**

If `!profile.has_imported_library`, show a subtle "Upload Library" button somewhere in the Live page header area. Once imported, it disappears forever.

- [ ] **Step 4: Commit**

```bash
git add Frontend/src/components/LiveMode.jsx
git commit -m "fix: remove redundant Live header, add nav clearance, conditional upload prompt"
```

---

## Task 9: Fix Playlist page — folder picker, re-scan, nav clearance

**Files:**
- Modify: `Frontend/src/components/playlist/PlaylistSidebar.jsx`
- Modify: `Frontend/src/components/playlist/PlaylistOrganiser.jsx`

- [ ] **Step 1: Replace text input with native folder picker**

In PlaylistSidebar.jsx (lines 640-686), replace the drag-drop zone + text input + START SCAN button with:
- A single button: "Re-scan Library" (if already imported) or "Import Music Folder" (if not)
- On click: call `showDirectoryPicker()` to get the folder handle
- Then call the existing `startIngest()` with the folder path
- On non-Chromium browsers: show Toast warning instead

Note: `showDirectoryPicker()` returns a `FileSystemDirectoryHandle`, not a path string. The backend `/ingest/start` expects a folder path. The handle gives us the folder name but not the full path. We need to keep the text input as a fallback for the backend ingest (which runs server-side and needs an absolute path).

**Revised approach:** The folder picker is for LOCAL AUDIO PLAYBACK (already working via localAudio.js). The backend ingest needs a server-side path. So:
- Keep the folder path input for server-side ingest (the backend scans files on the server's filesystem)
- But make it a proper input with a clear label, not the clunky drag-drop zone
- The `showDirectoryPicker()` is separate — it's for granting browser access to play audio locally

So the real fix is: simplify the ingest UI to just a clean text input + button, no drag-drop zone. Label it clearly. Change button text based on import state.

- [ ] **Step 2: Add top padding to PlaylistOrganiser**

Add `paddingTop: 60` to the outer container (line 270) so content clears the floating nav.

- [ ] **Step 3: Commit**

```bash
git add Frontend/src/components/playlist/PlaylistSidebar.jsx Frontend/src/components/playlist/PlaylistOrganiser.jsx
git commit -m "fix: simplify ingest UI, add nav clearance on playlists page"
```

---

## Task 10: Sync auth state with existing credential/supabase plumbing

**Files:**
- Modify: `Frontend/src/App.jsx`
- Modify: `Frontend/src/utils/supabaseClient.js`

- [ ] **Step 1: On auth state change, sync BYOS credentials**

When the auth store loads a profile with `supabase_url` + `supabase_key`, call `syncFromProfile(profile)` to populate localStorage. This keeps the existing `apiClient.js` header injection and `supabaseClient.js` dynamic client working without changes to those files.

- [ ] **Step 2: On logout, clear everything**

Call `clearCredentials()`, `resetSupabaseClient()`, `apiClient.clearCache()` — same as existing reconfigure flow.

- [ ] **Step 3: After successful ingest, mark imported**

After ingest completes successfully (in PlaylistSidebar or wherever ingest is triggered), call `authStore.setImportedLibrary()` to persist the flag.

- [ ] **Step 4: Commit**

```bash
git add Frontend/src/App.jsx Frontend/src/utils/supabaseClient.js
git commit -m "feat: sync auth profile with credential store and track import state"
```

---

## Task 11: Environment variables for central Supabase

**Files:**
- Modify: `Frontend/.env` or `.env.local` (if exists)
- Modify: Vercel environment variables

- [ ] **Step 1: Add env vars locally**

Create/update `.env.local`:
```
VITE_CENTRAL_SUPABASE_URL=<djmate-db project URL>
VITE_CENTRAL_SUPABASE_KEY=<djmate-db anon key>
```

- [ ] **Step 2: Add env vars on Vercel**

The deployer should add `VITE_CENTRAL_SUPABASE_URL` and `VITE_CENTRAL_SUPABASE_KEY` to Vercel environment variables for the frontend project.

- [ ] **Step 3: Commit .env.example (not actual secrets)**

```bash
echo "VITE_CENTRAL_SUPABASE_URL=\nVITE_CENTRAL_SUPABASE_KEY=" > Frontend/.env.example
git add Frontend/.env.example
git commit -m "docs: add env example for central Supabase credentials"
```

---

## Execution Order

Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

Tasks 3, 4, 5, 6 can be parallelised (independent components). Tasks 7-10 depend on the earlier tasks being complete.
