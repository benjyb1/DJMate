// Frontend/src/stores/authStore.js
import { create } from 'zustand';
import { centralSupabase, CENTRAL_URL, CENTRAL_KEY } from '../utils/centralSupabase';
import { syncFromProfile, getCredentials } from './credentialStore';

export const useAuthStore = create((set, get) => ({
  session: null,
  profile: null,
  loading: true,

  init: async () => {
    const { data: { session } } = await centralSupabase.auth.getSession();
    if (session) {
      const profile = await get().fetchProfile(session.user.id);
      // Sync BYOS credentials from the central DB profile into localStorage
      // so the user's linked project is available on any device.
      if (profile) {
        const synced = syncFromProfile(profile);
        if (!synced && getCredentials()) {
          // localStorage has creds but the profile doesn't — push them up
          // so the central DB stays in sync (e.g. user linked before this change).
          const local = getCredentials();
          get().linkSupabase(local.url, local.key).catch(() => {});
        }
      }
      set({ session, profile, loading: false });
    } else {
      set({ loading: false });
    }

    centralSupabase.auth.onAuthStateChange(async (_event, session) => {
      if (session) {
        const profile = await get().fetchProfile(session.user.id);
        if (profile) syncFromProfile(profile);
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
    const session = get().session;
    if (!session?.user?.id) throw new Error('Not signed in');

    // Direct fetch to PostgREST — bypasses the Supabase client's auth
    // middleware which can hang if the token refresh cycle is broken
    const resp = await fetch(
      `${CENTRAL_URL}/rest/v1/profiles?id=eq.${session.user.id}`,
      {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'apikey': CENTRAL_KEY,
          'Authorization': `Bearer ${session.access_token}`,
          'Prefer': 'return=representation',
        },
        body: JSON.stringify({ supabase_url: url, supabase_key: key }),
      },
    );

    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`Failed to save credentials (${resp.status}): ${body}`);
    }

    const rows = await resp.json();
    if (!rows?.length) throw new Error('Profile update affected no rows');

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
}));
