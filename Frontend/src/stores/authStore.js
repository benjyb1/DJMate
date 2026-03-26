// Frontend/src/stores/authStore.js
import { create } from 'zustand';
import { centralSupabase } from '../utils/centralSupabase';

export const useAuthStore = create((set, get) => ({
  session: null,
  profile: null,
  loading: true,

  init: async () => {
    const { data: { session } } = await centralSupabase.auth.getSession();
    if (session) {
      const profile = await get().fetchProfile(session.user.id);
      set({ session, profile, loading: false });
    } else {
      set({ loading: false });
    }

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
}));
