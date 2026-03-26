// Frontend/src/utils/centralSupabase.js
import { createClient } from '@supabase/supabase-js';

export const CENTRAL_URL = import.meta.env.VITE_CENTRAL_SUPABASE_URL || 'https://cvermotfxamubejfnoje.supabase.co';
export const CENTRAL_KEY = import.meta.env.VITE_CENTRAL_SUPABASE_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag';

export const centralSupabase = createClient(CENTRAL_URL, CENTRAL_KEY);
