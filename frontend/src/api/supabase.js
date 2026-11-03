import { createClient } from '@supabase/supabase-js'

const rawUrl = import.meta.env.VITE_SUPABASE_URL;
const rawKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isSupabaseConfigured = Boolean(
  rawUrl && rawUrl.startsWith("http") && !rawUrl.includes("YOUR_SUPABASE_URL_HERE")
);

// Fallback dummy credentials when Supabase is not configured
const supabaseUrl = isSupabaseConfigured ? rawUrl : 'https://dummy.supabase.co';
const supabaseAnonKey = isSupabaseConfigured ? rawKey : 'dummy-key';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: isSupabaseConfigured,
    autoRefreshToken: isSupabaseConfigured,
  }
});
