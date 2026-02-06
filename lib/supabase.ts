import { createClient } from "@supabase/supabase-js"

// Get environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// Create Supabase client
export const supabase = supabaseUrl && supabaseKey ? createClient(supabaseUrl, supabaseKey) : null

// Database types
export interface AudioAnalysisRecord {
  id?: string
  filename: string
  duration: number
  sample_rate: number
  average_rms: number
  detected_sounds: number
  dominant_frequency: number
  max_decibels: number
  sound_events: any[]
  frequency_spectrum: any[]
  analysis_results: any
  created_at?: string
  user_id?: string
}

export interface AnalysisSession {
  id?: string
  session_name: string
  audio_file_url: string
  analysis_data: AudioAnalysisRecord
  created_at?: string
  user_id?: string
}

// Helper function to check if Supabase is available
export const isSupabaseAvailable = (): boolean => {
  return !!(supabaseUrl && supabaseKey && supabase)
}

// Helper function to get connection status
export const getSupabaseStatus = (): { connected: boolean; url?: string; error?: string } => {
  if (!supabaseUrl) {
    return {
      connected: false,
      error: "NEXT_PUBLIC_SUPABASE_URL not found in environment variables",
    }
  }

  if (!supabaseKey) {
    return {
      connected: false,
      error: "NEXT_PUBLIC_SUPABASE_ANON_KEY not found in environment variables",
    }
  }

  if (!supabase) {
    return {
      connected: false,
      error: "Failed to create Supabase client",
    }
  }

  return {
    connected: true,
    url: supabaseUrl,
  }
}
