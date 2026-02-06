-- ================================
-- Audio Forensic Detector Database Setup
-- Run this script in your Supabase SQL Editor
-- ================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create audio_analyses table
CREATE TABLE IF NOT EXISTS public.audio_analyses (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    filename TEXT NOT NULL,
    duration REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    average_rms REAL NOT NULL,
    detected_sounds INTEGER NOT NULL,
    dominant_frequency REAL NOT NULL,
    max_decibels REAL NOT NULL,
    sound_events JSONB NOT NULL DEFAULT '[]',
    frequency_spectrum JSONB NOT NULL DEFAULT '[]',
    analysis_results JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID DEFAULT NULL
);

-- Create analysis_sessions table
CREATE TABLE IF NOT EXISTS public.analysis_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_name TEXT NOT NULL,
    audio_file_url TEXT,
    analysis_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID DEFAULT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_audio_analyses_created_at ON public.audio_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audio_analyses_filename ON public.audio_analyses(filename);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON public.analysis_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_name ON public.analysis_sessions(session_name);

-- Enable Row Level Security (RLS) - Optional for public access
ALTER TABLE public.audio_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_sessions ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (since you don't have auth yet)
-- These allow anyone to read/write - you can restrict later when you add authentication

DROP POLICY IF EXISTS "Allow public access to audio_analyses" ON public.audio_analyses;
CREATE POLICY "Allow public access to audio_analyses" ON public.audio_analyses
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow public access to analysis_sessions" ON public.analysis_sessions;
CREATE POLICY "Allow public access to analysis_sessions" ON public.analysis_sessions
    FOR ALL USING (true) WITH CHECK (true);

-- Insert some sample data for testing (optional)
INSERT INTO public.audio_analyses (
    filename, 
    duration, 
    sample_rate, 
    average_rms, 
    detected_sounds, 
    dominant_frequency, 
    max_decibels,
    sound_events,
    frequency_spectrum,
    analysis_results
) VALUES (
    'sample_audio.wav',
    10.5,
    44100,
    0.0234,
    5,
    440.0,
    -12.3,
    '[{"time": 1.2, "type": "Voice", "frequency": 440, "amplitude": 0.8, "decibels": -10}]'::jsonb,
    '[{"frequency": 440, "magnitude": 0.8}]'::jsonb,
    '{"analysisType": "sample", "timestamp": "2024-01-01T00:00:00Z"}'::jsonb
) ON CONFLICT DO NOTHING;

-- Verify tables were created
SELECT 
    table_name, 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' 
    AND table_name IN ('audio_analyses', 'analysis_sessions')
ORDER BY table_name, ordinal_position;

-- Show table counts
SELECT 
    'audio_analyses' as table_name, 
    COUNT(*) as record_count 
FROM public.audio_analyses
UNION ALL
SELECT 
    'analysis_sessions' as table_name, 
    COUNT(*) as record_count 
FROM public.analysis_sessions;
