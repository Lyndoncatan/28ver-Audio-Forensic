-- Create audio_analyses table
CREATE TABLE IF NOT EXISTS audio_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
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
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Create analysis_sessions table
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_name TEXT NOT NULL,
    audio_file_url TEXT,
    analysis_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_audio_analyses_created_at ON audio_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audio_analyses_user_id ON audio_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON analysis_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON analysis_sessions(user_id);

-- Enable Row Level Security (RLS)
ALTER TABLE audio_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own audio analyses" ON audio_analyses
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert their own audio analyses" ON audio_analyses
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own audio analyses" ON audio_analyses
    FOR UPDATE USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can delete their own audio analyses" ON audio_analyses
    FOR DELETE USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can view their own analysis sessions" ON analysis_sessions
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert their own analysis sessions" ON analysis_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own analysis sessions" ON analysis_sessions
    FOR UPDATE USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can delete their own analysis sessions" ON analysis_sessions
    FOR DELETE USING (auth.uid() = user_id OR user_id IS NULL);
