import React, { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, Volume2 } from 'lucide-react'

interface TrackProps {
  url: string;
  label: string;
  color: string;
}

const ForensicWaveform = ({ url, label, color }: TrackProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurfer = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    if (!containerRef.current) return

    // Initialize WaveSurfer for this specific track
    wavesurfer.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4b5563', // Gray-600
      progressColor: color, // Custom color for each stem
      cursorColor: '#ffffff',
      barWidth: 2,
      height: 60,
      url: url,
    })

    wavesurfer.current.on('play', () => setIsPlaying(true))
    wavesurfer.current.on('pause', () => setIsPlaying(false))

    return () => wavesurfer.current?.destroy()
  }, [url, color])

  return (
    <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700 mb-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">{label}</span>
        <div className="flex items-center gap-3">
            <button 
                onClick={() => wavesurfer.current?.playPause()}
                className="p-2 bg-indigo-600 rounded-full hover:bg-indigo-500 transition-colors"
            >
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <Volume2 size={16} className="text-slate-500" />
        </div>
      </div>
      <div ref={containerRef} />
    </div>
  )
}

export default ForensicWaveform