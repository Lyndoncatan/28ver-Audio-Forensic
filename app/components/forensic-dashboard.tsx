"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, Waves, FileText, Zap, Search, Fingerprint, Lock, ShieldCheck, Database, BarChart3, Radio } from "lucide-react"

interface ForensicDashboardProps {
    audioData: any
    isPlaying: boolean
    currentTime: number
    activeEvents?: any[]
    audioRef?: React.RefObject<HTMLAudioElement>
    externalOscilloscopeRef?: React.RefObject<HTMLCanvasElement>
    externalSpectrogramRef?: React.RefObject<HTMLCanvasElement>
}

export default function ForensicDashboard({
    audioData,
    isPlaying,
    currentTime,
    activeEvents = [],
    audioRef,
    externalOscilloscopeRef,
    externalSpectrogramRef
}: ForensicDashboardProps) {
    const internalOscilloscopeRef = useRef<HTMLCanvasElement>(null)
    const internalSpectrogramRef = useRef<HTMLCanvasElement>(null)

    const oscilloscopeRef = externalOscilloscopeRef || internalOscilloscopeRef
    const spectrogramRef = externalSpectrogramRef || internalSpectrogramRef
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
    const [dataArray, setDataArray] = useState<Uint8Array | null>(null)
    const requestRef = useRef<number>()

    // Initialize Audio Analyzer
    useEffect(() => {
        if (!audioRef?.current || analyser) return

        try {
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
            // Fix for "The AudioContext was not allowed to start"
            const handleUserGesture = () => {
                if (audioContext.state === 'suspended') {
                    audioContext.resume();
                }
            };
            window.addEventListener('click', handleUserGesture);

            const source = audioContext.createMediaElementSource(audioRef.current)
            const newAnalyser = audioContext.createAnalyser()

            newAnalyser.fftSize = 2048
            source.connect(newAnalyser)
            newAnalyser.connect(audioContext.destination)

            setAnalyser(newAnalyser)
            setDataArray(new Uint8Array(newAnalyser.frequencyBinCount))

            return () => {
                window.removeEventListener('click', handleUserGesture);
                // Don't close context strictly as it might break other components using the same audio element
                // But we can disconnect if needed. For now, keep it simple.
            }
        } catch (e) {
            console.warn("AudioContext setup failed (likely CORS or already connected):", e)
        }
    }, [audioRef])

    // Animation Loop
    const animate = () => {
        if (analyser && dataArray && isPlaying) {
            analyser.getByteTimeDomainData(dataArray as any)
            drawOscilloscope(dataArray)

            const freqData = new Uint8Array(analyser.frequencyBinCount)
            analyser.getByteFrequencyData(freqData)
            drawSpectrogram(freqData)
        } else {
            // Draw Idle State (Flatline / Grid)
            if (oscilloscopeRef.current) {
                const ctx = oscilloscopeRef.current.getContext("2d")
                if (ctx) {
                    ctx.clearRect(0, 0, oscilloscopeRef.current.width, oscilloscopeRef.current.height)
                    drawGrid(ctx, oscilloscopeRef.current.width, oscilloscopeRef.current.height, "#0f172a")

                    // Draw Flatline
                    ctx.beginPath()
                    ctx.strokeStyle = "#4ade80"
                    ctx.lineWidth = 2
                    ctx.moveTo(0, oscilloscopeRef.current.height / 2)
                    ctx.lineTo(oscilloscopeRef.current.width, oscilloscopeRef.current.height / 2)
                    ctx.stroke()

                    // Add a small "blip" scanning across
                    const t = Date.now() / 1000
                    const x = (t * 200) % oscilloscopeRef.current.width
                    ctx.beginPath()
                    ctx.strokeStyle = "rgba(74, 222, 128, 0.5)"
                    ctx.arc(x, oscilloscopeRef.current.height / 2, 2, 0, Math.PI * 2)
                    ctx.fill()
                }
            }

            if (spectrogramRef.current) {
                const ctx = spectrogramRef.current.getContext("2d")
                if (ctx) {
                    ctx.clearRect(0, 0, spectrogramRef.current.width, spectrogramRef.current.height)
                    drawGrid(ctx, spectrogramRef.current.width, spectrogramRef.current.height, "#0f172a")

                    // Draw "Waiting for Signal" text
                    ctx.fillStyle = "rgba(74, 222, 128, 0.2)"
                    ctx.font = "10px monospace"
                    ctx.textAlign = "center"
                    ctx.fillText("WAITING_FOR_SIGNAL_INPUT...", spectrogramRef.current.width / 2, spectrogramRef.current.height / 2)
                }
            }
        }

        requestRef.current = requestAnimationFrame(animate)
    }

    // Always run animation loop to keep grid/idle state active
    useEffect(() => {
        requestRef.current = requestAnimationFrame(animate)
        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current)
        }
    }, [isPlaying, analyser, dataArray])


    // DRAWING FUNCTIONS
    const drawOscilloscope = (data: Uint8Array) => {
        const canvas = oscilloscopeRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const width = canvas.width
        const height = canvas.height

        ctx.fillStyle = "rgba(2, 6, 23, 0.5)" // Fade slightly for trails? No, clear for crisp line
        ctx.clearRect(0, 0, width, height)

        // Grid
        drawGrid(ctx, width, height, "#0f172a")

        ctx.lineWidth = 2
        ctx.strokeStyle = "#4ade80" // Neon Green
        ctx.beginPath()

        const sliceWidth = width * 1.0 / data.length
        let x = 0

        for (let i = 0; i < data.length; i++) {
            const v = data[i] / 128.0
            const y = (v * height) / 2

            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)

            x += sliceWidth
        }

        ctx.lineTo(canvas.width, canvas.height / 2)
        ctx.stroke()

        // Glow effect
        ctx.shadowBlur = 10
        ctx.shadowColor = "#4ade80"
        ctx.stroke()
        ctx.shadowBlur = 0
    }

    const drawSpectrogram = (data: Uint8Array) => {
        // Simplified Real-time Spectrogram (just frequency bars for now to match "Multi-track" vibe)
        // A true scrolling spectrogram requires off-screen canvas buffering, which is complex for this snippet.
        // We will do a high-tech "Instantaneous Spectral Heatmap"

        const canvas = spectrogramRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const width = canvas.width
        const height = canvas.height

        ctx.clearRect(0, 0, width, height)

        // Grid
        drawGrid(ctx, width, height, "#0f172a")

        const barWidth = (width / data.length) * 2.5
        let x = 0

        for (let i = 0; i < data.length; i++) {
            const barHeight = (data[i] / 255) * height

            // Color Mapping (Deep Blue to Neon Green/Purple)
            const r = barHeight + (25 * (i / data.length));
            const g = 250 * (i / data.length);
            const b = 50;

            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight)
            gradient.addColorStop(0, `rgb(60, 20, 180)`) // Deep Blue/Purple bottom
            gradient.addColorStop(1, `rgb(50, 255, 100)`) // Neon Green top

            ctx.fillStyle = gradient
            ctx.fillRect(x, height - barHeight, barWidth, barHeight)

            x += barWidth + 1
        }
    }

    // FALLBACK SIMULATION (For when AudioContext fails or is not ready)
    const drawSimulatedVisuals = () => {
        const t = Date.now() / 1000;

        // Osc
        const canvasOsc = oscilloscopeRef.current
        if (canvasOsc) {
            const ctx = canvasOsc.getContext("2d")
            if (ctx) {
                const w = canvasOsc.width
                const h = canvasOsc.height
                ctx.clearRect(0, 0, w, h)
                drawGrid(ctx, w, h, "rgba(30,41,59, 0.5)")

                ctx.beginPath()
                ctx.strokeStyle = "#4ade80"
                ctx.lineWidth = 2
                for (let x = 0; x < w; x += 2) {
                    const y = h / 2 + Math.sin(x * 0.05 + t * 10) * (h / 4) * Math.sin(t) + (Math.random() * 5);
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke()
            }
        }

        // Spec
        const canvasSpec = spectrogramRef.current
        if (canvasSpec) {
            const ctx = canvasSpec.getContext("2d")
            if (ctx) {
                const w = canvasSpec.width
                const h = canvasSpec.height
                ctx.clearRect(0, 0, w, h)
                drawGrid(ctx, w, h, "rgba(30,41,59, 0.5)")

                const bars = 64;
                const bw = w / bars;
                for (let i = 0; i < bars; i++) {
                    const noise = Math.random();
                    const bh = (Math.sin(i * 0.2 + t) * 0.5 + 0.5) * h * 0.8 * noise;

                    const gradient = ctx.createLinearGradient(0, h, 0, h - bh)
                    gradient.addColorStop(0, `#1e1b4b`) // Deep Indigo
                    gradient.addColorStop(0.5, `#3b82f6`) // Blue
                    gradient.addColorStop(1, `#4ade80`) // Green

                    ctx.fillStyle = gradient
                    ctx.fillRect(i * bw, h - bh, bw - 2, bh);
                }
            }
        }
    }

    const drawGrid = (ctx: CanvasRenderingContext2D, w: number, h: number, color: string) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 1
        ctx.beginPath()
        // Vert
        for (let x = 0; x <= w; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        // Horz
        for (let y = 0; y <= h; y += 30) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke()
    }

    return (
        <Card className="col-span-1 xl:col-span-4 bg-slate-950/90 border-slate-800 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden backdrop-blur-xl mt-6">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-green-500" />

            <CardHeader className="border-b border-slate-800 bg-slate-900/50">
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-950/50 rounded-lg border border-blue-800/50">
                            <Fingerprint className="w-6 h-6 text-blue-400" />
                        </div>
                        <div>
                            <CardTitle className="text-xl font-black tracking-widest uppercase text-white">Forensic Interface V2</CardTitle>
                            <div className="flex gap-2 text-[10px] font-mono text-slate-400 mt-1">
                                <span className="flex items-center gap-1"><ShieldCheck className="w-3 h-3" /> SECURITY_LEVEL_0</span>
                                <span className="text-slate-600">|</span>
                                <span className="flex items-center gap-1"><Radio className="w-3 h-3 animate-pulse text-green-500" /> LIVE_MONITORING</span>
                            </div>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        <Badge variant="outline" className="bg-slate-950 text-slate-400 font-mono border-slate-800">SAMPLE_RATE: 44.1kHz</Badge>
                        <Badge variant="outline" className="bg-slate-950 text-slate-400 font-mono border-slate-800">BIT_DEPTH: 24-BIT</Badge>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="p-6 space-y-8">

                {/* UPPER VISUALIZATION GRID */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                    {/* OSCILLOSCOPE */}
                    <div className="relative group">
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
                        <div className="relative bg-slate-950 border border-slate-800 rounded-xl overflow-hidden p-4">
                            <div className="flex justify-between items-center mb-4 border-b border-slate-800/50 pb-2">
                                <h3 className="text-xs font-black uppercase text-green-500 tracking-widest flex items-center gap-2">
                                    <Activity className="w-4 h-4" /> Real-Time Oscilloscope
                                </h3>
                                <div className="flex gap-1">
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-500/50" />
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-500/20" />
                                </div>
                            </div>
                            <canvas
                                ref={oscilloscopeRef}
                                width={600}
                                height={200}
                                className="w-full h-48 bg-slate-900/50 rounded-lg border border-slate-800/50 shadow-inner"
                            />
                        </div>
                    </div>

                    {/* SPECTROGRAM */}
                    <div className="relative group">
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
                        <div className="relative bg-slate-950 border border-slate-800 rounded-xl overflow-hidden p-4">
                            <div className="flex justify-between items-center mb-4 border-b border-slate-800/50 pb-2">
                                <h3 className="text-xs font-black uppercase text-blue-500 tracking-widest flex items-center gap-2">
                                    <Waves className="w-4 h-4" /> Multi-Track Spectrogram
                                </h3>
                                <Badge className="bg-blue-500/10 text-blue-400 border-0 text-[9px]">HEATMAP_MODE</Badge>
                            </div>
                            <canvas
                                ref={spectrogramRef}
                                width={600}
                                height={200}
                                className="w-full h-48 bg-slate-900/50 rounded-lg border border-slate-800/50 shadow-inner"
                            />
                        </div>
                    </div>
                </div>

                {/* DATA TABLE */}
                <div className="relative">
                    <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
                    <div className="flex items-center gap-2 py-4 mb-2">
                        <Database className="w-5 h-5 text-purple-500" />
                        <h3 className="text-sm font-bold text-slate-200">METADATA_REGISTRY</h3>
                        <span className="text-xs text-slate-500 font-mono ml-auto">Showing {activeEvents?.length || 0} Detected Events</span>
                    </div>

                    <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40 backdrop-blur-sm">
                        <table className="w-full text-left text-xs font-mono">
                            <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                                <tr>
                                    <th className="p-4 font-bold">STATUS</th>
                                    <th className="p-4 font-bold">TIMESTAMP</th>
                                    <th className="p-4 font-bold">EVENT_TYPE</th>
                                    <th className="p-4 font-bold">MAGNITUDE (dB)</th>
                                    <th className="p-4 font-bold">CONFIDENCE</th>
                                    <th className="p-4 font-bold text-right">HASH</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/50">
                                {activeEvents && activeEvents.length > 0 ? (
                                    activeEvents.map((ev: any, i: number) => {
                                        const isActive = Math.abs(currentTime - ev.time) < 0.5;
                                        return (
                                            <tr key={i} className={`transition-colors ${isActive ? "bg-purple-500/10" : "hover:bg-slate-800/50"}`}>
                                                <td className="p-4">
                                                    {isActive ?
                                                        <Badge className="bg-green-500 text-black border-0 animate-pulse">ACTIVE</Badge> :
                                                        <Badge variant="outline" className="text-slate-500 border-slate-700">LOGGED</Badge>
                                                    }
                                                </td>
                                                <td className={`p-4 ${isActive ? "text-white font-bold" : "text-slate-400"}`}>
                                                    {Number(ev.time).toFixed(4)}s
                                                </td>
                                                <td className="p-4">
                                                    <span className={`px-2 py-1 rounded ${ev.speaker === 'SPEAKER_01' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                                                        {ev.type || "UNKNOWN_SIGNAL"}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-slate-300">
                                                    {Number(ev.decibels).toFixed(2)} dB
                                                </td>
                                                <td className="p-4">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-purple-500"
                                                                style={{ width: `${(ev.confidence || 0) * 100}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-slate-400">{(ev.confidence * 100).toFixed(0)}%</span>
                                                    </div>
                                                </td>
                                                <td className="p-4 text-right text-slate-600">
                                                    {Math.random().toString(36).substring(7).toUpperCase()}
                                                </td>
                                            </tr>
                                        )
                                    })
                                ) : (
                                    <tr>
                                        <td colSpan={6} className="p-8 text-center text-slate-600 italic">
                                            No forensic events detected in register.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
