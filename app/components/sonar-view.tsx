"use client"

import { Card } from "@/components/ui/card"
import { useEffect, useRef, useCallback, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import WaveSurfer from "wavesurfer.js"
import {
  Target, Play, Pause, Layers, Activity,
  FileText, ChevronRight, Scissors, Loader2,
  Download, Mic2, Wind, Database, Bird, Car, Footprints, AudioWaveform,
  Waves, Volume2, VolumeX, Bomb, Hammer, AlertTriangle, Megaphone, Zap
} from "lucide-react"

// --- FORENSIC TRACK COMPONENT ---
function ForensicTrack({ url, label, color, icon: Icon, masterPlaying, masterTime, stats }: any) {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveSurferRef = useRef<WaveSurfer | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [isLocalPlaying, setIsLocalPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)

  useEffect(() => {
    if (!containerRef.current || !url) return
    if (waveSurferRef.current) waveSurferRef.current.destroy();

    waveSurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#1e293b",
      progressColor: color,
      cursorColor: "#ffffff",
      barWidth: 2,
      barGap: 3,
      height: 60,
      url: url,
    })

    waveSurferRef.current.on("ready", () => setIsReady(true))
    waveSurferRef.current.on("play", () => setIsLocalPlaying(true))
    waveSurferRef.current.on("pause", () => setIsLocalPlaying(false))

    return () => waveSurferRef.current?.destroy()
  }, [url, color])

  useEffect(() => {
    if (!waveSurferRef.current || !isReady) return
    if (masterPlaying && !waveSurferRef.current.isPlaying()) waveSurferRef.current.play()
    else if (!masterPlaying && waveSurferRef.current.isPlaying()) waveSurferRef.current.pause()

    const wsTime = waveSurferRef.current.getCurrentTime()
    if (Math.abs(wsTime - masterTime) > 0.1) waveSurferRef.current.setTime(masterTime)
  }, [masterPlaying, masterTime, isReady])

  const toggleLocalPlay = () => waveSurferRef.current?.playPause()
  const toggleMute = () => {
    if (waveSurferRef.current) {
      const newMuteState = !isMuted
      waveSurferRef.current.setMuted(newMuteState)
      setIsMuted(newMuteState)
    }
  }

  return (
    <Card className={`bg-slate-950/50 border-slate-800 p-6 transition-all hover:border-slate-600 group relative overflow-hidden ${isMuted ? 'opacity-40' : 'opacity-100'}`}>
      <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: color }} />
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 shadow-xl group-hover:scale-105 transition-transform">
            <Icon size={20} style={{ color }} />
          </div>
          <div>
            <h4 className="text-[12px] font-black uppercase tracking-widest text-slate-100 italic leading-none">{label}</h4>
            <div className="flex items-center gap-2 mt-1">
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: color }} />
              <span className="text-[9px] text-slate-500 font-mono tracking-tighter uppercase">
                {url ? "Signal_Isolated" : "Waiting..."}
              </span>
            </div>
            {/* NEW: Stats Display */}
            {stats && (
              <div className="flex gap-2 mt-2">
                <Badge variant="outline" className="text-[8px] h-4 px-1 text-slate-400 border-slate-800 bg-slate-900/50">
                  {stats.confidence}% CONF
                </Badge>
                <Badge variant="outline" className="text-[8px] h-4 px-1 text-slate-400 border-slate-800 bg-slate-900/50">
                  {stats.db} dB
                </Badge>
                <Badge variant="outline" className="text-[8px] h-4 px-1 text-slate-400 border-slate-800 bg-slate-900/50">
                  {stats.dist}m DIST
                </Badge>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={toggleMute} variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800">
            {isMuted ? <VolumeX size={16} className="text-red-500" /> : <Volume2 size={16} className="text-slate-400" />}
          </Button>
          <Button onClick={toggleLocalPlay} variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800">
            {isLocalPlaying ? <Pause size={16} className="text-white" /> : <Play size={16} className="text-white ml-0.5" />}
          </Button>
          {url && (
            <Button variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800" asChild>
              <a href={url} download><Download size={16} className="text-slate-400" /></a>
            </Button>
          )}
        </div>
      </div>
      <div ref={containerRef} className="opacity-80 group-hover:opacity-100 transition-opacity cursor-pointer mt-2" />
    </Card>
  )
}

// --- MAIN SONAR VIEW ---
export default function SonarView({
  audioData,
  setAudioData, // New Prop
  liveEvents = [],
  isRecording = false,
  currentStems,
  setCurrentStems,
  showStems,
  setShowStems
}: any) {
  const canvas2DRef = useRef<HTMLCanvasElement>(null)
  const canvas3DRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [rotation, setRotation] = useState({ x: 0.5, y: 0.5 })
  const [currentTime, setCurrentTime] = useState(0)
  const [isSeparating, setIsSeparating] = useState(false)

  // Interaction State
  const [hoveredEvent, setHoveredEvent] = useState<any | null>(null);
  const mousePos = useRef({ x: 0, y: 0 });

  // 3D Interaction State
  const mousePos3D = useRef({ x: 0, y: 0 });
  const [hoveredVoxel, setHoveredVoxel] = useState<any | null>(null);

  const scanAngle = useRef(0)
  // Use the classification from the analysis result if available
  const activeEvents = isRecording ? liveEvents : (audioData?.analysisResults?.soundEvents || [])

  const handleMouseMove2D = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    mousePos.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleMouseMove3D = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    mousePos3D.current = { x, y };

    // Orbit Control (Left Click Drag)
    if (e.buttons === 1) {
      setRotation(r => ({
        x: r.x + e.movementY * 0.005,
        y: r.y + e.movementX * 0.005
      }));
    }
  };

  // DEBUG: Check why stats are missing
  useEffect(() => {
    // console.log("[SonarView] AudioData:", audioData);
    // console.log("[SonarView] ActiveEvents:", activeEvents);
  }, [audioData, activeEvents]);

  const jumpToTime = useCallback((time: any) => {
    const safeTime = typeof time === 'number' ? time : 0;
    if (audioRef.current) {
      audioRef.current.currentTime = safeTime
      audioRef.current.play()
      setIsPlaying(true)
    }
  }, [])

  // HELPER: Convert file to Base64 to match your API route requirements
  const fileToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        resolve(base64String);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const handleDeconstruct = async () => {
    if (!audioData?.url) return;
    setIsSeparating(true);
    try {
      // 1. Fetch the blob from the local URL
      const responseBlob = await fetch(audioData.url);
      const audioBlob = await responseBlob.blob();

      // 2. Convert to Base64 (API expects JSON body, not FormData)
      const base64Data = await fileToBase64(audioBlob);
      const safeFileName = (audioData.name || "audio.wav").replace(/[^a-z0-9.]/gi, "_").toLowerCase();

      // 3. POST as JSON to match your route.ts
      const response = await fetch("/api/classify-audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audioData: base64Data,
          filename: safeFileName
        })
      });

      const result = await response.json();

      // 4. Update UI with the job-specific stems AND Classification
      if (result.status === "Success" || result.stems) {
        if (setCurrentStems) setCurrentStems(result.stems);
        if (setShowStems) setShowStems(true);

        // CRITICAL FIX: Update the parent audioData with the fresh classification events
        if (setAudioData && result.classification) {
          setAudioData((prev: any) => ({
            ...prev,
            analysisResults: result.classification
          }));
        }
      } else {
        throw new Error(result.error || "Extraction failed");
      }
    } catch (error: any) {
      console.error("Deconstruct Error:", error);
      alert(`Forensic Engine Error: ${error.message}`);
    } finally {
      setIsSeparating(false);
    }
  };

  const draw2DRadar = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const cx = width / 2;
    const cy = height / 2;
    const maxR = Math.min(width, height) / 2 - 40;

    // Clear & Background
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    // Draw Grid (Circles)
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      ctx.strokeStyle = i === 5 ? "rgba(99, 102, 241, 0.5)" : "rgba(34, 197, 94, 0.1)";
      if (i === 5) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(cx, cy, (maxR / 5) * i, 0, Math.PI * 2);
      ctx.stroke();

      // Distance Labels on Grid
      if (i < 5) {
        ctx.fillStyle = "rgba(34, 197, 94, 0.3)";
        ctx.font = "9px monospace";
        ctx.fillText(`${(100 - i * 20)}m`, cx + 2, cy - (maxR / 5) * i - 2);
      }
    }

    // Draw Grid (Lines)
    ctx.strokeStyle = "rgba(34, 197, 94, 0.05)";
    ctx.setLineDash([]);
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
      ctx.stroke();
    }

    // Scanning Effect
    scanAngle.current = (scanAngle.current + 0.02) % (Math.PI * 2);
    const gradient = ctx.createConicGradient(scanAngle.current, cx, cy);
    gradient.addColorStop(0, "rgba(34, 197, 94, 0)");
    gradient.addColorStop(0.1, "rgba(34, 197, 94, 0.3)"); // Stronger scan
    gradient.addColorStop(0.2, "rgba(34, 197, 94, 0)");

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
    ctx.fill();

    // Scan Line
    ctx.strokeStyle = "#4ade80";
    ctx.lineWidth = 2;
    ctx.shadowBlur = 15;
    ctx.shadowColor = "#4ade80";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(scanAngle.current + 0.2) * maxR, cy + Math.sin(scanAngle.current + 0.2) * maxR);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Detect Hover
    let foundHover: any = null;

    // Draw Events
    activeEvents.forEach((ev: any) => {
      // Map Time to Angle (Clockwise from top)
      const duration = audioData?.analysisResults?.duration || 1;
      const normalizedTime = (ev.time || 0) / duration;
      const a = normalizedTime * Math.PI * 2 - Math.PI / 2;

      // Map dB to Distance (Louder = Closer)
      // -10dB (close) -> 10m
      // -60dB (far) -> 60m+
      const db = Number(ev.decibels || -60);
      // Normalize dB: -90...0 -> 0...1
      const normalizedDb = Math.max(0, Math.min(1, (db + 90) / 90));
      // Using "Louder is Closer" logic as requested "middle is the device who receives"
      const radiusFactor = 1.0 - normalizedDb;
      const d = (radiusFactor * maxR * 0.9) + (maxR * 0.1); // min 10% radius buffer

      const x = cx + Math.cos(a) * d;
      const y = cy + Math.sin(a) * d;

      const isActive = Math.abs(currentTime - (ev.time || 0)) < 0.3;
      // Check Hover
      const mh = mousePos.current;
      const distToMouse = Math.sqrt((mh.x - x) ** 2 + (mh.y - y) ** 2);
      const isHovered = distToMouse < 15;
      if (isHovered) foundHover = ev;

      const baseColor = ev.speaker === "SPEAKER_01" ? "#ef4444" : "#3b82f6";
      const color = isActive ? "#ffffff" : baseColor;

      // --- VISUALIZATION ---

      // 1. Spreading Ripple Effect (when active)
      if (isActive) {
        const pulse = (Date.now() / 1000) % 1; // 0 to 1
        ctx.beginPath();
        ctx.strokeStyle = baseColor;
        ctx.globalAlpha = 1 - pulse;
        ctx.lineWidth = 2;
        const rippleR = 5 + (pulse * 30); // Expand to 35px
        ctx.arc(x, y, rippleR, 0, Math.PI * 2);
        ctx.stroke();

        // Second ripple
        const pulse2 = ((Date.now() / 1000) + 0.5) % 1;
        ctx.globalAlpha = 1 - pulse2;
        const rippleR2 = 5 + (pulse2 * 30);
        ctx.arc(x, y, rippleR2, 0, Math.PI * 2);
        ctx.stroke();

        ctx.globalAlpha = 1.0;
      }

      // 2. The Point
      ctx.beginPath();
      ctx.arc(x, y, isActive ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = color;

      if (isActive || isHovered) {
        ctx.shadowBlur = 15;
        ctx.shadowColor = baseColor;
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      // 3. Labels
      if (isActive || isHovered) {
        // Draw connecting line to center if active
        if (isActive) {
          ctx.beginPath();
          ctx.strokeStyle = baseColor;
          ctx.globalAlpha = 0.2;
          ctx.moveTo(cx, cy);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }

        // Text Label
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "left";
        const labelText = `${ev.type || "SIGNAL"}`;
        ctx.fillText(labelText, x + 12, y);

        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = "10px monospace";
        const subLabel = `${Math.abs(db).toFixed(0)}dB | ${((ev.confidence || 0) * 100).toFixed(0)}%`;
        ctx.fillText(subLabel, x + 12, y + 12);
      }
    });

    setHoveredEvent(foundHover);

  }, [audioData, activeEvents, currentTime, isPlaying])

  const draw3DVoxel = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Clear
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    // Grid Gradient
    const cx = width / 2;
    const cy = height / 2;
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, width / 1.5);
    grd.addColorStop(0, "rgba(2, 6, 23, 0)");
    grd.addColorStop(1, "rgba(2, 6, 23, 1)");

    // 3D Projection Helper
    const project = (x: number, y: number, z: number) => {
      const scale = 400 / (400 + z);
      const x2D = width / 2 + x * scale;
      const y2D = height / 2.5 + y * scale;
      return { x: x2D, y: y2D, s: scale, zIndex: z }
    }

    const rx = rotation.x;
    const ry = rotation.y;

    // Draw Floor Grid
    ctx.strokeStyle = "rgba(99, 102, 241, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = -200; i <= 200; i += 40) {
      // Horizontal Lines
      let p1 = project(i * Math.cos(ry) - -200 * Math.sin(ry), 100, i * Math.sin(ry) + -200 * Math.cos(ry));
      let p2 = project(i * Math.cos(ry) - 200 * Math.sin(ry), 100, i * Math.sin(ry) + 200 * Math.cos(ry));
      if (p1.s > 0 && p2.s > 0) { ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); }

      // Vertical Lines
      let p3 = project(-200 * Math.cos(ry) - i * Math.sin(ry), 100, -200 * Math.sin(ry) + i * Math.cos(ry));
      let p4 = project(200 * Math.cos(ry) - i * Math.sin(ry), 100, 200 * Math.sin(ry) + i * Math.cos(ry));
      if (p3.s > 0 && p4.s > 0) { ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y); }
    }
    ctx.stroke();

    // Hit Testing Variables
    let bestDist = 20; // Hit radius
    let foundVoxel: any = null;
    const mx = mousePos3D.current.x;
    const my = mousePos3D.current.y;

    // Sort events by Z-index (Back to Front) for correct occlusion
    const renderList: any[] = [];

    activeEvents.forEach((ev: any) => {
      const duration = (audioData?.analysisResults?.duration || 1)

      // Calculate 3D Position
      const xRaw = (((ev.time || 0) / duration) * 400) - 200;

      // Z-Depth based on dB (Louder = Closer to center/camera?)
      // Let's map dB to 'depth' in the scene.
      // -10dB -> Close (z=-100), -90dB -> Far (z=100)
      const db = Number(ev.decibels || -60);
      const normalizedDb = Math.max(0, Math.min(1, (db + 90) / 90)); // 0..1
      const zRaw = ((1 - normalizedDb) * 300) - 150;

      // Apply Rotation
      const x = xRaw * Math.cos(ry) - zRaw * Math.sin(ry);
      const z = xRaw * Math.sin(ry) + zRaw * Math.cos(ry);
      const y = 100; // Floor level

      // Height
      const confidence = ev.confidence || 0.5;
      const h = Math.max(20, confidence * 150);

      // Project Base and Top
      const projBase = project(x, y, z);
      const projTop = project(x, y - h, z);

      if (projBase.s > 0) {
        renderList.push({ ev, projBase, projTop, zIndex: z, h, isActive: Math.abs(currentTime - (ev.time || 0)) < 0.3 });
      }
    });

    // Back-to-front painter's algorithm
    renderList.sort((a, b) => b.zIndex - a.zIndex);

    renderList.forEach((item) => {
      const { ev, projBase, projTop, isActive } = item;

      // Check Hover
      // Simple check: distance to the vertical line of the voxel
      const dx = mx - projBase.x;
      const dy = my - (projBase.y + projTop.y) / 2; // Midpoint
      // Or check against line segment... simplified to point dist for now
      const dist = Math.sqrt((mx - projTop.x) ** 2 + (my - projTop.y) ** 2);

      const isHovered = dist < 20;
      if (isHovered && dist < bestDist) {
        bestDist = dist;
        foundVoxel = ev;
      }
      // If this specific item matches the tracked hovered state
      const isSelected = hoveredVoxel && hoveredVoxel === ev;

      const color = ev.speaker === 'SPEAKER_01' ? '#ef4444' : '#3b82f6'; // Red vs Blue

      ctx.lineWidth = (isActive || isSelected) ? 3 : 1;
      ctx.strokeStyle = (isActive || isSelected) ? "#ffffff" : color;

      // Draw Pillar
      ctx.beginPath();
      ctx.moveTo(projBase.x, projBase.y);
      ctx.lineTo(projTop.x, projTop.y);
      ctx.stroke();

      // Draw Top Point / Head
      ctx.fillStyle = (isActive || isSelected) ? "#ffffff" : color;
      ctx.beginPath();
      ctx.arc(projTop.x, projTop.y, projTop.s * (isActive ? 6 : 3), 0, Math.PI * 2);
      ctx.fill();

      // Effect: Floor Ripple
      if (isActive) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = color;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.ellipse(projBase.x, projBase.y, projBase.s * 15, projBase.s * 8, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Label 3D
      if (isActive || isSelected) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px monospace";
        ctx.textAlign = "center";
        const label = `${ev.type} ${Math.abs(Number(ev.decibels)).toFixed(0)}dB`;
        ctx.fillText(label, projTop.x, projTop.y - 12);
      }
    });

    setHoveredVoxel(foundVoxel);

    // Vignette Overlay
    ctx.fillStyle = grd;
    ctx.globalCompositeOperation = "multiply"; // Darken edges
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = "source-over";

  }, [audioData, activeEvents, rotation, currentTime, hoveredVoxel])

  // RENDER LOOP
  useEffect(() => {
    let animationFrameId: number;
    const render = () => {
      if (canvas2DRef.current) {
        draw2DRadar(canvas2DRef.current.getContext("2d")!, canvas2DRef.current.width, canvas2DRef.current.height);
      }
      if (canvas3DRef.current) {
        draw3DVoxel(canvas3DRef.current.getContext("2d")!, canvas3DRef.current.width, canvas3DRef.current.height);
      }
      animationFrameId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [draw2DRadar, draw3DVoxel]);

  // HELPER: Aggregate Stats per Category
  const getStats = (keywords: string[]) => {
    const matches = activeEvents.filter((e: any) => keywords.some(k => (e.type || "").toLowerCase().includes(k.toLowerCase())));
    if (matches.length === 0) return null;

    // Max Confidence
    const maxConf = Math.max(...matches.map((e: any) => e.confidence || 0));
    // Avg dB
    const avgDb = matches.reduce((acc: number, e: any) => acc + Number(e.decibels || -60), 0) / matches.length;
    // Avg Distance
    const avgDist = Math.abs(avgDb); // Simple approx

    return {
      confidence: (maxConf * 100).toFixed(0),
      db: avgDb.toFixed(1),
      dist: avgDist.toFixed(1)
    }
  }

  return (
    <div className="w-full max-w-[1600px] mx-auto p-6 space-y-6 bg-[#020617] text-slate-100 min-h-screen font-mono">
      {/* HEADER SECTION */}
      <div className="flex flex-col lg:flex-row justify-between items-center gap-6 border-b border-slate-800 pb-8">
        <div className="flex items-center gap-4">
          <Target className="w-12 h-12 text-blue-500" />
          <div>
            <h1 className="text-4xl font-black italic tracking-tighter uppercase leading-none">Forensic Sonar V4</h1>
            <Badge variant="outline" className="text-green-500 border-green-500/30 mt-2 tracking-widest font-black uppercase">System_Active</Badge>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Button onClick={handleDeconstruct} disabled={isSeparating} className="bg-indigo-600 hover:bg-indigo-500 font-bold h-20 px-10 rounded-xl transition-all shadow-[0_0_20px_rgba(79,70,229,0.3)]">
            {isSeparating ? (
              <div className="flex flex-col items-center">
                <Loader2 className="animate-spin mb-1" />
                <span className="text-[10px] tracking-widest uppercase">Isolating_Signals...</span>
              </div>
            ) : (
              <><Scissors className="mr-3" /> DECONSTRUCT AUDIO</>
            )}
          </Button>
          <div className="flex items-center gap-4 bg-slate-900/80 p-3 rounded-2xl border border-slate-700">
            <audio ref={audioRef} src={audioData?.url} onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)} />
            <Button onClick={() => { isPlaying ? audioRef.current?.pause() : audioRef.current?.play(); setIsPlaying(!isPlaying); }} className="rounded-full w-14 h-14 bg-blue-600">
              {isPlaying ? <Pause /> : <Play className="ml-1" />}
            </Button>
            <div className="px-5 border-l border-slate-700 text-3xl font-black tabular-nums text-blue-400">
              {Number(currentTime || 0).toFixed(2)}s
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        <div className="xl:col-span-3 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-950/80 border-slate-800 overflow-hidden relative">
            <Badge className="absolute top-4 left-4 bg-slate-900/90 text-green-400 z-10 font-black">2D_SPATIAL_MAP</Badge>
            <canvas ref={canvas2DRef} width={800} height={600} className="w-full aspect-square" onMouseMove={handleMouseMove2D} />
          </Card>
          <Card className="bg-slate-950/80 border-slate-800 overflow-hidden relative">
            <Badge className="absolute top-4 left-4 bg-slate-900/90 text-blue-400 z-10 font-black">3D_TOPOGRAPHY</Badge>
            <canvas ref={canvas3DRef} width={800} height={600} className="w-full aspect-square" onMouseMove={handleMouseMove3D} />
          </Card>
        </div>

        <Card className="bg-slate-950/80 border-slate-800 flex flex-col h-full max-h-[600px]">
          <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
            <h2 className="text-[10px] font-black uppercase flex items-center gap-2 tracking-widest"><FileText className="w-4 h-4 text-blue-500" /> Signal Matrix</h2>
            {showStems && <Badge className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[8px] animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.2)]">LIVE_STEM_SYNC</Badge>}
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {activeEvents.map((ev: any, i: number) => {
              const isActive = Math.abs(currentTime - (ev.time || 0)) < 0.3;
              const db = Number(ev.decibels || -60).toFixed(1);
              const confidence = ((ev.confidence || 0) * 100).toFixed(0);
              // Estimate distance: -10db = 10m, -90db = 90m (Linear approx for viz)
              const distance = Math.abs(Number(db)).toFixed(1);

              return (
                <div
                  key={i}
                  onClick={() => jumpToTime(ev.time)}
                  className={`relative flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all border-l-2 bg-slate-900/40 group hover:bg-slate-800 ${isActive
                    ? `border-l-4 shadow-[0_0_20px_rgba(0,0,0,0.5)] scale-[1.02] z-10 ${ev.speaker === 'SPEAKER_01' ? 'border-red-500 bg-red-950/20' : 'border-blue-500 bg-blue-950/20'}`
                    : 'border-transparent opacity-70 hover:opacity-100'
                    }`}
                >
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-sm ${isActive ? "bg-current animate-ping" : "bg-slate-600"}`} />
                      <span className={`text-[10px] tabular-nums font-mono ${isActive ? "text-white font-bold" : "text-slate-500"}`}>{Number(ev?.time || 0).toFixed(3)}s</span>
                    </div>
                    <span className={`text-[12px] font-black uppercase tracking-widest mt-1 ${ev.speaker === 'SPEAKER_01' ? 'text-red-400' : 'text-blue-400'}`}>
                      {(ev.type || "SIGNAL").toUpperCase()}
                    </span>
                    <div className="flex gap-2 mt-1">
                      <Badge variant="outline" className="text-[8px] h-4 px-1 text-slate-500 border-slate-700 bg-slate-900">
                        CONF: {confidence}%
                      </Badge>
                      <Badge variant="outline" className="text-[8px] h-4 px-1 text-slate-500 border-slate-700 bg-slate-900">
                        DIST: {distance}m
                      </Badge>
                    </div>
                  </div>

                  {/* Digital Decoration */}
                  <div className="flex items-center gap-4">
                    <span className="text-[9px] font-mono text-slate-600 hidden xl:block">
                      <div className="flex flex-col items-end">
                        <span>PWR: {db}dB</span>
                      </div>
                    </span>
                    <ChevronRight className={`w-4 h-4 transition-transform ${isActive ? 'translate-x-1 text-white' : 'opacity-20'}`} />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      <Tabs defaultValue="separation" className="w-full">
        <TabsList className="bg-slate-950 border border-slate-800 h-16 p-1 gap-2">
          <TabsTrigger value="separation" className="gap-3 px-10 uppercase font-black text-[11px] data-[state=active]:bg-indigo-600"><Layers className="w-4 h-4" /> Forensic Stems</TabsTrigger>
          <TabsTrigger value="spectrogram" className="gap-3 px-10 uppercase font-black text-[11px] data-[state=active]:bg-blue-600"><Activity className="w-4 h-4" /> Spectral Analysis</TabsTrigger>
        </TabsList>
        <TabsContent value="separation" className="mt-8">
          {!showStems ? (
            <div className="h-96 border-2 border-dashed border-slate-800 rounded-3xl flex flex-col items-center justify-center text-slate-600">
              <Database className="w-16 h-16 mb-6 opacity-20" />
              <p className="text-xs font-black tracking-[0.4em] uppercase opacity-40">Execute "Deconstruct Audio" to activate AI Separation</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-20">
              <ForensicTrack url={audioData?.url} label="Master Mix" color="#ffffff" icon={AudioWaveform} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Master"])} />
              <ForensicTrack url={currentStems?.vocals} label="Vocals / Dialogue" color="#3b82f6" icon={Mic2} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Voice", "Speech"])} />
              <ForensicTrack url={currentStems?.background} label="Ambient / Noise" color="#10b981" icon={Waves} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Music", "Background"])} />
              <ForensicTrack url={currentStems?.vehicles} label="Vehicle / Machinery" color="#ef4444" icon={Car} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Vehicle", "Car", "Engine"])} />
              <ForensicTrack url={currentStems?.footsteps} label="Footsteps / Impact" color="#8b5cf6" icon={Footprints} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Footstep"])} />
              <ForensicTrack url={currentStems?.animals} label="Animal Signal" color="#f59e0b" icon={Bird} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Animal", "Bird", "Dog"])} />
              <ForensicTrack url={currentStems?.wind} label="Atmospheric Wind" color="#06b6d4" icon={Wind} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Wind", "Thunder"])} />

              {/* NEW FORENSIC CATEGORIES */}
              <ForensicTrack url={currentStems?.gunshots} label="Gunshot / Explosion" color="#dc2626" icon={Bomb} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Gunshot", "Explosion"])} />
              <ForensicTrack url={currentStems?.screams} label="Scream / Aggression" color="#be123c" icon={Megaphone} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Scream", "Shout"])} />
              <ForensicTrack url={currentStems?.sirens} label="Siren / Alarm" color="#f97316" icon={AlertTriangle} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Siren", "Alarm"])} />
              <ForensicTrack url={currentStems?.impact} label="Impact / Breach" color="#7c3aed" icon={Hammer} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Glass", "Hammer", "Slam"])} />
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}