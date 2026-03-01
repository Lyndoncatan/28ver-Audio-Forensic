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

import ForensicDashboard from "./forensic-dashboard" // New Component

// --- FORENSIC TRACK COMPONENT ---
function ForensicTrack({ url, label, color, icon: Icon, masterPlaying, masterTime, stats, isSeparating }: any) {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveSurferRef = useRef<WaveSurfer | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [isLocalPlaying, setIsLocalPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)

  const isEmpty = url === "__EMPTY__" || (!url && !isSeparating)
  const displayUrl = url === "__EMPTY__" ? null : url

  useEffect(() => {
    if (!containerRef.current || !displayUrl) return
    if (waveSurferRef.current) waveSurferRef.current.destroy();

    waveSurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#1e293b",
      progressColor: color,
      cursorColor: "#ffffff",
      barWidth: 2,
      barGap: 3,
      height: 60,
      url: displayUrl,
    })

    waveSurferRef.current.on("ready", () => setIsReady(true))
    waveSurferRef.current.on("play", () => setIsLocalPlaying(true))
    waveSurferRef.current.on("pause", () => setIsLocalPlaying(false))

    return () => waveSurferRef.current?.destroy()
  }, [displayUrl, color])

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
    <Card className={`bg-slate-950/50 border-slate-800 p-6 transition-all hover:border-slate-600 group relative overflow-hidden ${isMuted || isEmpty ? 'opacity-40' : 'opacity-100'}`}>
      <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: color }} />
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 shadow-xl group-hover:scale-105 transition-transform">
            <Icon size={20} style={{ color: isEmpty ? '#475569' : color }} />
          </div>
          <div>
            <h4 className="text-[12px] font-black uppercase tracking-widest text-slate-100 italic leading-none">{label}</h4>
            <div className="flex items-center gap-2 mt-1">
              {!isEmpty && <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: color }} />}
              <span className={`text-[9px] font-mono tracking-tighter uppercase ${isEmpty ? 'text-slate-600' : 'text-slate-500'}`}>
                {isSeparating ? "Isolating..." : (isEmpty ? "None_Identified" : "Signal_Isolated")}
              </span>
            </div>
            {/* NEW: Stats Display */}
            {stats && !isEmpty && (
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
          <Button onClick={toggleMute} disabled={isEmpty} variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800">
            {isMuted ? <VolumeX size={16} className="text-red-500" /> : <Volume2 size={16} className="text-slate-400" />}
          </Button>
          <Button onClick={toggleLocalPlay} disabled={isEmpty} variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800">
            {isLocalPlaying ? <Pause size={16} className="text-white" /> : <Play size={16} className="text-white ml-0.5" />}
          </Button>
          {displayUrl && (
            <Button variant="ghost" size="icon" className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800" asChild>
              <a href={displayUrl} download><Download size={16} className="text-slate-400" /></a>
            </Button>
          )}
        </div>
      </div>
      <div ref={containerRef} className="opacity-80 group-hover:opacity-100 transition-opacity cursor-pointer mt-2" />
      {isEmpty && !isSeparating && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/20 pointer-events-none">
          <span className="text-[10px] text-slate-700 font-black tracking-widest opacity-30">NO_SIGNAL_MATCH</span>
        </div>
      )}
    </Card>
  )
}

// Import PDF Generator
import { generatePDFReport } from "@/lib/pdf-generator"

// --- MAIN SONAR VIEW ---
export default function SonarView({
  audioData,
  setAudioData, // New Prop
  liveEvents = [],
  isRecording = false,
  currentStems,
  setCurrentStems,
  showStems,
  setShowStems,
  externalRefs // New Prop
}: any) {
  // Use passed refs or fallback to local (though page.tsx provides them now)
  const localCanvas2DRef = useRef<HTMLCanvasElement>(null)
  const localCanvas3DRef = useRef<HTMLCanvasElement>(null)
  const localOscRef = useRef<HTMLCanvasElement>(null)
  const localSpecRef = useRef<HTMLCanvasElement>(null)

  const canvas2DRef = externalRefs?.canvas2DRef || localCanvas2DRef
  const canvas3DRef = externalRefs?.canvas3DRef || localCanvas3DRef
  const oscCanvasRef = externalRefs?.oscRef || localOscRef
  const specCanvasRef = externalRefs?.specRef || localSpecRef

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
        // AND preserve/update frequencySpectrum to prevent blank charts
        if (setAudioData && result.classification) {
          setAudioData((prev: any) => ({
            ...prev,
            analysisResults: {
              ...result.classification,
              frequencySpectrum: result.frequencySpectrum || prev.analysisResults?.frequencySpectrum || []
            }
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

    // Outer glow ring
    const outerGlow = ctx.createRadialGradient(cx, cy, maxR * 0.85, cx, cy, maxR * 1.1);
    outerGlow.addColorStop(0, "rgba(34, 197, 94, 0)");
    outerGlow.addColorStop(0.5, "rgba(34, 197, 94, 0.08)");
    outerGlow.addColorStop(1, "rgba(34, 197, 94, 0)");
    ctx.fillStyle = outerGlow;
    ctx.fillRect(0, 0, width, height);

    // Draw Grid (Circles) — brighter
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      ctx.strokeStyle = i === 5 ? "rgba(99, 102, 241, 0.6)" : "rgba(34, 197, 94, 0.25)";
      if (i === 5) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(cx, cy, (maxR / 5) * i, 0, Math.PI * 2);
      ctx.stroke();

      // Distance Labels on Grid
      if (i < 5) {
        ctx.fillStyle = "rgba(34, 197, 94, 0.5)";
        ctx.font = "10px monospace";
        ctx.fillText(`${(100 - i * 20)}m`, cx + 4, cy - (maxR / 5) * i - 4);
      }
    }

    // Draw Grid (Lines) — brighter
    ctx.strokeStyle = "rgba(34, 197, 94, 0.12)";
    ctx.setLineDash([]);
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
      ctx.stroke();
    }

    // Center crosshair
    ctx.strokeStyle = "rgba(34, 197, 94, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#4ade80";
    ctx.fill();

    // Scanning Effect
    scanAngle.current = (scanAngle.current + 0.02) % (Math.PI * 2);
    const gradient = ctx.createConicGradient(scanAngle.current, cx, cy);
    gradient.addColorStop(0, "rgba(34, 197, 94, 0)");
    gradient.addColorStop(0.1, "rgba(34, 197, 94, 0.35)");
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

    // Draw Events — bigger dots with glow
    activeEvents.forEach((ev: any) => {
      const duration = audioData?.analysisResults?.duration || 1;
      const normalizedTime = (ev.time || 0) / duration;
      const a = normalizedTime * Math.PI * 2 - Math.PI / 2;

      const db = Number(ev.decibels || -60);
      const normalizedDb = Math.max(0, Math.min(1, (db + 90) / 90));
      const radiusFactor = 1.0 - normalizedDb;
      const d = (radiusFactor * maxR * 0.9) + (maxR * 0.1);

      const x = cx + Math.cos(a) * d;
      const y = cy + Math.sin(a) * d;

      const isActive = Math.abs(currentTime - Number(ev.time || 0)) < 0.3;
      const mh = mousePos.current;
      const distToMouse = Math.sqrt((mh.x - x) ** 2 + (mh.y - y) ** 2);
      const isHovered = distToMouse < 15;
      if (isHovered) foundHover = ev;

      const baseColor = ev.speaker === "SPEAKER_01" ? "#ef4444" : "#3b82f6";
      const color = isActive ? "#ffffff" : baseColor;

      // Always draw a subtle glow ring around each event
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.fillStyle = baseColor + "30";
      ctx.fill();

      // Spreading Ripple Effect (when active)
      if (isActive) {
        const pulse = (Date.now() / 1000) % 1;
        ctx.beginPath();
        ctx.strokeStyle = baseColor;
        ctx.globalAlpha = 1 - pulse;
        ctx.lineWidth = 2;
        ctx.arc(x, y, 5 + (pulse * 30), 0, Math.PI * 2);
        ctx.stroke();
        const pulse2 = ((Date.now() / 1000) + 0.5) % 1;
        ctx.globalAlpha = 1 - pulse2;
        ctx.beginPath();
        ctx.arc(x, y, 5 + (pulse2 * 30), 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }

      // The Point — bigger
      ctx.beginPath();
      ctx.arc(x, y, isActive ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowBlur = 12;
      ctx.shadowColor = baseColor;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Labels
      if (isActive || isHovered) {
        if (isActive) {
          ctx.beginPath();
          ctx.strokeStyle = baseColor;
          ctx.globalAlpha = 0.3;
          ctx.lineWidth = 1;
          ctx.moveTo(cx, cy);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "left";
        ctx.fillText(`${ev.type || "SIGNAL"}`, x + 14, y);
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = "10px monospace";
        ctx.fillText(`${Math.abs(db).toFixed(0)}dB | ${((ev.confidence || 0) * 100).toFixed(0)}%`, x + 14, y + 12);
      }
    });

    setHoveredEvent(foundHover);

  }, [audioData, activeEvents, currentTime, isPlaying])

  const draw3DVoxel = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Clear
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    // Grid Gradient (Floor Glow) — stronger
    const cx = width / 2;
    const cy = height / 2;
    const grd = ctx.createRadialGradient(cx, cy + 80, 0, cx, cy + 80, width * 0.8);
    grd.addColorStop(0, "rgba(30, 41, 59, 0.7)");
    grd.addColorStop(0.5, "rgba(15, 23, 42, 0.4)");
    grd.addColorStop(1, "rgba(2, 6, 23, 0)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, height * 0.3, width, height * 0.7);

    // 3D Projection Helper
    const project = (x: number, y: number, z: number) => {
      const fl = 400;
      const scale = fl / (fl + z + 200);
      if (scale <= 0) return { x: 0, y: 0, s: 0, zIndex: z };
      const x2D = width / 2 + x * scale;
      const y2D = height / 2 + y * scale;
      return { x: x2D, y: y2D, s: scale, zIndex: z }
    }

    const rx = rotation.x;
    const ry = rotation.y;

    const time = Date.now() / 1000;
    const ringOffset = (time * 20) % 50;

    // Draw Floor Grid — much more visible
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
    ctx.beginPath();

    for (let r = ringOffset; r <= 600; r += 50) {
      if (r < 10) continue;
      let firstPoint = true;
      for (let a = 0; a <= Math.PI * 2; a += 0.1) {
        const bx = Math.cos(a) * r;
        const bz = Math.sin(a) * r;
        const x = bx * Math.cos(ry) - bz * Math.sin(ry);
        const z = bx * Math.sin(ry) + bz * Math.cos(ry);
        const y = 100;
        const p = project(x, y, z);
        if (p.s > 0) {
          if (firstPoint) { ctx.moveTo(p.x, p.y); firstPoint = false; }
          else ctx.lineTo(p.x, p.y);
        }
      }
    }
    ctx.stroke();

    // Draw Grid Lines — brighter
    ctx.beginPath();
    for (let i = -300; i <= 300; i += 50) {
      let p1 = project(i * Math.cos(ry) - -300 * Math.sin(ry), 100, i * Math.sin(ry) + -300 * Math.cos(ry));
      let p2 = project(i * Math.cos(ry) - 300 * Math.sin(ry), 100, i * Math.sin(ry) + 300 * Math.cos(ry));
      if (p1.s > 0 && p2.s > 0) {
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
      }
      let p3 = project(-300 * Math.cos(ry) - i * Math.sin(ry), 100, -300 * Math.sin(ry) + i * Math.cos(ry));
      let p4 = project(300 * Math.cos(ry) - i * Math.sin(ry), 100, 300 * Math.sin(ry) + i * Math.cos(ry));
      if (p3.s > 0 && p4.s > 0) {
        ctx.moveTo(p3.x, p3.y);
        ctx.lineTo(p4.x, p4.y);
      }
    }
    ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
    ctx.stroke();

    // Draw a scanning plane (horizontal sweep)
    const scanZ = ((time * 60) % 600) - 300;
    const scanP1 = project(-300 * Math.cos(ry) - scanZ * Math.sin(ry), 100, -300 * Math.sin(ry) + scanZ * Math.cos(ry));
    const scanP2 = project(300 * Math.cos(ry) - scanZ * Math.sin(ry), 100, 300 * Math.sin(ry) + scanZ * Math.cos(ry));
    const scanP3 = project(300 * Math.cos(ry) - scanZ * Math.sin(ry), -100, 300 * Math.sin(ry) + scanZ * Math.cos(ry));
    const scanP4 = project(-300 * Math.cos(ry) - scanZ * Math.sin(ry), -100, -300 * Math.sin(ry) + scanZ * Math.cos(ry));
    if (scanP1.s > 0 && scanP2.s > 0) {
      ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
      ctx.beginPath();
      ctx.moveTo(scanP1.x, scanP1.y);
      ctx.lineTo(scanP2.x, scanP2.y);
      ctx.lineTo(scanP3.x, scanP3.y);
      ctx.lineTo(scanP4.x, scanP4.y);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(scanP1.x, scanP1.y);
      ctx.lineTo(scanP2.x, scanP2.y);
      ctx.stroke();
    }

    // Center marker
    const centerP = project(0, 100, 0);
    if (centerP.s > 0) {
      ctx.fillStyle = "rgba(99, 102, 241, 0.6)";
      ctx.beginPath();
      ctx.arc(centerP.x, centerP.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hit Testing Variables
    let bestDist = 20;
    let foundVoxel: any = null;
    const mx = mousePos3D.current.x;
    const my = mousePos3D.current.y;

    // Sort events
    const renderList: any[] = [];

    activeEvents.forEach((ev: any) => {
      const duration = (audioData?.analysisResults?.duration || 1)

      // Calculate 3D Position
      const xRaw = (((ev.time || 0) / duration) * 400) - 200;

      // Map dB to Depth
      const db = Number(ev.decibels || -60);
      const normalizedDb = Math.max(0, Math.min(1, (db + 90) / 90));
      const zRaw = ((1 - normalizedDb) * 300) - 150;

      // Apply Rotation
      const x = xRaw * Math.cos(ry) - zRaw * Math.sin(ry);
      const z = xRaw * Math.sin(ry) + zRaw * Math.cos(ry);
      const yBase = 100; // Floor

      // Height based on confidence
      const confidence = ev.confidence || 0.5;
      const h = Math.max(20, confidence * 150);

      // Project Base and Top
      const projBase = project(x, yBase, z);
      const projTop = project(x, yBase - h, z);

      if (projBase.s > 0) {
        renderList.push({ ev, projBase, projTop, zIndex: z, h, isActive: Math.abs(currentTime - (ev.time || 0)) < 0.3 });
      }
    });

    renderList.sort((a, b) => b.zIndex - a.zIndex);

    renderList.forEach((item) => {
      const { ev, projBase, projTop, isActive, h, zIndex } = item;

      // Check Hover
      const dist = Math.sqrt((mx - projTop.x) ** 2 + (my - projTop.y) ** 2);
      const isHovered = dist < 20;
      if (isHovered && dist < bestDist) {
        bestDist = dist;
        foundVoxel = ev;
      }
      const isSelected = hoveredVoxel && hoveredVoxel === ev;

      const baseColor = ev.speaker === 'SPEAKER_01' ? '#ef4444' : '#3b82f6'; // Red vs Blue

      // Calculate Opacity based on Z-Distance (Fog)
      // zIndex ranges roughly -200 to 200. Far away (200) should be faded.
      const fogFactor = Math.max(0.2, 1 - (zIndex + 200) / 600);

      const alpha = isActive || isSelected ? 1 : fogFactor;
      const color = isActive || isSelected ? "#ffffff" : baseColor;

      // 1. Draw "Volumetric" Pillar (Gradient Line)
      const grad = ctx.createLinearGradient(projBase.x, projBase.y, projTop.x, projTop.y);
      grad.addColorStop(0, `${baseColor}00`); // Transparent at bottom
      grad.addColorStop(0.2, `${baseColor}40`);
      grad.addColorStop(1, `${baseColor}FF`); // Opaque at top

      ctx.lineWidth = projBase.s * (isActive ? 12 : 6);
      ctx.strokeStyle = grad;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(projBase.x, projBase.y);
      ctx.lineTo(projTop.x, projTop.y);
      ctx.stroke();
      ctx.lineCap = "butt"; // Reset

      // 2. Connector Line (Thin, distinct)
      ctx.lineWidth = 1;
      ctx.strokeStyle = isActive ? "#fff" : `${baseColor}80`;
      ctx.beginPath();
      ctx.moveTo(projBase.x, projBase.y);
      ctx.lineTo(projTop.x, projTop.y);
      ctx.stroke();

      // 3. Base Ripple
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(projBase.x, projBase.y, projBase.s * 10, projBase.s * 4, 0, 0, Math.PI * 2);
      ctx.stroke();

      // 4. Top Orb (Glowing)
      const orbSize = projTop.s * (isActive ? 8 : 4);

      // Glow
      if (isActive || isSelected) {
        const glow = ctx.createRadialGradient(projTop.x, projTop.y, 0, projTop.x, projTop.y, orbSize * 4);
        glow.addColorStop(0, `${baseColor}FF`);
        glow.addColorStop(1, `${baseColor}00`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(projTop.x, projTop.y, orbSize * 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // Core
      ctx.fillStyle = isActive ? "#ffffff" : baseColor;
      ctx.beginPath();
      ctx.arc(projTop.x, projTop.y, orbSize, 0, Math.PI * 2);
      ctx.fill();

      // Label 3D (Floating Billboard)
      if (isActive || isSelected) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px monospace";
        ctx.textAlign = "center";
        ctx.shadowColor = "black";
        ctx.shadowBlur = 4;
        const label = `${ev.type}`;
        ctx.fillText(label, projTop.x, projTop.y - 15);
        ctx.font = "9px monospace";
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(`${Math.abs(Number(ev.decibels)).toFixed(1)}dB`, projTop.x, projTop.y - 5);
        ctx.shadowBlur = 0;

        // Draw leader line to actual 3D point
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(projTop.x, projTop.y);
        ctx.lineTo(projTop.x, projTop.y - 15); // Connect text to orb
        ctx.stroke();
      }
    });

    setHoveredVoxel(foundVoxel);

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

      <ForensicDashboard
        audioData={audioData}
        isPlaying={isPlaying}
        currentTime={currentTime}
        activeEvents={activeEvents}
        audioRef={audioRef}
        externalOscilloscopeRef={oscCanvasRef}
        externalSpectrogramRef={specCanvasRef}
      />

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
              <ForensicTrack url={audioData?.url} label="Master Mix" color="#ffffff" icon={AudioWaveform} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Master"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.vocals} label="Vocals / Dialogue" color="#3b82f6" icon={Mic2} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Voice", "Speech"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.background} label="Ambient / Noise" color="#10b981" icon={Waves} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Music", "Background"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.vehicles} label="Vehicle / Machinery" color="#ef4444" icon={Car} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Vehicle", "Car", "Engine"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.footsteps} label="Footsteps / Impact" color="#8b5cf6" icon={Footprints} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Footstep"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.animals} label="Animal Signal" color="#f59e0b" icon={Bird} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Animal", "Bird", "Dog"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.wind} label="Atmospheric Wind" color="#06b6d4" icon={Wind} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Wind", "Thunder"])} isSeparating={isSeparating} />

              {/* NEW FORENSIC CATEGORIES */}
              <ForensicTrack url={currentStems?.gunshots} label="Gunshot / Explosion" color="#dc2626" icon={Bomb} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Gunshot", "Explosion"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.screams} label="Scream / Aggression" color="#be123c" icon={Megaphone} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Scream", "Shout"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.sirens} label="Siren / Alarm" color="#f97316" icon={AlertTriangle} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Siren", "Alarm"])} isSeparating={isSeparating} />
              <ForensicTrack url={currentStems?.impact} label="Impact / Breach" color="#7c3aed" icon={Hammer} masterPlaying={isPlaying} masterTime={currentTime} stats={getStats(["Glass", "Hammer", "Slam"])} isSeparating={isSeparating} />
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}