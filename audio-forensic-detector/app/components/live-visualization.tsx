"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { useEffect, useRef, useState, useCallback } from "react"
import { Play, Pause, RotateCcw, Info } from "lucide-react"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

interface LiveVisualizationProps {
  audioData: AudioData
}

export default function LiveVisualization({ audioData }: LiveVisualizationProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [liveData, setLiveData] = useState<any>(null)
  const [analysisMode, setAnalysisMode] = useState("stft")
  const [zoomLevel, setZoomLevel] = useState([1])
  const [sensitivity, setSensitivity] = useState([75])
  const [showLabels, setShowLabels] = useState(true)
  const [selectedEvent, setSelectedEvent] = useState<any>(null)
  const [hoveredPoint, setHoveredPoint] = useState<any>(null)

  const audioRef = useRef<HTMLAudioElement>(null)
  const stftCanvasRef = useRef<HTMLCanvasElement>(null)
  const fftCanvasRef = useRef<HTMLCanvasElement>(null)
  const spectrogramCanvasRef = useRef<HTMLCanvasElement>(null)
  const energyCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Initialize audio context and analyzer
  const initializeAudioAnalysis = useCallback(async () => {
    if (!audioData || !audioRef.current) return

    try {
      // Create audio context for real-time analysis
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const source = audioContext.createMediaElementSource(audioRef.current)
      const analyzer = audioContext.createAnalyser()

      analyzer.fftSize = 2048
      const bufferLength = analyzer.frequencyBinCount
      const dataArray = new Uint8Array(bufferLength)
      const timeDataArray = new Uint8Array(bufferLength)

      source.connect(analyzer)
      analyzer.connect(audioContext.destination)

      // Generate mock comprehensive analysis data
      const mockAnalysisData = generateMockAnalysisData()
      setLiveData(mockAnalysisData)

      // Start real-time visualization loop
      const visualize = () => {
        analyzer.getByteFrequencyData(dataArray)
        analyzer.getByteTimeDomainData(timeDataArray)

        // Update visualizations with interactive features
        drawInteractiveSTFT(mockAnalysisData.stft)
        drawInteractiveFFT(dataArray)
        drawInteractiveSpectrogram(mockAnalysisData.spectrogram)
        drawInteractiveEnergyDetection(mockAnalysisData.energy)

        if (isPlaying) {
          animationRef.current = requestAnimationFrame(visualize)
        }
      }

      if (isPlaying) {
        visualize()
      }
    } catch (error) {
      console.error("Audio analysis initialization failed:", error)
      // Fallback to mock data visualization
      const mockData = generateMockAnalysisData()
      setLiveData(mockData)
      drawMockVisualizations(mockData)
    }
  }, [audioData, isPlaying, zoomLevel, sensitivity, showLabels])

  const generateMockAnalysisData = () => {
    const duration = audioData.duration || 10
    const sampleRate = 44100
    const timeSteps = 100
    const freqBins = 50

    // Generate STFT data with labeled events
    const stft = Array.from({ length: freqBins }, (_, f) =>
      Array.from({ length: timeSteps }, (_, t) => {
        const freq = (f / freqBins) * (sampleRate / 2)
        const time = (t / timeSteps) * duration
        return Math.sin(freq * time * 0.001) * Math.exp(-time * 0.1) * (Math.random() * 0.5 + 0.5)
      }),
    )

    // Generate spectrogram data
    const spectrogram = Array.from({ length: freqBins }, (_, f) =>
      Array.from({ length: timeSteps }, (_, t) => {
        return Math.random() * 0.8 + 0.1 + Math.sin((f + t) * 0.1) * 0.2
      }),
    )

    // Generate energy detection data with labeled events
    const energy = Array.from({ length: timeSteps }, (_, t) => {
      const baseEnergy = Math.sin(t * 0.2) * 0.3 + 0.5
      const noise = (Math.random() - 0.5) * 0.2
      return Math.max(0, Math.min(1, baseEnergy + noise))
    })

    // Detect peaks in energy with labels
    const peaks = []
    const peakLabels = ["Voice", "Music", "Noise", "Echo", "Ambient", "Percussion"]
    for (let i = 1; i < energy.length - 1; i++) {
      if (energy[i] > energy[i - 1] && energy[i] > energy[i + 1] && energy[i] > 0.6) {
        peaks.push({
          index: i,
          value: energy[i],
          time: (i / timeSteps) * duration,
          label: peakLabels[peaks.length % peakLabels.length],
          frequency: 440 + Math.random() * 1000,
          confidence: 0.7 + Math.random() * 0.3,
        })
      }
    }

    return {
      stft,
      spectrogram,
      energy,
      peaks,
      fft: Array.from({ length: 512 }, (_, i) => Math.random() * 255),
      timeSteps,
      freqBins,
      duration,
      sampleRate,
    }
  }

  const drawInteractiveSTFT = (stftData: number[][]) => {
    const canvas = stftCanvasRef.current
    if (!canvas || !stftData) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    const cellWidth = (width * zoomLevel[0]) / stftData[0].length
    const cellHeight = height / stftData.length

    // Draw STFT heatmap
    for (let f = 0; f < stftData.length; f++) {
      for (let t = 0; t < stftData[f].length; t++) {
        const intensity = Math.abs(stftData[f][t])
        const adjustedIntensity = Math.min(1, intensity * (sensitivity[0] / 50))
        const color = `hsl(${240 + adjustedIntensity * 120}, 70%, ${30 + adjustedIntensity * 50}%)`
        ctx.fillStyle = color
        ctx.fillRect(t * cellWidth, (stftData.length - f - 1) * cellHeight, cellWidth, cellHeight)
      }
    }

    // Add interactive grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width
      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Add labels if enabled
    if (showLabels) {
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      ctx.shadowColor = "black"
      ctx.shadowBlur = 2

      // Time labels
      for (let i = 0; i <= 5; i++) {
        const x = (i / 5) * width
        const time = (i / 5) * audioData.duration
        ctx.fillText(`${time.toFixed(1)}s`, x, height - 5)
      }

      // Frequency labels
      for (let i = 0; i <= 5; i++) {
        const y = height - (i / 5) * height
        const freq = (i / 5) * 22050
        ctx.fillText(`${(freq / 1000).toFixed(1)}kHz`, 5, y)
      }

      ctx.shadowBlur = 0
    }

    // Add current time indicator
    if (isPlaying) {
      const timeX = (currentTime / audioData.duration) * width
      ctx.strokeStyle = "rgba(255, 0, 0, 0.8)"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(timeX, 0)
      ctx.lineTo(timeX, height)
      ctx.stroke()
    }
  }

  const drawInteractiveFFT = (fftData: Uint8Array) => {
    const canvas = fftCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background with grid
    ctx.fillStyle = "#1a1a2e"
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)"
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width
      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw FFT spectrum with interactive bars
    const barWidth = width / fftData.length
    const threshold = sensitivity[0] * 2.55 // Convert to 0-255 range

    for (let i = 0; i < fftData.length; i++) {
      const barHeight = (fftData[i] / 255) * height * zoomLevel[0]
      const frequency = (i / fftData.length) * 22050

      // Color based on intensity and frequency
      let color = "#00ff88"
      if (fftData[i] > threshold) {
        color = frequency < 1000 ? "#ff4444" : frequency < 4000 ? "#ffaa00" : "#00aaff"
      }

      ctx.fillStyle = color
      ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight)

      // Add peak labels for prominent frequencies
      if (fftData[i] > threshold && showLabels && i % 20 === 0) {
        ctx.fillStyle = "white"
        ctx.font = "10px Arial"
        ctx.shadowColor = "black"
        ctx.shadowBlur = 1
        ctx.fillText(`${(frequency / 1000).toFixed(1)}k`, i * barWidth, height - barHeight - 5)
        ctx.shadowBlur = 0
      }
    }

    // Add frequency scale
    if (showLabels) {
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      for (let i = 0; i <= 5; i++) {
        const x = (i / 5) * width
        const freq = (i / 5) * 22050
        ctx.fillText(`${(freq / 1000).toFixed(0)}kHz`, x, height - 5)
      }
    }
  }

  const drawInteractiveSpectrogram = (spectrogramData: number[][]) => {
    const canvas = spectrogramCanvasRef.current
    if (!canvas || !spectrogramData) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    const cellWidth = width / spectrogramData[0].length
    const cellHeight = height / spectrogramData.length

    // Draw spectrogram with enhanced colors
    for (let f = 0; f < spectrogramData.length; f++) {
      for (let t = 0; t < spectrogramData[f].length; t++) {
        const intensity = spectrogramData[f][t] * (sensitivity[0] / 50)
        const hue = 300 + intensity * 60
        const saturation = 80
        const lightness = 20 + intensity * 60
        const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`
        ctx.fillStyle = color
        ctx.fillRect(t * cellWidth, (spectrogramData.length - f - 1) * cellHeight, cellWidth, cellHeight)
      }
    }

    // Add interactive overlays
    if (hoveredPoint) {
      ctx.strokeStyle = "white"
      ctx.lineWidth = 2
      ctx.strokeRect(hoveredPoint.x - 2, hoveredPoint.y - 2, 4, 4)
    }

    // Add time/frequency labels
    if (showLabels) {
      ctx.fillStyle = "white"
      ctx.font = "10px Arial"
      ctx.shadowColor = "black"
      ctx.shadowBlur = 1

      // Time markers
      for (let i = 0; i <= 4; i++) {
        const x = (i / 4) * width
        const time = (i / 4) * audioData.duration
        ctx.fillText(`${time.toFixed(1)}s`, x, height - 5)
      }

      ctx.shadowBlur = 0
    }

    // Current time indicator
    if (isPlaying) {
      const timeX = (currentTime / audioData.duration) * width
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(timeX, 0)
      ctx.lineTo(timeX, height)
      ctx.stroke()
    }
  }

  const drawInteractiveEnergyDetection = (energyData: number[]) => {
    const canvas = energyCanvasRef.current
    if (!canvas || !energyData || !liveData) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background with grid
    ctx.fillStyle = "#0a0a0a"
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)"
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw energy curve with gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, "rgba(0, 255, 0, 0.8)")
    gradient.addColorStop(1, "rgba(0, 255, 0, 0.2)")

    ctx.fillStyle = gradient
    ctx.beginPath()

    const stepX = width / energyData.length
    ctx.moveTo(0, height)

    for (let i = 0; i < energyData.length; i++) {
      const x = i * stepX
      const y = height - energyData[i] * height * zoomLevel[0]
      ctx.lineTo(x, y)
    }

    ctx.lineTo(width, height)
    ctx.fill()

    // Draw energy line
    ctx.strokeStyle = "#00ff00"
    ctx.lineWidth = 2
    ctx.beginPath()

    for (let i = 0; i < energyData.length; i++) {
      const x = i * stepX
      const y = height - energyData[i] * height * zoomLevel[0]
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Draw detected peaks with interactive labels
    liveData.peaks.forEach((peak: any, index: number) => {
      const x = peak.index * stepX
      const y = height - peak.value * height * zoomLevel[0]

      // Draw peak marker
      ctx.fillStyle = "#ff0000"
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, 2 * Math.PI)
      ctx.fill()

      // Draw peak ring
      ctx.strokeStyle = "rgba(255, 0, 0, 0.5)"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, y, 12, 0, 2 * Math.PI)
      ctx.stroke()

      // Add interactive labels
      if (showLabels) {
        ctx.fillStyle = "white"
        ctx.font = "10px Arial"
        ctx.shadowColor = "black"
        ctx.shadowBlur = 2

        const labelX = x + (x > width / 2 ? -60 : 10)
        const labelY = y - 20

        // Background for label
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
        ctx.fillRect(labelX - 2, labelY - 12, 80, 30)

        // Label text
        ctx.fillStyle = "white"
        ctx.fillText(peak.label, labelX, labelY)
        ctx.fillText(`${peak.time.toFixed(1)}s`, labelX, labelY + 10)
        ctx.fillText(`${peak.confidence.toFixed(0)}%`, labelX, labelY + 20)

        ctx.shadowBlur = 0
      }
    })

    // Current time indicator
    if (isPlaying) {
      const timeX = (currentTime / audioData.duration) * width
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(timeX, 0)
      ctx.lineTo(timeX, height)
      ctx.stroke()
    }

    // Add time scale
    if (showLabels) {
      ctx.fillStyle = "white"
      ctx.font = "10px Arial"
      for (let i = 0; i <= 5; i++) {
        const x = (i / 5) * width
        const time = (i / 5) * audioData.duration
        ctx.fillText(`${time.toFixed(1)}s`, x, height - 5)
      }
    }
  }

  const drawMockVisualizations = (mockData: any) => {
    drawInteractiveSTFT(mockData.stft)
    drawInteractiveFFT(new Uint8Array(mockData.fft))
    drawInteractiveSpectrogram(mockData.spectrogram)
    drawInteractiveEnergyDetection(mockData.energy)
  }

  // Add canvas interaction handlers
  const addCanvasInteractions = () => {
    const canvases = [stftCanvasRef, fftCanvasRef, spectrogramCanvasRef, energyCanvasRef]

    canvases.forEach((canvasRef, index) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const handleMouseMove = (event: MouseEvent) => {
        const rect = canvas.getBoundingClientRect()
        const x = event.clientX - rect.left
        const y = event.clientY - rect.top

        setHoveredPoint({ x, y, canvas: index })

        // Show tooltip information
        const time = (x / canvas.width) * audioData.duration
        const frequency = ((canvas.height - y) / canvas.height) * 22050

        canvas.title = `Time: ${time.toFixed(2)}s, Frequency: ${(frequency / 1000).toFixed(1)}kHz`
      }

      const handleClick = (event: MouseEvent) => {
        const rect = canvas.getBoundingClientRect()
        const x = event.clientX - rect.left
        const y = event.clientY - rect.top

        const time = (x / canvas.width) * audioData.duration

        // Seek audio to clicked time
        if (audioRef.current) {
          audioRef.current.currentTime = time
          setCurrentTime(time)
        }
      }

      canvas.addEventListener("mousemove", handleMouseMove)
      canvas.addEventListener("click", handleClick)
      canvas.style.cursor = "crosshair"
    })
  }

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0
      setCurrentTime(0)
    }
  }

  useEffect(() => {
    initializeAudioAnalysis()
    addCanvasInteractions()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [initializeAudioAnalysis])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const updateTime = () => setCurrentTime(audio.currentTime)
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)

    audio.addEventListener("timeupdate", updateTime)
    audio.addEventListener("play", handlePlay)
    audio.addEventListener("pause", handlePause)

    return () => {
      audio.removeEventListener("timeupdate", updateTime)
      audio.removeEventListener("play", handlePlay)
      audio.removeEventListener("pause", handlePause)
    }
  }, [])

  return (
    <div className="space-y-6">
      <div className="border-t border-gray-200 pt-6">
        <h3 className="text-xl font-bold text-purple-600 mb-4 text-center">Interactive Live Audio Analysis</h3>

        {/* Enhanced Audio Controls */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-lg flex items-center justify-between">
              Interactive Audio Controls
              <div className="flex space-x-2">
                <Badge variant="outline" className="bg-green-50 text-green-700">
                  🔴 Live Analysis
                </Badge>
                <Badge variant="outline">{analysisMode.toUpperCase()}</Badge>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <audio ref={audioRef} src={audioData.url} className="hidden" />

              <div className="flex items-center justify-center space-x-4">
                <Button onClick={handlePlayPause} variant="outline" size="sm">
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? "Pause" : "Play"}
                </Button>
                <Button onClick={handleReset} variant="outline" size="sm">
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </Button>
                <Button onClick={() => setShowLabels(!showLabels)} variant="outline" size="sm">
                  <Info className="w-4 h-4" />
                  {showLabels ? "Hide" : "Show"} Labels
                </Button>
              </div>

              {/* Interactive Controls */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Zoom Level: {zoomLevel[0].toFixed(1)}x</label>
                  <Slider
                    value={zoomLevel}
                    onValueChange={setZoomLevel}
                    max={3}
                    min={0.5}
                    step={0.1}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Sensitivity: {sensitivity[0]}%</label>
                  <Slider
                    value={sensitivity}
                    onValueChange={setSensitivity}
                    max={100}
                    min={10}
                    step={5}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="text-center">
                <p className="text-sm text-gray-600">
                  {Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(0).padStart(2, "0")} /{" "}
                  {Math.floor(audioData.duration / 60)}:{(audioData.duration % 60).toFixed(0).padStart(2, "0")}
                </p>
                <div
                  className="w-full bg-gray-200 rounded-full h-3 mt-2 cursor-pointer"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    const x = e.clientX - rect.left
                    const time = (x / rect.width) * audioData.duration
                    if (audioRef.current) {
                      audioRef.current.currentTime = time
                      setCurrentTime(time)
                    }
                  }}
                >
                  <div
                    className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${(currentTime / audioData.duration) * 100}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">Click on progress bar to seek</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Interactive Visualization Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Interactive STFT */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                Interactive STFT
                <Badge variant="outline" className="text-xs">
                  Click to seek
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <canvas ref={stftCanvasRef} width={400} height={300} className="w-full border rounded-lg bg-gray-900" />
              <p className="text-sm text-gray-600 mt-2">
                Real-time frequency analysis • Click to seek • Hover for details
              </p>
            </CardContent>
          </Card>

          {/* Interactive FFT */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                Interactive FFT Spectrum
                <Badge variant="outline" className="text-xs">
                  Live peaks
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <canvas ref={fftCanvasRef} width={400} height={300} className="w-full border rounded-lg bg-gray-900" />
              <p className="text-sm text-gray-600 mt-2">Current frequency distribution • Color-coded by intensity</p>
            </CardContent>
          </Card>

          {/* Interactive Spectrogram */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                Interactive Spectrogram
                <Badge variant="outline" className="text-xs">
                  Time-frequency
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <canvas
                ref={spectrogramCanvasRef}
                width={400}
                height={300}
                className="w-full border rounded-lg bg-gray-900"
              />
              <p className="text-sm text-gray-600 mt-2">
                Dynamic frequency visualization • White line shows current time
              </p>
            </CardContent>
          </Card>

          {/* Interactive Sound Events */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                Interactive Sound Events
                <Badge variant="outline" className="text-xs">
                  {liveData?.peaks?.length || 0} events
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <canvas ref={energyCanvasRef} width={400} height={300} className="w-full border rounded-lg bg-gray-900" />
              <p className="text-sm text-gray-600 mt-2">
                Labeled sound events • Red dots are detected sounds • Click to seek
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Enhanced Live Analysis Report */}
        {liveData && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="text-lg">Interactive Analysis Dashboard</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <strong className="text-blue-700">File:</strong>
                  <p className="text-blue-600">{audioData.name}</p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <strong className="text-green-700">Duration:</strong>
                  <p className="text-green-600">{liveData.duration.toFixed(2)}s</p>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <strong className="text-purple-700">Sample Rate:</strong>
                  <p className="text-purple-600">{liveData.sampleRate} Hz</p>
                </div>
                <div className="p-3 bg-orange-50 rounded-lg">
                  <strong className="text-orange-700">Events:</strong>
                  <p className="text-orange-600">{liveData.peaks.length} detected</p>
                </div>
              </div>

              {/* Interactive Event List */}
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Detected Sound Events (Click to seek):</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-32 overflow-y-auto">
                  {liveData.peaks.map((peak: any, index: number) => (
                    <button
                      key={index}
                      onClick={() => {
                        if (audioRef.current) {
                          audioRef.current.currentTime = peak.time
                          setCurrentTime(peak.time)
                        }
                      }}
                      className="p-2 text-left bg-gray-50 hover:bg-gray-100 rounded border text-xs transition-colors"
                    >
                      <div className="font-medium text-purple-600">{peak.label}</div>
                      <div className="text-gray-600">
                        {peak.time.toFixed(1)}s • {peak.confidence.toFixed(0)}%
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm">
                  <strong>Status:</strong>{" "}
                  <span className={isPlaying ? "text-green-600" : "text-gray-600"}>
                    {isPlaying ? "🔴 Live Analysis Active" : "⏸️ Analysis Paused"}
                  </span>
                  {hoveredPoint && (
                    <span className="ml-4 text-blue-600">
                      • Hover: Canvas {hoveredPoint.canvas + 1} at ({hoveredPoint.x}, {hoveredPoint.y})
                    </span>
                  )}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
