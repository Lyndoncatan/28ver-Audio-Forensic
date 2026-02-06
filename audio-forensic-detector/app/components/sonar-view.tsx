"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useRef, useCallback } from "react"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

interface SonarViewProps {
  audioData: AudioData | null
}

export default function SonarView({ audioData }: SonarViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (audioData?.analysisResults && canvasRef.current) {
      drawSonarView()

      // Set up animation loop for scanning line
      const animationId = setInterval(() => {
        drawSonarView()
      }, 100)

      return () => {
        clearInterval(animationId)
      }
    }
  }, [audioData?.analysisResults]) // Only depend on analysisResults

  const drawSonarView = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioData?.analysisResults) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = "#1a1a2e"
    ctx.fillRect(0, 0, width, height)

    // Draw concentric circles (sonar rings)
    const centerX = width / 2
    const centerY = height / 2
    const maxRadius = Math.min(width, height) / 2 - 40 // More space for labels

    // Draw range circles with labels
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius / 5) * i
      const frequency = (2000 / 5) * i

      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
      ctx.strokeStyle = `rgba(0, 255, 0, ${0.3 - i * 0.05})`
      ctx.lineWidth = 1
      ctx.stroke()

      // Add frequency labels
      ctx.fillStyle = "rgba(0, 255, 0, 0.7)"
      ctx.font = "10px Arial"
      ctx.fillText(`${frequency}Hz`, centerX + radius - 25, centerY - 5)
    }

    // Draw crosshairs with time labels
    ctx.strokeStyle = "rgba(0, 255, 0, 0.3)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, height)
    ctx.stroke()

    // Add time labels around the circle
    ctx.fillStyle = "rgba(0, 255, 0, 0.7)"
    ctx.font = "10px Arial"
    const duration = audioData.analysisResults.duration
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * 2 * Math.PI - Math.PI / 2
      const labelRadius = maxRadius + 15
      const x = centerX + Math.cos(angle) * labelRadius
      const y = centerY + Math.sin(angle) * labelRadius
      const time = (i / 8) * duration
      ctx.fillText(`${time.toFixed(1)}s`, x - 15, y + 3)
    }

    // Draw sound events as labeled blips
    const results = audioData.analysisResults
    if (results?.soundEvents) {
      results.soundEvents.forEach((event: any, index: number) => {
        const angle = (event.time / results.duration) * 2 * Math.PI - Math.PI / 2
        const distance = Math.min((event.frequency / 2000) * maxRadius, maxRadius)
        const intensity = event.amplitude

        const x = centerX + Math.cos(angle) * distance
        const y = centerY + Math.sin(angle) * distance

        // Draw blip with glow effect
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8 + intensity * 10)
        gradient.addColorStop(
          0,
          `rgba(${Math.floor(255 * intensity)}, ${Math.floor(255 * (1 - intensity))}, 0, ${0.9})`,
        )
        gradient.addColorStop(1, `rgba(${Math.floor(255 * intensity)}, ${Math.floor(255 * (1 - intensity))}, 0, 0.1)`)

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(x, y, 3 + intensity * 5, 0, 2 * Math.PI)
        ctx.fill()

        // Draw pulse effect
        ctx.beginPath()
        ctx.arc(x, y, 8 + intensity * 10, 0, 2 * Math.PI)
        ctx.strokeStyle = `rgba(${Math.floor(255 * intensity)}, ${Math.floor(255 * (1 - intensity))}, 0, ${0.3 * intensity})`
        ctx.lineWidth = 2
        ctx.stroke()

        // Add sound type labels for prominent sounds
        if (intensity > 0.6 || index < 3) {
          ctx.fillStyle = "white"
          ctx.font = "9px Arial"
          ctx.shadowColor = "black"
          ctx.shadowBlur = 2

          // Position label to avoid overlap
          const labelX = x + (Math.cos(angle) > 0 ? 10 : -30)
          const labelY = y + (Math.sin(angle) > 0 ? -5 : 15)

          ctx.fillText(event.type, labelX, labelY)
          ctx.fillText(`${event.frequency}Hz`, labelX, labelY + 10)
          ctx.fillText(`${(intensity * 100).toFixed(0)}%`, labelX, labelY + 20)

          ctx.shadowBlur = 0
        }
      })
    }

    // Draw scanning line with trail effect
    const scanAngle = (Date.now() / 2000) % (2 * Math.PI)

    // Draw trail
    for (let i = 0; i < 20; i++) {
      const trailAngle = scanAngle - i * 0.05
      const alpha = ((20 - i) / 20) * 0.3

      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(
        centerX + Math.cos(trailAngle - Math.PI / 2) * maxRadius,
        centerY + Math.sin(trailAngle - Math.PI / 2) * maxRadius,
      )
      ctx.strokeStyle = `rgba(0, 255, 0, ${alpha})`
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Draw main scanning line
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(
      centerX + Math.cos(scanAngle - Math.PI / 2) * maxRadius,
      centerY + Math.sin(scanAngle - Math.PI / 2) * maxRadius,
    )
    ctx.strokeStyle = "rgba(0, 255, 0, 0.8)"
    ctx.lineWidth = 3
    ctx.stroke()

    // Add center dot
    ctx.fillStyle = "rgba(0, 255, 0, 0.8)"
    ctx.beginPath()
    ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI)
    ctx.fill()
  }, [audioData])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioData?.analysisResults) return

    const handleCanvasClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top

      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const maxRadius = Math.min(canvas.width, canvas.height) / 2 - 40

      // Calculate clicked position in polar coordinates
      const dx = x - centerX
      const dy = y - centerY
      const distance = Math.sqrt(dx * dx + dy * dy)
      const angle = Math.atan2(dy, dx) + Math.PI / 2

      if (distance <= maxRadius) {
        const time = (angle / (2 * Math.PI)) * audioData.analysisResults.duration
        const frequency = (distance / maxRadius) * 2000

        console.log(`Clicked at: ${time.toFixed(2)}s, ${frequency.toFixed(0)}Hz`)

        // Find nearest sound event
        const nearestEvent = audioData.analysisResults.soundEvents.reduce((nearest: any, event: any) => {
          const eventDistance = Math.abs(event.time - time) + Math.abs(event.frequency - frequency) / 100
          const nearestDistance = Math.abs(nearest.time - time) + Math.abs(nearest.frequency - frequency) / 100
          return eventDistance < nearestDistance ? event : nearest
        })

        if (nearestEvent) {
          alert(
            `Sound Event: ${nearestEvent.type}\nTime: ${nearestEvent.time}s\nFrequency: ${nearestEvent.frequency}Hz\nAmplitude: ${(nearestEvent.amplitude * 100).toFixed(1)}%`,
          )
        }
      }
    }

    canvas.addEventListener("click", handleCanvasClick)
    canvas.style.cursor = "crosshair"

    return () => {
      canvas.removeEventListener("click", handleCanvasClick)
    }
  }, [audioData])

  if (!audioData) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Sonar Visualization</h2>
        <p className="text-gray-600 mb-8">Upload or record audio to see sonar visualization</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">No audio data available for visualization</p>
        </div>
      </div>
    )
  }

  if (!audioData.analysisResults) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Sonar Visualization</h2>
        <p className="text-gray-600 mb-8">Analyze audio first to see sonar visualization</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">Analysis required for visualization</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-purple-600 mb-6 text-center">Sonar Visualization</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sonar Display */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Audio Sonar Display</CardTitle>
            </CardHeader>
            <CardContent>
              <canvas ref={canvasRef} width={600} height={400} className="w-full border rounded-lg bg-gray-900" />
              <p className="text-sm text-gray-600 mt-2 text-center">
                Sound events plotted by time (angle) and frequency (distance from center)
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Legend and Info */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Legend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-500 rounded-full opacity-30"></div>
                  <span className="text-sm">Sonar Grid</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-yellow-500 rounded-full"></div>
                  <span className="text-sm">Sound Events</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Scanning Line</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Detection Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <p>
                  <strong>Total Events:</strong> {audioData.analysisResults.detectedSounds}
                </p>
                <p>
                  <strong>Max Range:</strong> 2000 Hz
                </p>
                <p>
                  <strong>Scan Duration:</strong> {audioData.analysisResults.duration}s
                </p>
                <p>
                  <strong>Resolution:</strong> High
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Sound Types</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Array.from(new Set(audioData.analysisResults.soundEvents.map((e: any) => e.type))).map((type: any) => (
                  <div key={type} className="flex justify-between">
                    <span>{type}</span>
                    <span className="text-sm text-gray-600">
                      {audioData.analysisResults.soundEvents.filter((e: any) => e.type === type).length}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
