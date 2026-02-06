"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Mic, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import AudioAnalysis from "./components/audio-analysis"
import SonarView from "./components/sonar-view"
import AudioSettings from "./components/audio-settings"
import LiveVisualization from "./components/live-visualization"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

export default function AudioForensicDetector() {
  const [activeTab, setActiveTab] = useState("Record")
  const [isRecording, setIsRecording] = useState(false)
  const [audioData, setAudioData] = useState<AudioData | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const tabs = ["Record", "Upload", "Analyze", "Sonar View", "Sessions", "Settings"]

  useEffect(() => {
    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
      }
    }
  }, [])

  const startRecording = async () => {
    try {
      // Check if we're in a secure context (HTTPS or localhost)
      if (!window.isSecureContext) {
        alert("Microphone access requires HTTPS or localhost. Please use a secure connection.")
        return
      }

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(
          "Your browser doesn't support audio recording. Please use a modern browser like Chrome, Firefox, or Safari.",
        )
        return
      }

      // Request microphone permission with better constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: { ideal: 44100 },
          channelCount: { ideal: 1 },
        },
      })

      // Check MediaRecorder support and find compatible MIME type
      let mimeType = "audio/webm;codecs=opus"
      const supportedTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus", "audio/wav"]

      for (const type of supportedTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type
          break
        }
      }

      if (!MediaRecorder.isTypeSupported(mimeType)) {
        // Fallback to default
        mimeType = ""
      }

      // Create MediaRecorder with compatible options
      const options: MediaRecorderOptions = {}
      if (mimeType) {
        options.mimeType = mimeType
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mimeType || "audio/wav",
        })
        const audioUrl = URL.createObjectURL(audioBlob)

        // Determine file extension based on MIME type
        let extension = ".wav"
        if (mimeType.includes("webm")) extension = ".webm"
        else if (mimeType.includes("mp4")) extension = ".mp4"
        else if (mimeType.includes("ogg")) extension = ".ogg"

        setAudioData({
          blob: audioBlob,
          url: audioUrl,
          name: `Recording_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}${extension}`,
          duration: recordingTime,
        })

        // Stop all tracks to release the microphone
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorderRef.current.onerror = (event) => {
        console.error("MediaRecorder error:", event)
        alert("Recording error occurred. Please try again.")
        setIsRecording(false)
        if (recordingIntervalRef.current) {
          clearInterval(recordingIntervalRef.current)
        }
      }

      // Start recording
      mediaRecorderRef.current.start(1000) // Collect data every second
      setIsRecording(true)
      setRecordingTime(0)

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)

      console.log("Recording started successfully with MIME type:", mimeType)
    } catch (error: any) {
      console.error("Error accessing microphone:", error)

      // Provide specific error messages based on the error type
      if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        alert("Microphone access denied. Please allow microphone permissions and try again.")
      } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        alert("No microphone found. Please connect a microphone and try again.")
      } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
        alert(
          "Microphone is already in use by another application. Please close other apps using the microphone and try again.",
        )
      } else if (error.name === "OverconstrainedError" || error.name === "ConstraintNotSatisfiedError") {
        alert("Microphone doesn't support the requested settings. Trying with basic settings...")

        // Retry with minimal constraints
        try {
          const basicStream = await navigator.mediaDevices.getUserMedia({ audio: true })

          mediaRecorderRef.current = new MediaRecorder(basicStream)
          audioChunksRef.current = []

          mediaRecorderRef.current.ondataavailable = (event) => {
            if (event.data.size > 0) {
              audioChunksRef.current.push(event.data)
            }
          }

          mediaRecorderRef.current.onstop = () => {
            const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" })
            const audioUrl = URL.createObjectURL(audioBlob)

            setAudioData({
              blob: audioBlob,
              url: audioUrl,
              name: `Recording_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.wav`,
              duration: recordingTime,
            })

            basicStream.getTracks().forEach((track) => track.stop())
          }

          mediaRecorderRef.current.start(1000)
          setIsRecording(true)
          setRecordingTime(0)

          recordingIntervalRef.current = setInterval(() => {
            setRecordingTime((prev) => prev + 1)
          }, 1000)

          console.log("Recording started with basic settings")
        } catch (retryError) {
          console.error("Retry failed:", retryError)
          alert("Unable to access microphone. Please check your browser settings and permissions.")
        }
      } else {
        alert(
          `Microphone error: ${error.message || "Unknown error occurred"}. Please check your browser settings and try again.`,
        )
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)

      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
      }
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const audioUrl = URL.createObjectURL(file)
      setAudioData({
        blob: file,
        url: audioUrl,
        name: file.name,
        duration: 0, // Will be updated when audio loads
      })
    }
  }

  const playAudio = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const analyzeAudio = useCallback(async () => {
    if (!audioData) return

    setIsAnalyzing(true)
    setAnalysisProgress(0)

    // Simulate analysis progress
    const progressInterval = setInterval(() => {
      setAnalysisProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval)
          return 100
        }
        return prev + 10
      })
    }, 500)

    try {
      // Simulate the analysis results without recursive calls
      setTimeout(() => {
        const mockResults = {
          duration: audioData.duration || 0,
          sampleRate: 44100,
          averageRMS: 0.0234,
          detectedSounds: 5,
          dominantFrequency: 440,
          maxDecibels: -12.5,
          soundEvents: [
            { time: 0.5, frequency: 440, amplitude: 0.8, type: "Voice" },
            { time: 1.2, frequency: 880, amplitude: 0.6, type: "Background" },
            { time: 2.1, frequency: 220, amplitude: 0.4, type: "Ambient" },
            { time: 3.0, frequency: 1760, amplitude: 0.7, type: "Noise" },
            { time: 4.2, frequency: 330, amplitude: 0.5, type: "Echo" },
          ],
          frequencySpectrum: Array.from({ length: 100 }, (_, i) => ({
            frequency: i * 220,
            magnitude: Math.random() * 0.8 + 0.1,
          })),
        }

        // Update audioData with analysis results without causing recursion
        setAudioData((currentData) => {
          if (currentData) {
            return { ...currentData, analysisResults: mockResults }
          }
          return currentData
        })

        setIsAnalyzing(false)
        clearInterval(progressInterval)
        setAnalysisProgress(100)
      }, 5000)
    } catch (error) {
      console.error("Analysis error:", error)
      setIsAnalyzing(false)
      clearInterval(progressInterval)
    }
  }, [audioData])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const checkMicrophoneSupport = () => {
    const isSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
    const isSecure = window.isSecureContext
    const hasMediaRecorder = typeof MediaRecorder !== "undefined"

    return {
      isSupported,
      isSecure,
      hasMediaRecorder,
      canRecord: isSupported && isSecure && hasMediaRecorder,
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="text-center py-8 px-4">
        <h1 className="text-4xl font-bold text-purple-600 mb-2">Audio Forensic Detector</h1>
        <p className="text-purple-500 text-lg mb-4">Advanced Audio Analysis & Instrument Detection System</p>
        <Badge variant="secondary" className="bg-orange-100 text-orange-800 border-orange-200">
          Local Mode
        </Badge>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-6xl mx-auto px-4 mb-8">
        <div className="flex flex-wrap justify-center gap-2">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === tab
                  ? "bg-purple-600 text-white shadow-lg"
                  : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {activeTab === "Record" && (
            <div className="text-center">
              <h2 className="text-2xl font-bold text-purple-600 mb-4">Audio Recording</h2>
              <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
                Record live audio for forensic analysis with instrument detection - data flows to all tabs
              </p>

              {/* Browser Compatibility Check */}
              {(() => {
                const support = checkMicrophoneSupport()
                if (!support.canRecord) {
                  return (
                    <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg max-w-2xl mx-auto">
                      <h3 className="font-semibold text-yellow-800 mb-2">Recording Requirements</h3>
                      <ul className="text-sm text-yellow-700 space-y-1">
                        {!support.isSecure && <li>• HTTPS connection required for microphone access</li>}
                        {!support.isSupported && <li>• Browser doesn't support audio recording</li>}
                        {!support.hasMediaRecorder && <li>• MediaRecorder API not available</li>}
                      </ul>
                    </div>
                  )
                }
                return null
              })()}

              <div className="space-y-6">
                <Button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!checkMicrophoneSupport().canRecord}
                  className={`px-8 py-4 text-lg font-medium rounded-lg transition-all ${
                    isRecording
                      ? "bg-red-600 hover:bg-red-700 text-white"
                      : "bg-purple-600 hover:bg-purple-700 text-white disabled:bg-gray-400 disabled:cursor-not-allowed"
                  }`}
                >
                  <Mic className="w-5 h-5 mr-2" />
                  {isRecording ? "Stop Recording" : "Start Recording"}
                </Button>

                {isRecording && (
                  <Card className="max-w-md mx-auto">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-center space-x-2 mb-4">
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-red-700 font-medium">Recording in progress...</span>
                      </div>
                      <div className="text-2xl font-mono text-center">{formatTime(recordingTime)}</div>
                    </CardContent>
                  </Card>
                )}

                {audioData && (
                  <Card className="max-w-md mx-auto">
                    <CardHeader>
                      <CardTitle className="text-lg">Recorded Audio</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <p className="text-sm text-gray-600">{audioData.name}</p>
                        <audio
                          ref={audioRef}
                          src={audioData.url}
                          onLoadedMetadata={(e) => {
                            const audio = e.target as HTMLAudioElement
                            setAudioData((prev) => (prev ? { ...prev, duration: Math.floor(audio.duration) } : null))
                          }}
                          onPlay={() => setIsPlaying(true)}
                          onPause={() => setIsPlaying(false)}
                          onEnded={() => setIsPlaying(false)}
                          controls
                          className="w-full"
                        />
                        <Button onClick={analyzeAudio} className="w-full">
                          Analyze Audio
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          )}

          {activeTab === "Upload" && (
            <div className="text-center">
              <h2 className="text-2xl font-bold text-purple-600 mb-4">Upload Audio File</h2>
              <p className="text-gray-600 mb-8">Upload audio files for forensic analysis and instrument detection</p>

              <div className="max-w-2xl mx-auto">
                <label className="block">
                  <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 hover:border-purple-400 transition-colors cursor-pointer">
                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-500">Drag and drop audio files here or click to browse</p>
                    <p className="text-sm text-gray-400 mt-2">Supports MP3, WAV, M4A, and other audio formats</p>
                  </div>
                </label>

                {audioData && (
                  <Card className="mt-6">
                    <CardHeader>
                      <CardTitle className="text-lg">Uploaded Audio</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <p className="text-sm text-gray-600">{audioData.name}</p>
                        <audio
                          src={audioData.url}
                          controls
                          className="w-full"
                          onLoadedMetadata={(e) => {
                            const audio = e.target as HTMLAudioElement
                            setAudioData((prev) => (prev ? { ...prev, duration: Math.floor(audio.duration) } : null))
                          }}
                        />
                        <Button onClick={analyzeAudio} className="w-full" disabled={isAnalyzing}>
                          {isAnalyzing ? "Analyzing..." : "Analyze Audio"}
                        </Button>
                        {isAnalyzing && (
                          <div className="space-y-2">
                            <Progress value={analysisProgress} className="w-full" />
                            <p className="text-sm text-gray-600 text-center">{analysisProgress}% Complete</p>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          )}

          {activeTab === "Analyze" && <AudioAnalysis audioData={audioData} />}

          {activeTab === "Sonar View" && <SonarView audioData={audioData} />}

          {/* Add Live Visualization below Sonar View */}
          {activeTab === "Sonar View" && audioData && (
            <div className="mt-8">
              <LiveVisualization audioData={audioData} />
            </div>
          )}

          {activeTab === "Sessions" && (
            <div className="text-center">
              <h2 className="text-2xl font-bold text-purple-600 mb-4">Analysis Sessions</h2>
              <p className="text-gray-600 mb-8">Manage and review previous forensic analysis sessions</p>
              <div className="bg-gray-100 rounded-lg p-8">
                <p className="text-gray-500">No previous sessions found</p>
              </div>
            </div>
          )}

          {activeTab === "Settings" && <AudioSettings />}
        </div>
      </div>
    </div>
  )
}
