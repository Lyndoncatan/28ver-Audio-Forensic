"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

interface AudioAnalysisProps {
  audioData: AudioData | null
}

export default function AudioAnalysis({ audioData }: AudioAnalysisProps) {
  if (!audioData) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Audio Analysis</h2>
        <p className="text-gray-600 mb-8">Upload or record audio to begin forensic analysis</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">No audio data available for analysis</p>
        </div>
      </div>
    )
  }

  if (!audioData.analysisResults) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Audio Analysis</h2>
        <p className="text-gray-600 mb-8">Click "Analyze Audio" to process the uploaded file</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">Analysis not yet performed</p>
        </div>
      </div>
    )
  }

  const results = audioData.analysisResults

  // Add safety checks for all data access
  return (
    <div>
      <h2 className="text-2xl font-bold text-purple-600 mb-6 text-center">Audio Analysis Results</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {/* Basic Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">File Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>File:</strong> {audioData.name}
              </p>
              <p>
                <strong>Duration:</strong> {results?.duration || 0}s
              </p>
              <p>
                <strong>Sample Rate:</strong> {results?.sampleRate || 0} Hz
              </p>
              <p>
                <strong>Average RMS:</strong> {results?.averageRMS?.toFixed(4) || "0.0000"}
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Sound Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Sound Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>Detected Events:</strong> {results?.detectedSounds || 0}
              </p>
              <p>
                <strong>Dominant Frequency:</strong> {results?.dominantFrequency || 0} Hz
              </p>
              <p>
                <strong>Max Decibels:</strong> {results?.maxDecibels || 0} dB
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Loudest Sound */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Loudest Sound</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {results?.soundEvents && results.soundEvents.length > 0 ? (
                <>
                  <p>
                    <strong>Type:</strong> {results.soundEvents[0].type}
                  </p>
                  <p>
                    <strong>Time:</strong> {results.soundEvents[0].time}s
                  </p>
                  <p>
                    <strong>Frequency:</strong> {results.soundEvents[0].frequency} Hz
                  </p>
                  <p>
                    <strong>Amplitude:</strong> {(results.soundEvents[0].amplitude * 100).toFixed(1)}%
                  </p>
                </>
              ) : (
                <p className="text-gray-500">No sound events detected</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sound Events Timeline */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg">Detected Sound Events</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {results?.soundEvents && results.soundEvents.length > 0 ? (
              results.soundEvents.map((event: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <Badge variant="outline">{event.type}</Badge>
                    <span className="font-mono text-sm">{event.time}s</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm">{event.frequency} Hz</span>
                    <Progress value={event.amplitude * 100} className="w-20" />
                    <span className="text-sm font-medium">{(event.amplitude * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center">No sound events to display</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Frequency Spectrum */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Frequency Spectrum</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-gray-100 rounded-lg flex items-end justify-center p-4">
            <div className="flex items-end space-x-1 h-full w-full max-w-2xl">
              {results?.frequencySpectrum && results.frequencySpectrum.length > 0 ? (
                results.frequencySpectrum.slice(0, 50).map((point: any, index: number) => (
                  <div
                    key={index}
                    className="bg-purple-500 rounded-t"
                    style={{
                      height: `${(point.magnitude || 0) * 100}%`,
                      width: "100%",
                      minHeight: "2px",
                    }}
                    title={`${point.frequency} Hz: ${((point.magnitude || 0) * 100).toFixed(1)}%`}
                  />
                ))
              ) : (
                <div className="flex items-center justify-center w-full h-full">
                  <p className="text-gray-500">No frequency data available</p>
                </div>
              )}
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-2 text-center">
            Frequency range: 0 - {results?.frequencySpectrum?.[results.frequencySpectrum.length - 1]?.frequency || 0} Hz
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
