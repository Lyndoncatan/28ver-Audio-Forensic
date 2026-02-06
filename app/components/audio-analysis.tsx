"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Download, Database, AlertCircle, CheckCircle } from "lucide-react"
import { generatePDFReport } from "@/lib/pdf-generator"
import { DatabaseService } from "@/lib/database"
import { getSupabaseStatus } from "@/lib/supabase"
import { useState, useEffect } from "react"

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
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false)
  const [pdfProgress, setPdfProgress] = useState(0)
  const [pdfStage, setPdfStage] = useState("")
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<string>("")
  const [dbStatus, setDbStatus] = useState<{ connected: boolean; message?: string }>({ connected: false })

  // Check database connection on component mount
  useEffect(() => {
    const checkDatabaseConnection = async () => {
      const supabaseStatus = getSupabaseStatus()
      if (!supabaseStatus.connected) {
        setDbStatus({
          connected: false,
          message: "Supabase not configured",
        })
        return
      }

      try {
        const testResult = await DatabaseService.testConnection()
        setDbStatus({
          connected: testResult.success,
          message: testResult.message,
        })
      } catch (error) {
        setDbStatus({
          connected: false,
          message: "Connection test failed",
        })
      }
    }

    checkDatabaseConnection()
  }, [])

  const handleDownloadPDF = async () => {
    if (!audioData?.analysisResults) {
      alert("No analysis results available for PDF generation")
      return
    }

    setIsGeneratingPDF(true)
    setPdfProgress(0)
    setPdfStage("Starting PDF generation...")

    try {
      await generatePDFReport(audioData, (progress, stage) => {
        setPdfProgress(progress)
        setPdfStage(stage)
      })

      setSaveStatus("✅ Comprehensive PDF report generated successfully!")
      setPdfStage("PDF generation completed!")
      setTimeout(() => {
        setSaveStatus("")
        setPdfStage("")
      }, 3000)
    } catch (error) {
      console.error("PDF generation error:", error)
      setSaveStatus("❌ Failed to generate PDF report")
      setPdfStage("PDF generation failed")
      setTimeout(() => {
        setSaveStatus("")
        setPdfStage("")
      }, 3000)
    } finally {
      setIsGeneratingPDF(false)
      setPdfProgress(0)
    }
  }

  const handleSaveToDatabase = async () => {
    if (!audioData?.analysisResults) {
      alert("No analysis results available to save")
      return
    }

    if (!dbStatus.connected) {
      alert("Database is not available. Please check your Supabase configuration.")
      return
    }

    setIsSaving(true)
    try {
      const analysisId = await DatabaseService.saveAnalysis(audioData)
      if (analysisId) {
        setSaveStatus("✅ Analysis saved to database successfully!")
      } else {
        setSaveStatus("❌ Failed to save analysis to database")
      }
      setTimeout(() => setSaveStatus(""), 3000)
    } catch (error: any) {
      console.error("Database save error:", error)
      setSaveStatus(`❌ Database error: ${error.message || "Unknown error"}`)
      setTimeout(() => setSaveStatus(""), 5000)
    } finally {
      setIsSaving(false)
    }
  }

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

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-purple-600">Audio Analysis Results</h2>

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <Button
            onClick={handleDownloadPDF}
            disabled={isGeneratingPDF}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            <Download className="w-4 h-4 mr-2" />
            {isGeneratingPDF ? "Generating..." : "Download Full Report PDF"}
          </Button>

          <Button
            onClick={handleSaveToDatabase}
            disabled={isSaving || !dbStatus.connected}
            className={`text-white ${
              dbStatus.connected ? "bg-green-600 hover:bg-green-700" : "bg-gray-400 cursor-not-allowed"
            }`}
            title={!dbStatus.connected ? "Database not available" : "Save to Supabase database"}
          >
            <Database className="w-4 h-4 mr-2" />
            {isSaving ? "Saving..." : "Save to Database"}
          </Button>
        </div>
      </div>

      {/* PDF Generation Progress */}
      {isGeneratingPDF && (
        <Card className="mb-4">
          <CardContent className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-purple-700">Generating Comprehensive PDF Report</span>
                <span className="text-sm text-purple-600">{pdfProgress}%</span>
              </div>
              <Progress value={pdfProgress} className="w-full" />
              <p className="text-xs text-gray-600">{pdfStage}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Database Status */}
      <div className="mb-4 p-3 rounded-lg border flex items-center space-x-2">
        {dbStatus.connected ? (
          <>
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-green-700 text-sm">Database Connected</span>
          </>
        ) : (
          <>
            <AlertCircle className="w-4 h-4 text-orange-600" />
            <span className="text-orange-700 text-sm">
              Database Offline - {dbStatus.message || "Connection unavailable"}
            </span>
          </>
        )}
      </div>

      {/* Status Messages */}
      {saveStatus && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-700 text-sm">{saveStatus}</p>
        </div>
      )}

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

      {/* Export Information */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-lg">Export & Storage Options</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-red-50 rounded-lg border border-red-200">
              <h4 className="font-semibold text-red-800 mb-2">📄 Comprehensive PDF Report</h4>
              <p className="text-sm text-red-700 mb-3">
                Generate a detailed forensic analysis report with all findings, statistical analysis, and expert
                conclusions.
              </p>
              <ul className="text-xs text-red-600 space-y-1">
                <li>• Complete analysis summary with technical specifications</li>
                <li>• Detailed sound event timeline and classification</li>
                <li>• Advanced frequency analysis with visual charts</li>
                <li>• Statistical analysis and expert conclusions</li>
                <li>• Professional formatting suitable for reports</li>
                <li>• Real-time generation progress tracking</li>
              </ul>
            </div>

            <div
              className={`p-4 rounded-lg border ${
                dbStatus.connected ? "bg-green-50 border-green-200" : "bg-gray-50 border-gray-200"
              }`}
            >
              <h4 className={`font-semibold mb-2 ${dbStatus.connected ? "text-green-800" : "text-gray-600"}`}>
                💾 Database Storage
              </h4>
              <p className={`text-sm mb-3 ${dbStatus.connected ? "text-green-700" : "text-gray-600"}`}>
                {dbStatus.connected
                  ? "Save analysis results to Supabase database for future reference and comparison."
                  : "Database storage is currently unavailable. Please check your Supabase configuration."}
              </p>
              <ul className={`text-xs space-y-1 ${dbStatus.connected ? "text-green-600" : "text-gray-500"}`}>
                <li>• Persistent cloud storage</li>
                <li>• Search and filter capabilities</li>
                <li>• Historical analysis tracking</li>
                <li>• Secure backup and recovery</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
