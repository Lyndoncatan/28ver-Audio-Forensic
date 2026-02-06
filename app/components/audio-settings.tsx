"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"

export default function AudioSettings() {
  const [sensitivity, setSensitivity] = useState([75])
  const [noiseReduction, setNoiseReduction] = useState([60])
  const [frequencyRange, setFrequencyRange] = useState([20, 20000])
  const [analysisMode, setAnalysisMode] = useState("comprehensive")

  return (
    <div>
      <h2 className="text-2xl font-bold text-purple-600 mb-6 text-center">Audio Analysis Settings</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Detection Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Detection Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">Detection Sensitivity: {sensitivity[0]}%</label>
              <Slider
                value={sensitivity}
                onValueChange={setSensitivity}
                max={100}
                min={1}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-gray-600 mt-1">Higher values detect more subtle sounds</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Noise Reduction: {noiseReduction[0]}%</label>
              <Slider
                value={noiseReduction}
                onValueChange={setNoiseReduction}
                max={100}
                min={0}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-gray-600 mt-1">Filters out background noise</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Frequency Range: {frequencyRange[0]} - {frequencyRange[1]} Hz
              </label>
              <div className="px-3">
                <Slider
                  value={frequencyRange}
                  onValueChange={setFrequencyRange}
                  max={22050}
                  min={1}
                  step={10}
                  className="w-full"
                />
              </div>
              <p className="text-xs text-gray-600 mt-1">Frequency range for analysis</p>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Mode */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Analysis Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-3">Analysis Mode</label>
              <div className="space-y-2">
                {[
                  { id: "quick", name: "Quick Scan", desc: "Fast basic analysis" },
                  { id: "comprehensive", name: "Comprehensive", desc: "Detailed forensic analysis" },
                  { id: "realtime", name: "Real-time", desc: "Live monitoring mode" },
                ].map((mode) => (
                  <div
                    key={mode.id}
                    className={`p-3 border rounded-lg cursor-pointer transition-all ${
                      analysisMode === mode.id
                        ? "border-purple-500 bg-purple-50"
                        : "border-gray-200 hover:border-gray-300"
                    }`}
                    onClick={() => setAnalysisMode(mode.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{mode.name}</p>
                        <p className="text-sm text-gray-600">{mode.desc}</p>
                      </div>
                      {analysisMode === mode.id && <Badge variant="default">Active</Badge>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Output Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Output Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span className="text-sm">Generate detailed report</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span className="text-sm">Export frequency spectrum</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span className="text-sm">Save sound event timeline</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">Enable audio enhancement</span>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* System Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">System Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Analysis Engine:</span>
                <Badge variant="outline">LibROSA v0.10.1</Badge>
              </div>
              <div className="flex justify-between">
                <span>Sample Rate:</span>
                <span>44.1 kHz</span>
              </div>
              <div className="flex justify-between">
                <span>Bit Depth:</span>
                <span>16-bit</span>
              </div>
              <div className="flex justify-between">
                <span>Processing:</span>
                <Badge variant="outline" className="bg-green-50 text-green-700">
                  Real-time
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8 text-center">
        <Button className="px-8 py-2">Save Settings</Button>
      </div>
    </div>
  )
}
