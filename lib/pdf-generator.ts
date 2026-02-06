import jsPDF from "jspdf"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

// New interface for passing canvas captures
interface ReportVisuals {
  radarCanvas?: string; // Base64 string
  topographyCanvas?: string; // Base64 string
}

type ProgressCallback = (progress: number, stage: string) => void

export const generatePDFReport = async (
  audioData: AudioData, 
  onProgress?: ProgressCallback,
  visuals?: ReportVisuals // Added visuals parameter
): Promise<void> => {
  if (!audioData.analysisResults) {
    throw new Error("No analysis results available for PDF generation")
  }

  try {
    const autoTable = (await import("jspdf-autotable")).default
    const doc = new jsPDF()
    const results = audioData.analysisResults

    const updateProgress = (progress: number, stage: string) => {
      if (onProgress) onProgress(progress, stage)
    }

    updateProgress(10, "Initializing Forensic PDF...")

    // --- Page 1: Metadata & Visuals ---
    doc.setFont("helvetica", "bold")
    doc.setFontSize(22)
    doc.setTextColor(128, 0, 128)
    doc.text("Forensic Audio Intelligence Report", 20, 25)
    doc.line(20, 30, 190, 30)

    let yPosition = 45
    
    // Summary Table
    autoTable(doc, {
      startY: yPosition,
      head: [["Forensic Metadata", "Value"]],
      body: [
        ["Case Source", audioData.name],
        ["Total Duration", `${results.duration?.toFixed(2)}s`],
        ["Detected Speakers", new Set(results.diarization?.segments.map((s: any) => s.speaker)).size.toString()],
      ],
      theme: "grid",
      headStyles: { fillColor: [128, 0, 128] },
    })

    yPosition = (doc as any).lastAutoTable.finalY + 15

    // --- ADDING THE VISUAL SNAPSHOTS ---
    if (visuals?.radarCanvas || visuals?.topographyCanvas) {
      doc.setFontSize(14)
      doc.text("Spatial Evidence Snapshots", 20, yPosition)
      yPosition += 10

      // Add Radar View
      if (visuals.radarCanvas) {
        doc.setFontSize(10);
        doc.text("Exhibit A: Spatial Radar Projection", 20, yPosition);
        doc.addImage(visuals.radarCanvas, 'PNG', 20, yPosition + 5, 80, 60);
      }

      // Add Topography View
      if (visuals.topographyCanvas) {
        doc.text("Exhibit B: Voxel Topography", 110, yPosition);
        doc.addImage(visuals.topographyCanvas, 'PNG', 110, yPosition + 5, 80, 60);
      }
      
      yPosition += 75; // Advance Y after images
    }

    updateProgress(40, "Exporting Speaker Diarization...");

    // --- Speaker Diarization Table ---
    // (Existing diarization table logic here...)

    // ... [Rest of the existing PDF generation code] ...

    updateProgress(100, "Finalizing Report...")
    doc.save(`Forensic_Analysis_${Date.now()}.pdf`)

  } catch (error) {
    console.error("PDF Error:", error)
  }
}