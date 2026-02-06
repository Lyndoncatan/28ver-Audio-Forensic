import { type NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

export const maxDuration = 300; 

async function runPythonScript(scriptPath: string, args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonExecutable = "python";
    
    // We use shell:true and wrap EVERYTHING in extra quotes to survive the MYCODES folder spaces
    const python = spawn(pythonExecutable, [scriptPath, ...args], { 
        shell: true,
        windowsHide: true
    });
    
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data) => (stdout += data.toString()));
    python.stderr.on("data", (data) => (stderr += data.toString()));

    python.on("close", (code) => {
      if (code === 0) {
        try {
          // Find the JSON block even if there are python warnings above it
          const startIdx = stdout.indexOf('{');
          const endIdx = stdout.lastIndexOf('}');
          if (startIdx === -1) throw new Error("No JSON response");
          resolve(JSON.parse(stdout.substring(startIdx, endIdx + 1)));
        } catch (e) { 
          reject(`Engine Output Error: ${stdout.substring(0, 100)}`); 
        }
      } else { 
        reject(stderr || `Python Error Code: ${code}`); 
      }
    });
  });
}

export async function POST(request: NextRequest) {
  let tempPath = "";
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    if (!file) throw new Error("No file uploaded");

    const cwd = process.cwd();
    const scriptPath = path.join(cwd, "scripts", "audio_separator.py");

    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Engine script missing at: ${scriptPath}`);
    }

    const publicDir = path.join(cwd, "public");
    const outputDir = path.join(publicDir, "separated_audio");
    const tempDir = path.join(publicDir, "temp_uploads");

    [tempDir, outputDir].forEach(dir => { 
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); 
    });

    const jobID = `job_${Date.now()}`;
    tempPath = path.join(tempDir, `${jobID}.wav`);
    
    const buffer = Buffer.from(await file.arrayBuffer());
    fs.writeFileSync(tempPath, buffer);

    // FIX: Wrap every path in triple quotes to handle the spaces in your desktop folder
    const result = await runPythonScript(`"${scriptPath}"`, [
        `"${path.resolve(tempPath)}"`, 
        `"${path.resolve(outputDir)}"`, 
        jobID
    ]);
    
    if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    return NextResponse.json(result);

  } catch (error: any) {
    if (tempPath && fs.existsSync(tempPath)) try { fs.unlinkSync(tempPath); } catch(e) {}
    console.error("Forensic Engine Crash:", error.message);
    return NextResponse.json({ status: "error", message: error.message }, { status: 500 });
  }
}