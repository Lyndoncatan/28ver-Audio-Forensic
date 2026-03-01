import { type NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";

export const maxDuration = 300;

async function runPython(scriptName: string, args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", scriptName);
    // Use double quotes for paths that might contain spaces
    const quotedScriptPath = `"${scriptPath}"`;
    const formattedArgs = args.map(arg => arg.startsWith('"') ? arg : `"${arg}"`);

    console.log(`[Forensic] Spawning: python ${quotedScriptPath} ${formattedArgs.join(" ")}`);

    const python = spawn("python", [quotedScriptPath, ...formattedArgs], {
      shell: true,
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";

    const timeout = setTimeout(() => {
      console.error(`[Forensic] Timeout executing ${scriptName}`);
      python.kill();
      reject(new Error(`Timeout executing ${scriptName}. The operation took too long (20 min).`));
    }, 1200000); // 20 Minutes

    python.stdout.on("data", (data) => (stdout += data.toString()));
    python.stderr.on("data", (data) => {
      const msg = data.toString();
      stderr += msg;
      console.error(`[Python-Stderr] ${msg}`);
    });

    python.on("close", (code) => {
      clearTimeout(timeout);

      if (code !== 0) {
        return reject(new Error(`Python process exited with code ${code}. Error: ${stderr}`));
      }

      try {
        const start = stdout.indexOf('{');
        const end = stdout.lastIndexOf('}');
        if (start === -1) {
          return reject(new Error(`No JSON found in Python output. Stderr: ${stderr}`));
        }
        resolve(JSON.parse(stdout.substring(start, end + 1)));
      } catch (e: any) {
        reject(new Error(`Parse error: ${e.message}. Output: ${stdout.substring(0, 500)}...`));
      }
    });

    python.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

export async function POST(request: NextRequest) {
  let tempFilePath = "";
  let audioDataToUse = ""; // Store for fallback

  try {
    const contentType = request.headers.get("content-type") || "";
    let jobID = "";

    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("audio") as File;
      if (!file) throw new Error("No file uploaded");

      jobID = file.name ? file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `job_${Date.now()}`;

      const tempDir = os.tmpdir();
      tempFilePath = path.join(tempDir, `${jobID}_input.wav`);

      const arrayBuffer = await file.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      fs.writeFileSync(tempFilePath, buffer);

      // Store base64 for fallback
      audioDataToUse = buffer.toString('base64');

    } else {
      // JSON Handling (Base64)
      const { audioData, filename } = await request.json();
      audioDataToUse = audioData; // Store for fallback

      jobID = filename ? filename.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `job_${Date.now()}`;

      const tempDir = os.tmpdir();
      tempFilePath = path.join(tempDir, `${jobID}_input.wav`);
      fs.writeFileSync(tempFilePath, Buffer.from(audioData, 'base64'));
    }

    const outputDir = path.join(process.cwd(), "public", "separated_audio");
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    // Pass the PATH to the file, not the actual audio string
    const classification = await runPython("mediapipe_audio_classifier.py", [
      `"${tempFilePath}"`,
      `"${jobID}"`
    ]);

    // Save classification to a file so audio_separator can derive stems from it
    const classificationPath = path.join(os.tmpdir(), `${jobID}_classification.json`);
    fs.writeFileSync(classificationPath, JSON.stringify(classification));

    const separation = await runPython("audio_separator.py", [
      `"${tempFilePath}"`,
      `"${outputDir}"`,
      `"${jobID}"`,
      `"${classificationPath}"`
    ]);

    return NextResponse.json({
      status: "Success",
      jobID,
      classification,
      stems: separation.stems,
      debug: separation.debug // Return debug info for inspection
    });

  } catch (error: any) {
    console.error("Forensic Engine Error:", error.message);
    return NextResponse.json(
      {
        status: "Error",
        error: error.message,
        details: "The forensic engine failed to process the audio. Check backend logs."
      },
      { status: 500 }
    );
  } finally {
    // Cleanup the temp file after processing is done
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try { fs.unlinkSync(tempFilePath); } catch (e) { }
    }
  }
}