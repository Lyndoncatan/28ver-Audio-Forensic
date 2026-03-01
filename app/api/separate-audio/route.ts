import { type NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";

export const maxDuration = 900;

async function runPython(scriptName: string, args: string[]): Promise<any> {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(process.cwd(), "scripts", scriptName);
        const quotedScriptPath = `"${scriptPath}"`;
        const formattedArgs = args.map(arg => arg.startsWith('"') ? arg : `"${arg}"`);

        console.log(`[Forensic-Separation] Spawning: python ${quotedScriptPath} ${formattedArgs.join(" ")}`);

        const python = spawn("python", [quotedScriptPath, ...formattedArgs], {
            shell: true,
            windowsHide: true
        });

        let stdout = "";
        let stderr = "";

        const timeout = setTimeout(() => {
            python.kill();
            reject(new Error(`Timeout executing ${scriptName}. Engine took too long.`));
        }, 1200000); // 20 Minutes

        python.stdout.on("data", (data) => {
            const msg = data.toString();
            stdout += msg;
            // Only log if NOT part of the JSON payload
            if (!msg.includes("[JSON_START]")) {
                process.stdout.write(msg);
            }
        });
        python.stderr.on("data", (data) => {
            const msg = data.toString();
            stderr += msg;
            process.stderr.write(msg);
        });

        python.on("close", (code) => {
            clearTimeout(timeout);
            if (code !== 0) return reject(new Error(`Exit ${code}: ${stderr}`));
            try {
                const startMarker = "[JSON_START]";
                const endMarker = "[JSON_END]";
                const startIndex = stdout.indexOf(startMarker);
                const endIndex = stdout.lastIndexOf(endMarker);

                if (startIndex !== -1 && endIndex !== -1) {
                    const jsonStr = stdout.substring(startIndex + startMarker.length, endIndex);
                    resolve(JSON.parse(jsonStr));
                } else {
                    // Fallback
                    const start = stdout.indexOf('{');
                    const end = stdout.lastIndexOf('}');
                    resolve(JSON.parse(stdout.substring(start, end + 1)));
                }
            } catch (e: any) {
                reject(new Error(`Parse error: ${e.message}`));
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
    let classificationPath = "";

    try {
        const formData = await request.formData();
        const file = formData.get("audio") as File;
        const classificationJson = formData.get("classification") as string;

        if (!file || !classificationJson) throw new Error("Missing audio or classification data");

        const jobID = file.name ? file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `sep_${Date.now()}`;
        const tempDir = os.tmpdir();

        // 1. Save Audio to Temp
        tempFilePath = path.join(tempDir, `${jobID}_sep_input.wav`);
        const arrayBuffer = await file.arrayBuffer();
        fs.writeFileSync(tempFilePath, Buffer.from(arrayBuffer));

        // 2. Save Classification to Temp (so python can read it)
        classificationPath = path.join(tempDir, `${jobID}_sep_class.json`);
        fs.writeFileSync(classificationPath, classificationJson);

        const outputDir = path.join(process.cwd(), "public", "separated_audio");
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

        // 3. Run Separation
        const separation = await runPython("audio_separator.py", [
            `"${tempFilePath}"`,
            `"${outputDir}"`,
            `"${jobID}"`,
            `"${classificationPath}"`
        ]);

        return NextResponse.json({
            status: "Success",
            stems: separation.stems,
            debug: separation.debug
        });

    } catch (error: any) {
        console.error("Separation Error:", error.message);
        return NextResponse.json({ status: "Error", error: error.message }, { status: 500 });
    } finally {
        if (tempFilePath && fs.existsSync(tempFilePath)) try { fs.unlinkSync(tempFilePath); } catch (e) { }
        if (classificationPath && fs.existsSync(classificationPath)) try { fs.unlinkSync(classificationPath); } catch (e) { }
    }
}
