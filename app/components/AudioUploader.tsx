import { useState } from "react";

export default function AudioUploader() {
  const [result, setResult] = useState<any>(null);

  async function handleUpload(e: any) {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/upload-audio", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
  }

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleUpload} />

      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
