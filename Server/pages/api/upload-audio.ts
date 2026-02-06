import type { NextApiRequest, NextApiResponse } from "next";

export const config = {
  api: {
    bodyParser: false, // important for sending audio files
  },
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const pythonServer = "http://localhost:8000/analyze";

  const response = await fetch(pythonServer, {
    method: "POST",
    headers: req.headers,
    body: req.body,
  });

  const result = await response.json();
  res.status(200).json(result);
}
