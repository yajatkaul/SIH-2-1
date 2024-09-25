import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const getKeywords = async (req, res) => {
  try {
    const filePath = req.file
      ? path.join(__dirname, "../../uploads", req.file.filename)
      : null;

    const response = await fetch("http://localhost:8000/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ file_path: filePath }),
    });

    const data = await response.json();

    res.status(200).json({ result: data.transcription });
  } catch (err) {
    console.log(err);
    res.status(500).json({ error: "Internal server error" });
  }
};
