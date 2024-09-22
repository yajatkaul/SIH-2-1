import path from "path";
import { fileURLToPath } from "url";
import ollama from "ollama";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const getKeywords = async (req, res) => {
  try {
    const filePath = req.file
      ? path.join(__dirname, "../../uploads", req.file.filename)
      : null;

    const response = await fetch("http://localhost:8000/transcribe", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ file_path: filePath }),
    });

    const data = await response.json();

    console.log(data.transcription);

    const ollamaResponse = await ollama.chat({
      model: "gemma2:2b",
      messages: [
        {
          role: "user",
          content: `I'm making a threat detection system. Please analyze the following text and provide a list of bullet points containing keywords for any content that may be illegal or unethical. Only include the bullet points with keywords, no other text:\n\n${data.transcription}`,
        },
      ],
    });

    console.log(ollamaResponse.message.content);
    res.status(200).json({ result: ollamaResponse.message.content });
  } catch (err) {
    console.log(err);
    res.status(500).json({ error: "Internal server error" });
  }
};
