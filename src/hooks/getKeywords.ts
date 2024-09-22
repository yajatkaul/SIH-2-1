import { useState } from "react";

export const useKeyWords = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState();
  const send = async (file: File) => {
    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("audio", file);

      const res = await fetch("http://localhost:5000/api/ai/uploadAudio", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.ok) {
        throw new Error(data.ok);
      }

      setData(data.result);
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  };

  return { loading, send, data };
};
