import express from "express";
import aiRoutes from "./router/ai.routes.js";
import cookieParser from "cookie-parser";
import cors from "cors";

const app = express();

app.use(
  cors({
    origin: "http://localhost:3000",
    credentials: true,
  })
);

app.use(express.json());
app.use(cookieParser());

app.use("/api/ai", aiRoutes);

app.listen(5000, () => {
  console.log("Server running at http://localhost:5000");
});
