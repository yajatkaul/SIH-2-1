import express from "express";
import { getKeywords } from "../controller/ai.controller.js";
import upload from "../../utils/multer.js";

const router = express();

router.post("/uploadAudio", upload.single("audio"), getKeywords);

export default router;
