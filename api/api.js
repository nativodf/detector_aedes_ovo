import express from "express";
import { exec } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = 3000;
const rootPath = path.resolve(__dirname, '../');

const storage = multer.diskStorage({ // upload configuration
    destination: (req, file, cb) => {
        cb(null, rootPath); 
    },
    filename: (req, file, cb) => {
        cb(null, "ovitrampa.jpg"); 
    }
});
const upload = multer({ storage });

app.use(express.static(__dirname));  

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.post("/upload", upload.single("file"), (req, res) => {
    res.send({ status: "ok" });
});

app.get("/api", (req, res) => {
    const pythonScriptPath = path.join(__dirname, "..", "detector_ovos", "main.py");
    const pythonPath = process.platform === "win32"
        ? path.join(__dirname, "..", "venv", "Scripts", "python.exe")
        : path.join(__dirname, "..", "venv", "bin", "python");


    exec(`"${pythonPath}" "${pythonScriptPath}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error running Python script: ${error.message}`);
            res.status(500).send("Error processing request.");
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
        }
        try {
            const json = JSON.parse(stdout); 
            res.json(json); // sends as JSON to the frontend
        } catch (e) {
            console.error(e.message);
            res.status(500).send("Error interpreting Python output.");
        }
    });
});

app.listen(PORT, () => {console.log(`The server is running on the port ${PORT}`)});