const express = require('express');
const fs = require('fs');
const cors = require('cors');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();
const PORT = 3000;
const dataDir = '/home/ilija/data/bimanual-annotate/mp4s';
const annotationDir = '/home/ilija/data/bimanual-annotate/annotations';

app.use(cors());

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'build'))); // Serve static files (React build and videos)

app.use('/videos', express.static(dataDir));

// Endpoint to list directories
app.get('/directories', (req, res) => {
    fs.readdir(dataDir, { withFileTypes: true }, (err, files) => {
      if (err) {
        return res.status(500).send(err);
      }
      const directories = files.filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
      console.log('list of directories', directories);
      res.json(directories);
    });
});

// Endpoint to get videos in a directory
app.get('/videos/:dir', (req, res) => {
    const dirPath = path.join(dataDir, req.params.dir);
    fs.readdir(dirPath, (err, files) => {
      if (err) {
        return res.status(500).send(err);
      }
      const videos = files.filter(file => file.endsWith('.mp4'));
      res.json(videos);
    });
});

app.post('/get-annotations', (req, res) => {
    const { datasetName, videoName } = req.body;
    const dirPath = path.join(annotationDir, `${datasetName}`); // Make sure this directory exists or is created
    const filePath = path.join(dirPath, `${videoName}.json`);

    console.log('Getting annotations for:', filePath)

    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            // console.log(err);
            res.json([]);
        } else {
            // Read file if it exists
            fs.readFile(filePath, 'utf8', (readErr, data) => {
                if (readErr) {
                    // console.error(readErr);
                    res.json([]);
                }
                else {
                    res.json(JSON.parse(data));
                }
            });
        }
    });
});


app.post('/save-annotations', (req, res) => {
    const { annotations, datasetName, videoName } = req.body;
    const dirPath = path.join(annotationDir, `${datasetName}`); // Make sure this directory exists or is created
    const filePath = path.join(dirPath, `${videoName}.json`);

    fs.mkdirSync(dirPath, { recursive: true }); // Ensure the directory exists

    fs.writeFile(filePath, JSON.stringify(annotations, null, 2), (err) => {
        if (err) {
            console.error('Failed to save file:', err);
            return res.status(500).send('Failed to save annotations');
        }
	    console.log("Saving to:", filePath);
        res.send({ message: 'Annotations saved successfully to: ' + filePath });
    });
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

