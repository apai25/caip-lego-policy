import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import AnnotationModal from './AnnotationModal';
import { useParams, useNavigate } from 'react-router-dom';

function frameToTimestamp(frameNumber, frameRate = 30.0) {
    // Calculate total seconds from the frame number
    let totalSeconds = frameNumber / frameRate;
    
    // Calculate minutes and seconds from total seconds
    let minutes = Math.floor(totalSeconds / 60);
    let seconds = Math.floor(totalSeconds % 60);
    
    // Format seconds to have leading zero if less than 10
    let formattedSeconds = seconds < 10 ? `0${seconds}` : seconds;

    // Return timestamp in "minutes:seconds" format
    return `${minutes}:${formattedSeconds}`;
}

function VideoPlayer() {
    const videoRef = useRef(null);
    const [annotations, setAnnotations] = useState([]);
    const [segmentStart, setSegmentStart] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const frameRate = 30.0; // Adjust as needed for your videos

    const [directories, setDirectories] = useState([]);
    const [selectedDir, setSelectedDir] = useState('');
    const [videos, setVideos] = useState([]);
    const [currentVideo, setCurrentVideo] = useState('');

    const dropdownRef = useRef(null);
    const navigate = useNavigate();
    const { selectedDirectory, selectedVideo } = useParams();


    useEffect(() => {
        axios.get('/directories')
          .then(response => {
            setDirectories(response.data);
            if (selectedDirectory && response.data.includes(selectedDirectory)) {
                console.log("selectedDirectory", selectedDirectory, selectedVideo);
                setSelectedDir(selectedDirectory);
                fetchVideos(selectedDirectory, selectedVideo);
            }
          })
          .catch(error => console.error('Error fetching directories:', error));
      }, [selectedDirectory, selectedVideo]);
    
    // useEffect(() => {
    // if (selectedDir) {
    //     axios.get(`/videos/${selectedDir}`)
    //     .then(response => {
    //         console.log('all videos:', response.data);
    //         setVideos(response.data);
    //         if (response.data.length > 0) {
    //             console.log('Setting current video:', response.data[0]);
    //             setCurrentVideo(response.data[0]);
    //             const datasetName = String(selectedDir);
    //             const videoName = response.data[0].split('.')[0];
    //             fetch('/get-annotations', {
    //                     method: 'POST',
    //                     headers: {
    //                         'Content-Type': 'application/json'
    //                     },
    //                     body: JSON.stringify({ datasetName, videoName })
    //                 })
    //                 .then(response => response.json())
    //                 .then(data => {
    //                     console.log('Found annotations:', data);
    //                     setAnnotations(data);
    //                 })
    //                 .catch((error) => {
    //                     console.error('Error:', error);
    //                     setAnnotations([]);
    //                 });
    //         }
    //     })
    //     .catch(error => console.error('Error fetching videos:', error));
    // }
    // }, [selectedDir]);
    

    useEffect(() => {
        const handleKeyPress = (event) => {
            if (!isModalOpen) {
                switch (event.key) {
                    case ' ':
                        handlePlayPause();
                        event.preventDefault();
                        break;
                    case 's':
                        startSegment();
                        break;
                    case 'f':
                        finishSegment();
                        event.preventDefault();
                        break;
                    case 'n':
                        changeVideo(currentVideo, 'next');
                        break;
                    case 'b':
                        changeVideo(currentVideo, 'prev');
                        break;
                    default:
                        break;
                }
            }
        };
        document.addEventListener('keydown', handleKeyPress);
        return () => {
            document.removeEventListener('keydown', handleKeyPress);
        };
    }, [isModalOpen, currentVideo]);


    const fetchVideos = (dir, vid) => {
        axios.get(`/videos/${dir}`)
            .then(response => {
                setVideos(response.data);
                if (response.data.length > 0) {
                    let selvid;
                    if(vid && response.data.includes(`${vid}.mp4`)) {
                        console.log('Setting current video from url:', vid);
                        setCurrentVideo(`${vid}.mp4`);
                        selvid = vid;
                    } else {
                        console.log('Setting first video in dir:', response.data[0]);
                        setCurrentVideo(response.data[0]);
                        selvid = response.data[0].split('.')[0];
                    }
                    const datasetName = String(dir);
                    const videoName = String(selvid);
                    console.log('Fetching annotations for:', datasetName, videoName);
                    fetch('/get-annotations', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ datasetName, videoName })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Found annotations:', data);
                            setAnnotations(data);
                            navigate(`/${dir}/${videoName}`);
                        })
                        .catch((error) => {
                            console.error('Error:', error);
                            setAnnotations([]);
                            navigate(`/${dir}/${videoName}`);
                        });
                }
            })
            .catch(error => console.error('Error fetching videos:', error));
    };

    const handleDirectoryChange = (event) => {
        setSelectedDir(event.target.value);
        if (dropdownRef.current) {
            dropdownRef.current.blur(); // Step 3: Blur the dropdown
        }
        fetchVideos(event.target.value);
    };

    const changeVideo = (curVid, direction) => {
        const currentIndex = videos.indexOf(curVid);
        const nextIndex = direction === 'next' ? currentIndex + 1 : currentIndex - 1;
        console.log('curidx, nextidx', currentIndex, curVid, nextIndex);
        const datasetName = String(selectedDir);
        
        if (nextIndex >= 0 && nextIndex < videos.length) {
            console.log('Changing video:', videos[nextIndex]);
          setCurrentVideo(videos[nextIndex]);
          const videoName = videos[nextIndex].split('.')[0];

          fetch('/get-annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ datasetName, videoName })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Found annotations:', data);
                setAnnotations(data);
                navigate(`/${datasetName}/${videoName}`);
            })
            .catch((error) => {
                console.error('Error:', error);
                setAnnotations([]);
                navigate(`/${datasetName}/${videoName}`);
            });
        }
      };

    const handlePlayPause = () => {
        if (videoRef.current.paused) {
            videoRef.current.play();
        } else {
            videoRef.current.pause();
        }
    };

    // const handleScrub = (e) => {
    //     const time = (e.target.value / 100) * videoRef.current.duration;
    //     videoRef.current.currentTime = time;
    // };

    const startSegment = () => {
        if (segmentStart !== null) {
            setSegmentStart(null);
        } else {
            const frameNumber = Math.round(videoRef.current.currentTime * frameRate);
            setSegmentStart(frameNumber);
        }
    };

    const finishSegment = () => {
        handlePlayPause();
        setIsModalOpen(true);
    };

    const handleDeleteAnnotation = (index) => {
        const newAnnotations = annotations.filter((_, i) => i !== index);
        setAnnotations(newAnnotations);
        const videoName = videoRef.current.src.split('/').pop().split('.')[0];
        const datasetName = String(selectedDir);

        console.log('annotations after deleting: ', newAnnotations);

        fetch('/save-annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ annotations: newAnnotations, datasetName, videoName })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    };
 
    const saveAnnotation = (text, failed) => {
        const newAnnotation = { start: segmentStart, end: Math.round(videoRef.current.currentTime * frameRate), text, failed };
        const updatedAnnotations = [...annotations, newAnnotation];
        console.log('before annotation', annotations);
        setAnnotations(updatedAnnotations);
        console.log('after annotation', annotations);
        setSegmentStart(null);
        setIsModalOpen(false);
        handlePlayPause();

        // Extract video file name without extension
        const videoName = videoRef.current.src.split('/').pop().split('.')[0];
        const datasetName = String(selectedDir);
        console.log('Video name:', videoName);
        console.log('Selected dataset:', datasetName);
        console.log('Updated annotations', updatedAnnotations);
	    
        // Send the data to the server
        fetch('/save-annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ annotations: updatedAnnotations, datasetName, videoName })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    };

    return (
        <div>
            Dataset: <select value={selectedDir} ref={dropdownRef} onChange={handleDirectoryChange}>
                <option value="">Select a dataset</option>
                {directories.map(dir => (
                <option key={dir} value={dir}>{dir}</option>
                ))}
            </select>
            {currentVideo && (
                <div>
                    <video ref={videoRef} src={`/videos/${selectedDir}/${currentVideo}`} controls autoPlay /> <br/>
                    <button onClick={() => changeVideo(currentVideo, 'prev')}>Prev (b)</button>
                    <button onClick={() => changeVideo(currentVideo, 'next')}>Next (n)</button>
                    <button onClick={handlePlayPause}>Play/Pause (space)</button>
                    <button onClick={startSegment}>Start Segment (s)</button>
                    <button onClick={finishSegment}>Finish Segment (f)</button>
                    {segmentStart !== null ? <span style={{color: 'red', marginLeft: '10px'}}>Recording...</span> : null}
                </div>
            )}
            <AnnotationModal isOpen={isModalOpen} onClose={() => {setIsModalOpen(false); setSegmentStart(null);}} onSave={saveAnnotation} />
            <ul>
                {annotations.map((annot, index) => (
                    <li key={index}>{`Frame ${annot.start} - ${annot.end} (${frameToTimestamp(annot.start)} - ${frameToTimestamp(annot.end)}): ${annot.text} [${annot.failed ? 'Failed' : 'Success'}]`} <button onClick={() => handleDeleteAnnotation(index)}>Delete</button></li>
                ))}
            </ul>
        </div>
    );
}

export default VideoPlayer;
