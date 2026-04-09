import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';  // Assuming your main component is App

function Main() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<App />} />
                <Route path="/:selectedDirectory" element={<App />} />
                <Route path="/:selectedDirectory/:selectedVideo" element={<App />} />
            </Routes>
        </Router>
    );
}

export default Main;
