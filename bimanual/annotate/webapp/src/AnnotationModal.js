import React, { useState, useEffect, useRef } from 'react';

function AnnotationModal({ isOpen, onClose, onSave }) {
    const [annotation, setAnnotation] = useState('');
    const [isFailure, setIsFailure] = useState(false);
    const textAreaRef = useRef(null);

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSave();
            }
            // Stop the propagation of the 'f' key when modal is open
            if (event.key === 'f') {
                event.stopPropagation();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            // Automatically focus the text area when the modal is opened
            if (textAreaRef.current) {
                textAreaRef.current.focus();
            }
        }
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [isOpen, annotation, isFailure]);

    const handleSave = () => {
        if (annotation.trim() !== "") {
            onSave(annotation, isFailure);
            onClose(); // Close modal and reset fields
            setAnnotation('');
            setIsFailure(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ background: 'white', padding: 20, borderRadius: 5 }}>
                <h2>Enter Annotation</h2>
                <textarea
                    ref={textAreaRef}
                    value={annotation}
                    onChange={e => setAnnotation(e.target.value)}
                    style={{ width: '300px', height: '100px' }}
                />
                <div>
                    <label>
                        <input
                            type="checkbox"
                            checked={isFailure}
                            onChange={e => setIsFailure(e.target.checked)}
                        />
                        Mark as failure
                    </label>
                </div>
                <button onClick={handleSave}>Save Annotation</button>
                <button onClick={onClose}>Cancel</button>
            </div>
        </div>
    );
}

export default AnnotationModal;

