import React, { useState } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';

const BloodCellAnalyzer = ({ onAnalysisComplete }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [results, setResults] = useState(null);
    const [annotatedImage, setAnnotatedImage] = useState(null);

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResults(null);
            setAnnotatedImage(null);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setAnalyzing(true);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('http://localhost:8000/api/image-analysis/analyze-blood-smear', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                setResults(data.counts);
                setAnnotatedImage(data.annotated_image);

                // Callback to parent with cell counts
                if (onAnalysisComplete) {
                    onAnalysisComplete(data.counts);
                }
            }
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Failed to analyze image. Please try again.');
        } finally {
            setAnalyzing(false);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResults(null);
        setAnnotatedImage(null);
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Blood Cell Image Analysis</h3>

            {/* Upload Section */}
            {!selectedFile && (
                <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer">
                    <label htmlFor="file-upload" className="cursor-pointer">
                        <Upload className="w-12 h-12 mx-auto text-slate-400 mb-3" />
                        <p className="text-sm text-slate-600 mb-1">Click to upload blood smear image</p>
                        <p className="text-xs text-slate-400">PNG, JPG up to 10MB</p>
                        <input
                            id="file-upload"
                            type="file"
                            accept="image/*"
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                    </label>
                </div>
            )}

            {/* Preview & Results */}
            {selectedFile && (
                <div className="space-y-4">
                    <div className="flex justify-between items-center mb-2">
                        <p className="text-sm text-slate-600">{selectedFile.name}</p>
                        <button onClick={handleReset} className="text-slate-400 hover:text-slate-600">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Image Display */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {previewUrl && (
                            <div>
                                <p className="text-sm font-medium text-slate-700 mb-2">Original Image</p>
                                <img src={previewUrl} alt="Preview" className="w-full rounded-lg border border-slate-200" />
                            </div>
                        )}

                        {annotatedImage && (
                            <div>
                                <p className="text-sm font-medium text-slate-700 mb-2">Detected Cells</p>
                                <img src={annotatedImage} alt="Annotated" className="w-full rounded-lg border border-slate-200" />
                            </div>
                        )}
                    </div>

                    {/* Analyze Button */}
                    {!results && (
                        <button
                            onClick={handleAnalyze}
                            disabled={analyzing}
                            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {analyzing ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                'Analyze Blood Smear'
                            )}
                        </button>
                    )}

                    {/* Results Display */}
                    {results && (
                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                            <h4 className="font-semibold text-slate-900 mb-3">Cell Count Results</h4>
                            <div className="grid grid-cols-3 gap-4">
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-blue-600">{results.WBC}</div>
                                    <div className="text-xs text-slate-600 mt-1">White Blood Cells</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-red-600">{results.RBC}</div>
                                    <div className="text-xs text-slate-600 mt-1">Red Blood Cells</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-green-600">{results.Platelet}</div>
                                    <div className="text-xs text-slate-600 mt-1">Platelets</div>
                                </div>
                            </div>
                            <button
                                onClick={handleReset}
                                className="w-full mt-4 bg-slate-100 text-slate-700 py-2 rounded-lg font-medium hover:bg-slate-200"
                            >
                                Upload New Image
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default BloodCellAnalyzer;
