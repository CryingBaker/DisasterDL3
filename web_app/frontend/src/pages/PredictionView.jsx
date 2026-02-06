import { useState } from 'react';
import axios from 'axios';
import { Upload, AlertCircle, CheckCircle2, Loader2, Activity } from 'lucide-react';
import ImageViewer from '../components/ImageViewer';

const API_BASE = 'http://localhost:8000';

export default function PredictionView() {
    const [postFile, setPostFile] = useState(null);
    const [preFile, setPreFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        if (!postFile) {
            setError("Post-event image is required.");
            return;
        }

        setLoading(true);
        setError(null);
        setPrediction(null);

        const formData = new FormData();
        formData.append('post_image', postFile);
        if (preFile) {
            formData.append('pre_image', preFile);
        }

        try {
            const res = await axios.post(`${API_BASE}/predict`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setPrediction(res.data);
        } catch (err) {
            console.error("Prediction failed", err);
            setError(err.response?.data?.detail || "An error occurred during prediction.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-container" style={{ height: 'calc(100vh - 64px)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <header>
                <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Live <span className="gradient-text">Inference</span></h2>
                <p style={{ color: 'var(--text-secondary)' }}>Upload Sentinel-1 SAR TIF files (VV/VH channels) to predict flood extent.</p>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 400px) 1fr', gap: '2rem' }}>
                {/* Upload Config */}
                <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Post-Event SAR (Required)</label>
                        <div
                            style={{
                                border: '2px dashed var(--glass-border)',
                                borderRadius: '12px',
                                padding: '2rem',
                                textAlign: 'center',
                                cursor: 'pointer',
                                background: postFile ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                                transition: 'border-color 0.2s'
                            }}
                            onClick={() => document.getElementById('post-upload').click()}
                        >
                            <Upload className="text-secondary" style={{ margin: '0 auto 0.5rem' }} />
                            <div style={{ fontSize: '0.9rem', color: postFile ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                                {postFile ? postFile.name : "Click to upload Post-event TIF"}
                            </div>
                            <input
                                id="post-upload"
                                type="file"
                                hidden
                                accept=".tif,.tiff"
                                onChange={(e) => setPostFile(e.target.files[0])}
                            />
                        </div>
                    </div>

                    <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Pre-Event SAR (Optional)</label>
                        <div
                            style={{
                                border: '2px dashed var(--glass-border)',
                                borderRadius: '12px',
                                padding: '1.5rem',
                                textAlign: 'center',
                                cursor: 'pointer',
                                background: preFile ? 'rgba(139, 92, 246, 0.1)' : 'transparent'
                            }}
                            onClick={() => document.getElementById('pre-upload').click()}
                        >
                            <Upload className="text-secondary" style={{ margin: '0 auto 0.5rem' }} size={20} />
                            <div style={{ fontSize: '0.8rem', color: preFile ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                                {preFile ? preFile.name : "Click to upload Pre-event TIF"}
                            </div>
                            <input
                                id="pre-upload"
                                type="file"
                                hidden
                                accept=".tif,.tiff"
                                onChange={(e) => setPreFile(e.target.files[0])}
                            />
                        </div>
                    </div>

                    {error && (
                        <div style={{ color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', padding: '0.75rem', borderRadius: '8px', display: 'flex', gap: '0.5rem', fontSize: '0.9rem' }}>
                            <AlertCircle size={18} />
                            {error}
                        </div>
                    )}

                    <button
                        className="btn-primary"
                        onClick={handlePredict}
                        disabled={loading || !postFile}
                        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: 'auto' }}
                    >
                        {loading ? <Loader2 className="animate-spin" size={20} /> : <Activity size={20} />}
                        {loading ? "Processing..." : "Run Inference"}
                    </button>
                </div>

                {/* Results Area */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    {prediction ? (
                        <div className="glass-panel" style={{ padding: '1.5rem', animation: 'fadeIn 0.5s ease-out' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', color: prediction.is_flooded ? '#ef4444' : '#10b981' }}>
                                <CheckCircle2 size={24} />
                                <h3 style={{ margin: 0 }}>Prediction Complete: {prediction.is_flooded ? "Flood Detected" : "No Major Flooding"}</h3>
                            </div>

                            {/* Stats Row */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                                <div className="metric-card glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>FLOOD COVERAGE</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: prediction.flood_percentage > 10 ? '#ef4444' : '#10b981' }}>
                                        {prediction.flood_percentage || 0}%
                                    </div>
                                </div>
                                <div className="metric-card glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>IMAGE SIZE</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                                        {prediction.image_size ? `${prediction.image_size[0]} × ${prediction.image_size[1]}` : 'N/A'}
                                    </div>
                                </div>
                                <div className="metric-card glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>TILES PROCESSED</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                                        {prediction.tiles_processed || 1}
                                    </div>
                                </div>
                                <div className="metric-card glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>PRE-EVENT</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600, color: preFile ? '#3b82f6' : '#666' }}>
                                        {preFile ? 'Used' : 'None'}
                                    </div>
                                </div>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    <h4 style={{ margin: 0, color: 'var(--text-secondary)' }}>Input Files</h4>
                                    <div className="glass-panel" style={{ padding: '1rem' }}>
                                        <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                            <strong>Post-Event:</strong> {postFile?.name}
                                        </div>
                                        {preFile && (
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                                <strong>Pre-Event:</strong> {preFile.name}
                                            </div>
                                        )}
                                        {!preFile && (
                                            <div style={{ fontSize: '0.85rem', color: '#f59e0b' }}>
                                                ⚠️ No pre-event image provided (reduced accuracy)
                                            </div>
                                        )}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    <h4 style={{ margin: 0, color: 'var(--text-secondary)' }}>Generated Flood Mask</h4>
                                    <ImageViewer src={prediction.prediction} title="Model Output" />
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', borderStyle: 'dashed', minHeight: '300px' }}>
                            <div style={{ textAlign: 'center' }}>
                                <Activity size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
                                <p style={{ margin: 0 }}>Upload SAR TIF files and click "Run Inference"</p>
                                <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.7 }}>
                                    Supports images larger than 512×512 via automatic tiling
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    );
}
