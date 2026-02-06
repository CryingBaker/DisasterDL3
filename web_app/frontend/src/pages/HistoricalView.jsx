import { useState, useEffect } from 'react';
import axios from 'axios';
import MapViewer from '../components/MapViewer';
import ImageViewer from '../components/ImageViewer';
import { Search, ChevronRight, AlertCircle, X } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function HistoricalView() {
    const [samples, setSamples] = useState([]);
    const [selectedId, setSelectedId] = useState(null);
    const [metadata, setMetadata] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [loading, setLoading] = useState(true);
    const [showGroundTruth, setShowGroundTruth] = useState(true);
    const [showPrediction, setShowPrediction] = useState(true);

    useEffect(() => {
        fetchSamples();
    }, []);

    useEffect(() => {
        if (selectedId) {
            fetchMetadata(selectedId);
        }
    }, [selectedId]);

    const fetchSamples = async () => {
        try {
            const res = await axios.get(`${API_BASE}/samples`);
            setSamples(res.data.samples);
            setLoading(false);
        } catch (err) {
            console.error("Failed to fetch samples", err);
            setLoading(false);
        }
    };

    const fetchMetadata = async (id) => {
        try {
            const res = await axios.get(`${API_BASE}/samples/${id}/metadata`);
            setMetadata(res.data);
        } catch (err) {
            console.error("Failed to fetch metadata", err);
        }
    };

    const filteredSamples = samples.filter(s =>
        (s.id || s).toLowerCase().includes(searchTerm.toLowerCase())
    );

    const getSplitColor = (split) => {
        switch (split) {
            case 'train': return '#10b981';  // green
            case 'valid': return '#3b82f6';  // blue
            case 'test': return '#f59e0b';   // orange
            default: return '#6b7280';       // gray
        }
    };

    const getSplitLabel = (split) => {
        switch (split) {
            case 'train': return 'TR';
            case 'valid': return 'VL';
            case 'test': return 'TS';
            default: return '??';
        }
    };

    return (
        <div className="content-grid" style={{ height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
            {/* Sidebar */}
            <div className="glass-panel" style={{ margin: '1rem 0 1rem 1rem', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#fff' }}>
                <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <div style={{ position: 'relative' }}>
                        <Search className="text-secondary" size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)' }} />
                        <input
                            type="text"
                            placeholder="Search samples..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            style={{
                                width: '100%',
                                padding: '0.6rem 0.6rem 0.6rem 2.5rem',
                                background: '#f8fafc',
                                border: '1px solid var(--glass-border)',
                                borderRadius: '10px',
                                color: 'var(--text-primary)',
                                outline: 'none',
                                fontSize: '0.9rem'
                            }}
                        />
                    </div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem' }}>
                    {loading ? (
                        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="animate-spin" style={{ marginBottom: '1rem' }}>...</div>
                            Loading samples...
                        </div>
                    ) : (
                        filteredSamples.map(sample => {
                            const sampleId = sample.id || sample;
                            const split = sample.split || 'unknown';
                            return (
                                <div
                                    key={sampleId}
                                    onClick={() => setSelectedId(sampleId)}
                                    style={{
                                        padding: '0.8rem 1rem',
                                        borderRadius: '10px',
                                        cursor: 'pointer',
                                        background: selectedId === sampleId ? 'var(--bg-primary)' : 'transparent',
                                        color: selectedId === sampleId ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        marginBottom: '4px',
                                        transition: 'all 0.2s',
                                        border: selectedId === sampleId ? '1px solid var(--glass-border)' : '1px solid transparent'
                                    }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
                                        <span
                                            style={{
                                                fontSize: '0.6rem',
                                                fontWeight: 700,
                                                padding: '2px 5px',
                                                borderRadius: '4px',
                                                background: getSplitColor(split),
                                                color: 'white',
                                                flexShrink: 0
                                            }}
                                        >
                                            {getSplitLabel(split)}
                                        </span>
                                        <span style={{ fontSize: '0.85rem', fontWeight: selectedId === sampleId ? 600 : 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                            {sampleId}
                                        </span>
                                    </div>
                                    {selectedId === sampleId && <ChevronRight size={14} />}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Main Content Area */}
            <div style={{ padding: '1.5rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                {selectedId ? (
                    <>
                        <header>
                            <h2 style={{ margin: 0, fontSize: '1.75rem' }}>Event <span className="gradient-text">Visualization</span></h2>
                            <p style={{ color: 'var(--text-secondary)', margin: '0.25rem 0' }}>Deep learning analysis for flood extent detection.</p>
                        </header>

                        {/* Info Section */}
                        <div className="glass-panel" style={{ padding: '1.25rem', background: '#fff' }}>
                            <h3 style={{ marginTop: 0, fontSize: '1.1rem', color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <AlertCircle size={20} /> Application Guide
                            </h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', fontSize: '0.85rem', lineHeight: 1.5 }}>
                                <div>
                                    <strong style={{ display: 'block', marginBottom: '0.25rem', color: 'var(--text-primary)' }}>1. What is SAR?</strong>
                                    Synthetic Aperture Radar (SAR) can see through clouds and at night. It's the primary data source for our flood detection model.
                                </div>
                                <div>
                                    <strong style={{ display: 'block', marginBottom: '0.25rem', color: 'var(--text-primary)' }}>2. Pre vs Post Data</strong>
                                    We compare imagery from "Pre-Event" (normal) and "Post-Event" (flood) to identify where water has appeared.
                                </div>
                                <div>
                                    <strong style={{ display: 'block', marginBottom: '0.25rem', color: 'var(--text-primary)' }}>3. Understanding Overlay</strong>
                                    The blue overlay on the interactive map represents the Ground Truth (validated flood area) for historical samples.
                                </div>
                            </div>
                        </div>

                        {/* Metrics Banner */}
                        <div className="glass-panel" style={{ padding: '1.25rem', background: '#fff' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
                                <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Accuracy</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-primary)' }}>
                                        {metadata?.metrics?.accuracy ? (metadata.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}
                                    </div>
                                </div>
                                <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Mean IoU</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-secondary)' }}>
                                        {metadata?.metrics?.iou ? metadata.metrics.iou.toFixed(4) : 'N/A'}
                                    </div>
                                </div>
                                <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Status</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: metadata?.metrics?.status === 'Flooded' ? '#ef4444' : '#10b981' }}>
                                        {metadata?.metrics?.status || 'N/A'}
                                    </div>
                                </div>
                                <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Sample ID</div>
                                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{selectedId}</div>
                                </div>
                            </div>
                        </div>

                        {/* Overlay Toggle Controls */}
                        <div className="glass-panel" style={{ padding: '1rem', background: '#fff', display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
                            <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Overlay Controls:</span>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={showGroundTruth}
                                    onChange={(e) => setShowGroundTruth(e.target.checked)}
                                    style={{ width: '18px', height: '18px', accentColor: '#0064ff' }}
                                />
                                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <span style={{ width: '12px', height: '12px', background: 'rgba(0, 100, 255, 0.7)', borderRadius: '2px' }}></span>
                                    Ground Truth
                                </span>
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={showPrediction}
                                    onChange={(e) => setShowPrediction(e.target.checked)}
                                    style={{ width: '18px', height: '18px', accentColor: '#ff3c3c' }}
                                />
                                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <span style={{ width: '12px', height: '12px', background: 'rgba(255, 60, 60, 0.7)', borderRadius: '2px' }}></span>
                                    Model Prediction
                                </span>
                            </label>
                        </div>

                        {/* Map Viewer */}
                        <div style={{ height: '500px', flexShrink: 0 }}>
                            <MapViewer
                                bounds={metadata?.bounds}
                                groundTruthUrl={showGroundTruth ? `${API_BASE}/samples/${selectedId}/images/gt` : null}
                                predictionUrl={showPrediction ? `${API_BASE}/samples/${selectedId}/images/pred` : null}
                                opacity={0.6}
                            />
                        </div>

                        {/* Image Comparison */}
                        <div style={{ marginBottom: '2rem' }}>
                            <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Image Comparison Detail</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                                <ImageViewer title="Pre-Event SAR" src={`${API_BASE}/samples/${selectedId}/images/pre`} />
                                <ImageViewer title="Post-Event SAR" src={`${API_BASE}/samples/${selectedId}/images/post`} />
                                <ImageViewer title="Infrastructure" src={`${API_BASE}/samples/${selectedId}/images/infra`} />
                            </div>
                        </div>
                    </>
                ) : (
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', background: '#fff', maxWidth: '400px' }}>
                            <Search size={48} style={{ color: 'var(--accent-primary)', marginBottom: '1.5rem', opacity: 0.5 }} />
                            <h3 style={{ marginBottom: '0.5rem' }}>Select a Sample</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>Choose a historical event from the sidebar to visualize flood extent and model metrics.</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
