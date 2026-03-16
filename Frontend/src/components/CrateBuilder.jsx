import React, { useState, useCallback } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';
import CrateFlowDiagram from './CrateFlowDiagram';
import CrateDetail from './CrateDetail';
import CrateChat from './CrateChat';

export default function CrateBuilder() {
  const [crates, setCrates] = useState([]);
  const [selectedCrateId, setSelectedCrateId] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [branchParentId, setBranchParentId] = useState(null);
  const [error, setError] = useState('');

  const selectedCrate = crates.find(c => c.id === selectedCrateId) || null;
  const branchParent = crates.find(c => c.id === branchParentId) || null;

  const ensureSession = useCallback(async () => {
    if (activeSessionId) return activeSessionId;
    try {
      const res = await apiClient.post('/crates/session', { name: 'Live Session' });
      const id = res.id;
      setActiveSessionId(id);
      return id;
    } catch (e) {
      setError('Failed to create crate session');
      return null;
    }
  }, [activeSessionId]);

  const handleGenerate = useCallback(async (prompt) => {
    setError('');
    setIsGenerating(true);
    try {
      const sessionId = await ensureSession();
      if (!sessionId) return;

      const res = await apiClient.post('/crates/generate', {
        session_id: sessionId,
        prompt,
        parent_crate_id: branchParentId || undefined,
      });

      setCrates(prev => [...prev, res]);
      setSelectedCrateId(res.id);
      setBranchParentId(null);
    } catch (e) {
      setError(e.message || 'Failed to generate crate');
    } finally {
      setIsGenerating(false);
    }
  }, [ensureSession, branchParentId]);

  const handleAddBranch = useCallback((crateId) => {
    setBranchParentId(crateId);
    setSelectedCrateId(crateId);
  }, []);

  const handleRemoveTrack = useCallback(async (trackid) => {
    if (!selectedCrateId) return;
    setCrates(prev => prev.map(c => {
      if (c.id !== selectedCrateId) return c;
      return { ...c, tracks: (c.tracks || []).filter(t => t.trackid !== trackid) };
    }));
    try {
      await apiClient.post(`/crates/${selectedCrateId}/tracks`, {
        action: 'remove',
        trackid,
      });
    } catch (e) {
      // silently fail — local state already updated
    }
  }, [selectedCrateId]);

  const handleRequestBridge = useCallback(async () => {
    if (!selectedCrate?.parent_crate_id) return;
    const parentCrate = crates.find(c => c.id === selectedCrate.parent_crate_id);
    if (!parentCrate) return;

    try {
      const res = await apiClient.post('/crates/bridge', {
        crate_a_id: parentCrate.id,
        crate_b_id: selectedCrate.id,
      });
      // Update bridge warning metadata
      setCrates(prev => prev.map(c => {
        if (c.id !== selectedCrate.id) return c;
        return {
          ...c,
          metadata: { ...c.metadata, bridge_warning: res },
        };
      }));
    } catch (e) {
      setError('Failed to analyze bridge');
    }
  }, [selectedCrate, crates]);

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Left: Flow Diagram */}
      <GlassPanel animate={false} depth={1} style={{
        flex: 1, borderRadius: 0,
        borderRight: '1px solid rgba(124,58,237,0.08)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 16px',
          borderBottom: '1px solid rgba(124,58,237,0.06)',
          fontSize: 8, letterSpacing: '0.2em',
          color: 'rgba(124,58,237,0.4)',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          CRATE FLOW
        </div>
        <CrateFlowDiagram
          crates={crates}
          selectedCrateId={selectedCrateId}
          onSelectCrate={setSelectedCrateId}
          onAddBranch={handleAddBranch}
        />
      </GlassPanel>

      {/* Right: Detail + Chat */}
      <GlassPanel animate={false} depth={1} style={{
        width: 340, minWidth: 340, borderRadius: 0,
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ flex: 1, overflow: 'auto' }}>
          <CrateDetail
            crate={selectedCrate}
            onRemoveTrack={handleRemoveTrack}
            onRequestBridge={handleRequestBridge}
          />
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <m.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                padding: '6px 16px', fontSize: 9,
                color: 'rgba(239,68,68,0.8)',
                fontFamily: "'JetBrains Mono', monospace",
                background: 'rgba(239,68,68,0.05)',
                borderTop: '1px solid rgba(239,68,68,0.1)',
              }}
            >
              {error}
            </m.div>
          )}
        </AnimatePresence>

        <CrateChat
          onSubmit={handleGenerate}
          isLoading={isGenerating}
          selectedCrateLabel={branchParent?.label || ''}
        />
      </GlassPanel>
    </div>
  );
}
