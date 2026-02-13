/**
 * FluxTrainer Pro — Training Dashboard
 * =====================================
 * 
 * Профессиональный dashboard для мониторинга и управления тренировкой
 * нейросетей прямо внутри ComfyUI.
 * 
 * Функции:
 *   - Real-time графики loss/lr/grad_norm
 *   - Мониторинг VRAM/RAM
 *   - Превью генераций во время тренировки
 *   - Визуальный конфигуратор параметров
 *   - Менеджер датасетов
 *   - Система пресетов
 * 
 * v2.5.0 | (c) 2024-2026 nkVas1
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ============================================================
// Section 1: CSS Injection
// ============================================================

const DASHBOARD_CSS = `
/* ============================================================
   FluxTrainer Pro Dashboard — Glassmorphism Dark Theme
   Гармонирует с ComfyUI dark theme
   ============================================================ */

/* === Floating Toggle Button === */
.ftpro-toggle-btn {
    position: fixed;
    bottom: 80px;
    right: 20px;
    z-index: 999;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    border: 2px solid rgba(99, 179, 237, 0.4);
    background: linear-gradient(135deg, rgba(26, 32, 44, 0.95), rgba(45, 55, 72, 0.95));
    color: #63b3ed;
    font-size: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), 0 0 30px rgba(99, 179, 237, 0.1);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    user-select: none;
}
.ftpro-toggle-btn:hover {
    transform: scale(1.1);
    border-color: rgba(99, 179, 237, 0.8);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 40px rgba(99, 179, 237, 0.3);
}
.ftpro-toggle-btn.training {
    animation: ftpro-pulse 2s infinite;
    border-color: rgba(72, 187, 120, 0.6);
    color: #48bb78;
}
@keyframes ftpro-pulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(0,0,0,0.4), 0 0 20px rgba(72,187,120,0.2); }
    50% { box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 40px rgba(72,187,120,0.4); }
}

/* === Mini Status Badge === */
.ftpro-mini-badge {
    position: absolute;
    top: -4px;
    right: -4px;
    min-width: 20px;
    height: 20px;
    border-radius: 10px;
    background: #48bb78;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 5px;
    display: none;
}
.ftpro-toggle-btn.training .ftpro-mini-badge {
    display: flex;
}

/* === Overlay Backdrop === */
.ftpro-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 9990;
    background: rgba(0, 0, 0, 0.5);
}
.ftpro-overlay.open {
    display: block;
}

/* === Main Dashboard Panel === */
.ftpro-dashboard {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 9991;
    width: 92vw;
    max-width: 1400px;
    height: 85vh;
    max-height: 900px;
    border-radius: 16px;
    border: 1px solid rgba(99, 179, 237, 0.2);
    background: linear-gradient(180deg, rgba(26, 32, 44, 0.98), rgba(17, 24, 39, 0.99));
    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.6), 0 0 60px rgba(99, 179, 237, 0.08);
    flex-direction: column;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e2e8f0;
}
.ftpro-dashboard.open {
    display: flex;
}

/* === Header === */
.ftpro-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(255, 255, 255, 0.02);
    flex-shrink: 0;
}
.ftpro-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.ftpro-logo {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #b794f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.ftpro-version {
    font-size: 11px;
    color: #718096;
    padding: 2px 8px;
    border: 1px solid rgba(113, 128, 150, 0.3);
    border-radius: 6px;
}
.ftpro-status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.ftpro-status-pill.idle { background: rgba(113, 128, 150, 0.2); color: #a0aec0; }
.ftpro-status-pill.training { background: rgba(72, 187, 120, 0.15); color: #48bb78; }
.ftpro-status-pill.completed { background: rgba(99, 179, 237, 0.15); color: #63b3ed; }
.ftpro-status-pill.error { background: rgba(245, 101, 101, 0.15); color: #fc8181; }
.ftpro-status-pill.preparing { background: rgba(237, 137, 54, 0.15); color: #ed8936; }
.ftpro-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}
.ftpro-status-pill.training .ftpro-status-dot {
    animation: ftpro-blink 1.5s infinite;
}
@keyframes ftpro-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.ftpro-close-btn {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: #a0aec0;
    font-size: 18px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}
.ftpro-close-btn:hover {
    background: rgba(245, 101, 101, 0.15);
    color: #fc8181;
    border-color: rgba(245, 101, 101, 0.3);
}

/* === Tab Bar === */
.ftpro-tabs {
    display: flex;
    gap: 4px;
    padding: 0 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
    background: rgba(255, 255, 255, 0.01);
}
.ftpro-tab {
    padding: 12px 20px;
    font-size: 13px;
    font-weight: 500;
    color: #718096;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}
.ftpro-tab:hover {
    color: #a0aec0;
    background: rgba(255, 255, 255, 0.02);
}
.ftpro-tab.active {
    color: #63b3ed;
    border-bottom-color: #63b3ed;
}
.ftpro-tab-icon {
    font-size: 15px;
}

/* === Content Area === */
.ftpro-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    scrollbar-width: thin;
    scrollbar-color: rgba(99, 179, 237, 0.2) transparent;
}
.ftpro-content::-webkit-scrollbar { width: 6px; }
.ftpro-content::-webkit-scrollbar-thumb { 
    background: rgba(99, 179, 237, 0.2); 
    border-radius: 3px; 
}

/* === Tab Panels === */
.ftpro-panel {
    display: none;
    animation: ftpro-fadeIn 0.3s ease;
}
.ftpro-panel.active {
    display: block;
}
@keyframes ftpro-fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* === Cards === */
.ftpro-cards-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}
.ftpro-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
}
.ftpro-card:hover {
    border-color: rgba(99, 179, 237, 0.2);
}
.ftpro-card-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #718096;
    margin-bottom: 8px;
}
.ftpro-card-value {
    font-size: 28px;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.ftpro-card-value.accent { color: #63b3ed; }
.ftpro-card-value.success { color: #48bb78; }
.ftpro-card-value.warning { color: #ed8936; }
.ftpro-card-sub {
    font-size: 12px;
    color: #718096;
    margin-top: 4px;
}

/* === Chart Container === */
.ftpro-chart-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.ftpro-chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.ftpro-chart-title {
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
}
.ftpro-chart-legend {
    display: flex;
    gap: 16px;
}
.ftpro-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #a0aec0;
}
.ftpro-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.ftpro-chart-canvas {
    width: 100%;
    height: 280px;
    border-radius: 8px;
}

/* === Progress Bar === */
.ftpro-progress-bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 20px;
}
.ftpro-progress-bar {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #63b3ed, #b794f4);
    transition: width 0.5s ease;
    min-width: 0%;
}

/* === Empty State === */
.ftpro-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: #718096;
    text-align: center;
}
.ftpro-empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
}
.ftpro-empty-title {
    font-size: 18px;
    font-weight: 600;
    color: #a0aec0;
    margin-bottom: 8px;
}
.ftpro-empty-desc {
    font-size: 13px;
    max-width: 400px;
    line-height: 1.6;
}

/* === Footer Bar === */
.ftpro-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 11px;
    color: #718096;
    flex-shrink: 0;
    background: rgba(255, 255, 255, 0.01);
}
.ftpro-footer-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.ftpro-footer a {
    color: #63b3ed;
    text-decoration: none;
}

/* === Responsive === */
@media (max-width: 900px) {
    .ftpro-dashboard {
        width: 98vw;
        height: 95vh;
        border-radius: 8px;
    }
    .ftpro-cards-row {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* === Sample Grid === */
.ftpro-sample-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
}
.ftpro-sample-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.2s;
    cursor: pointer;
}
.ftpro-sample-card:hover {
    border-color: rgba(99, 179, 237, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
.ftpro-sample-card img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
}
.ftpro-sample-meta {
    padding: 8px 10px;
    font-size: 11px;
    color: #a0aec0;
}

/* === Section Headers === */
.ftpro-section-title {
    font-size: 15px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
`;

function injectCSS() {
    if (document.getElementById('ftpro-dashboard-css')) return;
    const style = document.createElement('style');
    style.id = 'ftpro-dashboard-css';
    style.textContent = DASHBOARD_CSS;
    document.head.appendChild(style);
}


// ============================================================
// Section 2: API Client
// ============================================================

class FTProAPI {
    constructor() {
        this._listeners = {};
        this._lastStep = -1;
        this._wsSetup = false;
        this._shouldPoll = false;
        this._pollTimeout = null;
        this._pollIntervalMs = 3000;
        this._activeControllers = new Set();
    }

    async _request(endpoint, options = {}) {
        const controller = new AbortController();
        this._activeControllers.add(controller);
        try {
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            const resp = await window.fetch(`/api/fluxtrainer/${endpoint}`, {
                ...options,
                signal: controller.signal,
            });
            clearTimeout(timeoutId);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (e) {
            if (e.name !== 'AbortError') {
                console.debug(`[FTPro API] ${endpoint}:`, e.message);
            }
            return null;
        } finally {
            this._activeControllers.delete(controller);
        }
    }

    /** Cancel all in-flight requests */
    abortAll() {
        for (const c of this._activeControllers) {
            try { c.abort(); } catch (_) {}
        }
        this._activeControllers.clear();
    }

    async getStatus() { return this._request('status'); }
    async getLoss(since = 0) { return this._request(`loss?since=${since}`); }
    async getLR(since = 0) { return this._request(`lr?since=${since}`); }
    async getVRAM(last = 100) { return this._request(`vram?last=${last}`); }
    async getSamples(last = 20) { return this._request(`samples?last=${last}`); }
    async getConfig() { return this._request('config'); }
    async getPresets() { return this._request('presets/list'); }
    async savePreset(data) {
        return this._request('presets/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
    }
    async loadPreset(filename) {
        return this._request('presets/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
    }

    // Event system
    on(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
    }
    
    _emit(event, data) {
        (this._listeners[event] || []).forEach(cb => {
            try { cb(data); } catch (e) { console.error('[FTPro]', e); }
        });
    }

    // === Sequential Polling (never concurrent) ===
    startPolling(intervalMs = 3000) {
        this.stopPolling();
        this._pollIntervalMs = intervalMs;
        this._shouldPoll = true;
        this._setupWebSocket();
        // First poll after DOM settles
        this._pollTimeout = setTimeout(() => this._pollLoop(), 500);
    }

    stopPolling() {
        this._shouldPoll = false;
        if (this._pollTimeout) {
            clearTimeout(this._pollTimeout);
            this._pollTimeout = null;
        }
    }

    /** Sequential poll loop — next poll starts ONLY after previous completes */
    async _pollLoop() {
        if (!this._shouldPoll) return;
        try {
            const status = await this.getStatus();
            if (status && this._shouldPoll) {
                this._emit('status_update', status);
                if (status.step !== this._lastStep) {
                    this._lastStep = status.step;
                    this._emit('step_update', status);
                }
            }
        } catch (e) {
            console.debug('[FTPro] Poll error:', e.message);
        }
        // Schedule next poll AFTER this one finishes (never concurrent)
        if (this._shouldPoll) {
            this._pollTimeout = setTimeout(() => this._pollLoop(), this._pollIntervalMs);
        }
    }

    _setupWebSocket() {
        if (this._wsSetup) return;
        this._wsSetup = true;
        try {
            if (typeof api !== 'undefined' && api && typeof api.addEventListener === 'function') {
                api.addEventListener("fluxtrainer.progress", (d) => this._emit('progress', d.detail));
                api.addEventListener("fluxtrainer.status", (d) => this._emit('status_change', d.detail));
                api.addEventListener("fluxtrainer.sample", (d) => this._emit('sample', d.detail));
                api.addEventListener("fluxtrainer.started", (d) => this._emit('started', d.detail));
                api.addEventListener("fluxtrainer.finished", (d) => this._emit('finished', d.detail));
            }
        } catch (e) {
            console.debug('[FTPro] WebSocket setup skipped');
        }
    }
}


// ============================================================
// Section 3: Chart Renderer (Canvas-based, no dependencies)
// ============================================================

class FTProChart {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.options = {
            lineColor: '#63b3ed',
            lineWidth: 2,
            fillColor: 'rgba(99, 179, 237, 0.08)',
            gridColor: 'rgba(255, 255, 255, 0.04)',
            textColor: '#718096',
            axisColor: 'rgba(255, 255, 255, 0.08)',
            fontSize: 10,
            padding: { top: 10, right: 16, bottom: 30, left: 60 },
            showGrid: true,
            showFill: true,
            showDots: false,
            smoothing: 0.2,
            secondaryLine: null,
            secondaryColor: '#b794f4',
            ...options,
        };
        this._data = [];
        this._secondaryData = [];
        this._resize();
    }

    _resize() {
        try {
            const rect = this.canvas.parentElement?.getBoundingClientRect();
            if (!rect || rect.width < 10 || rect.height < 10) return false;
            const dpr = window.devicePixelRatio || 1;
            this.canvas.width = rect.width * dpr;
            this.canvas.height = (this.options.height || rect.height || 280) * dpr;
            this.canvas.style.width = rect.width + 'px';
            this.canvas.style.height = (this.options.height || rect.height || 280) + 'px';
            this.ctx.scale(dpr, dpr);
            this.w = rect.width;
            this.h = this.options.height || rect.height || 280;
            return true;
        } catch (e) {
            console.debug('[FTPro Chart] Resize error:', e.message);
            return false;
        }
    }

    setData(data, secondaryData = null) {
        this._data = Array.isArray(data) ? data : [];
        this._secondaryData = Array.isArray(secondaryData) ? secondaryData : [];
        try { this.render(); } catch (e) { console.debug('[FTPro Chart] Render error:', e.message); }
    }

    render() {
        if (!this._resize()) return;
        const { ctx, w, h, options: o } = this;
        if (!ctx || !w || !h) return;
        const p = o.padding;

        ctx.clearRect(0, 0, w, h);

        if (this._data.length < 2) {
            ctx.fillStyle = o.textColor;
            ctx.font = `13px -apple-system, sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillText('Ожидание данных...', w / 2, h / 2);
            return;
        }

        const plotW = w - p.left - p.right;
        const plotH = h - p.top - p.bottom;
        
        // Calculate ranges
        const values = this._data.map(d => {
            const v = d.value ?? d.y ?? 0;
            return Number.isFinite(v) ? v : 0;
        });
        if (values.length === 0) return;
        let minVal = Math.min(...values);
        let maxVal = Math.max(...values);
        if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return;
        if (minVal === maxVal) { minVal -= 0.001; maxVal += 0.001; }
        const valRange = maxVal - minVal;
        minVal -= valRange * 0.05;
        maxVal += valRange * 0.05;

        const minStep = this._data[0].step || this._data[0].x || 0;
        const maxStep = this._data[this._data.length - 1].step || this._data[this._data.length - 1].x || 0;

        const xScale = (v) => p.left + ((v - minStep) / (maxStep - minStep || 1)) * plotW;
        const yScale = (v) => p.top + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH;

        // Grid
        if (o.showGrid) {
            ctx.strokeStyle = o.gridColor;
            ctx.lineWidth = 1;
            const gridLines = 5;
            for (let i = 0; i <= gridLines; i++) {
                const y = p.top + (plotH / gridLines) * i;
                ctx.beginPath();
                ctx.moveTo(p.left, y);
                ctx.lineTo(w - p.right, y);
                ctx.stroke();

                // Y labels
                const val = maxVal - (i / gridLines) * (maxVal - minVal);
                ctx.fillStyle = o.textColor;
                ctx.font = `${o.fontSize}px -apple-system, sans-serif`;
                ctx.textAlign = 'right';
                ctx.fillText(val < 0.01 ? val.toExponential(2) : val.toFixed(4), p.left - 8, y + 3);
            }
        }

        // X labels
        ctx.fillStyle = o.textColor;
        ctx.textAlign = 'center';
        const xLabelCount = Math.min(6, this._data.length);
        for (let i = 0; i < xLabelCount; i++) {
            const idx = Math.floor((i / (xLabelCount - 1)) * (this._data.length - 1));
            const d = this._data[idx];
            const step = d.step || d.x || idx;
            const x = xScale(step);
            ctx.fillText(step.toString(), x, h - 8);
        }

        // Fill area
        if (o.showFill) {
            ctx.beginPath();
            ctx.moveTo(xScale(this._data[0].step || 0), yScale(minVal));
            this._data.forEach(d => {
                ctx.lineTo(xScale(d.step || d.x || 0), yScale(d.value || d.y || 0));
            });
            ctx.lineTo(xScale(this._data[this._data.length - 1].step || 0), yScale(minVal));
            ctx.closePath();
            ctx.fillStyle = o.fillColor;
            ctx.fill();
        }

        // Main line
        this._drawLine(this._data, o.lineColor, o.lineWidth);

        // Secondary line (e.g. moving average)
        if (this._secondaryData.length > 1) {
            this._drawLine(this._secondaryData, o.secondaryColor, 2);
        }
    }

    _drawLine(data, color, lineWidth) {
        if (!data || data.length < 2) return;
        const { ctx } = this;
        const p = this.options.padding;
        const plotW = this.w - p.left - p.right;
        const plotH = this.h - p.top - p.bottom;
        if (plotW <= 0 || plotH <= 0) return;

        const allValues = [...this._data, ...this._secondaryData].map(d => {
            const v = d.value ?? d.y ?? 0;
            return Number.isFinite(v) ? v : 0;
        });
        if (allValues.length === 0) return;
        let minVal = Math.min(...allValues);
        let maxVal = Math.max(...allValues);
        if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return;
        if (minVal === maxVal) { minVal -= 0.001; maxVal += 0.001; }
        const valRange = maxVal - minVal;
        minVal -= valRange * 0.05;
        maxVal += valRange * 0.05;

        const minStep = this._data[0]?.step || 0;
        const maxStep = this._data[this._data.length - 1]?.step || 1;

        const xScale = (v) => p.left + ((v - minStep) / (maxStep - minStep || 1)) * plotW;
        const yScale = (v) => p.top + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        data.forEach((d, i) => {
            const x = xScale(d.step || d.x || 0);
            const y = yScale(d.value || d.y || 0);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}


// ============================================================
// Section 4: Dashboard Component
// ============================================================

class FTProDashboard {
    constructor() {
        this.api = new FTProAPI();
        this.isOpen = false;
        this.currentTab = 'monitor';
        this.charts = {};
        this._state = {};
        this._elements = {};
        this._chartsInitialized = false;
        this._chartUpdating = false;
        this._chartUpdateTimer = null;
        
        // Tabs definition
        this.tabs = [
            { id: 'monitor',  icon: '📊', label: 'Мониторинг' },
            { id: 'config',   icon: '⚙️', label: 'Параметры' },
            { id: 'samples',  icon: '🖼️', label: 'Превью' },
            { id: 'dataset',  icon: '📁', label: 'Датасет' },
            { id: 'presets',  icon: '💾', label: 'Пресеты' },
        ];
    }

    init() {
        injectCSS();
        this._createToggleButton();
        this._createDashboard();
        this._setupEventHandlers();
        // WebSocket и polling запускаются ТОЛЬКО при открытии dashboard
    }

    // === Create Toggle Button ===
    _createToggleButton() {
        const btn = document.createElement('div');
        btn.className = 'ftpro-toggle-btn';
        btn.innerHTML = `
            <span>⚡</span>
            <div class="ftpro-mini-badge" id="ftpro-mini-progress">0%</div>
        `;
        btn.addEventListener('click', () => this.toggle());
        btn.title = 'FluxTrainer Pro Dashboard';
        document.body.appendChild(btn);
        this._elements.toggleBtn = btn;
    }

    // === Create Dashboard ===
    _createDashboard() {
        // Overlay
        const overlay = document.createElement('div');
        overlay.className = 'ftpro-overlay';
        overlay.addEventListener('click', () => this.close());
        document.body.appendChild(overlay);
        this._elements.overlay = overlay;

        // Dashboard
        const dash = document.createElement('div');
        dash.className = 'ftpro-dashboard';
        dash.innerHTML = this._renderHTML();
        dash.addEventListener('click', e => e.stopPropagation());
        document.body.appendChild(dash);
        this._elements.dashboard = dash;

        // Cache element references
        this._elements.statusPill = dash.querySelector('#ftpro-status-pill');
        this._elements.content = dash.querySelector('.ftpro-content');
        this._elements.progressBar = dash.querySelector('#ftpro-main-progress');
        this._elements.footer = dash.querySelector('.ftpro-footer-left');

        // Close button
        dash.querySelector('.ftpro-close-btn').addEventListener('click', () => this.close());

        // Tab clicks
        dash.querySelectorAll('.ftpro-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });

        // Charts инициализируются при первом открытии dashboard (когда canvases visible)
    }

    _renderHTML() {
        return `
            <!-- Header -->
            <div class="ftpro-header">
                <div class="ftpro-header-left">
                    <div class="ftpro-logo">FluxTrainer Pro</div>
                    <div class="ftpro-version">v2.5.0</div>
                    <div class="ftpro-status-pill idle" id="ftpro-status-pill">
                        <div class="ftpro-status-dot"></div>
                        <span>idle</span>
                    </div>
                </div>
                <button class="ftpro-close-btn">✕</button>
            </div>

            <!-- Tab Bar -->
            <div class="ftpro-tabs">
                ${this.tabs.map(t => `
                    <div class="ftpro-tab ${t.id === 'monitor' ? 'active' : ''}" data-tab="${t.id}">
                        <span class="ftpro-tab-icon">${t.icon}</span>
                        ${t.label}
                    </div>
                `).join('')}
            </div>

            <!-- Progress Bar -->
            <div class="ftpro-progress-bar-container">
                <div class="ftpro-progress-bar" id="ftpro-main-progress" style="width: 0%"></div>
            </div>

            <!-- Content -->
            <div class="ftpro-content">
                ${this._renderMonitorPanel()}
                ${this._renderConfigPanel()}
                ${this._renderSamplesPanel()}
                ${this._renderDatasetPanel()}
                ${this._renderPresetsPanel()}
            </div>

            <!-- Footer -->
            <div class="ftpro-footer">
                <div class="ftpro-footer-left">
                    <span id="ftpro-footer-status">Подключено к серверу</span>
                    <span>|</span>
                    <span id="ftpro-footer-time">—</span>
                </div>
                <div>
                    <a href="https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro" target="_blank">GitHub</a>
                </div>
            </div>
        `;
    }

    // === Tab Panels ===

    _renderMonitorPanel() {
        return `
        <div class="ftpro-panel active" id="ftpro-panel-monitor">
            <!-- Stat Cards -->
            <div class="ftpro-cards-row">
                <div class="ftpro-card">
                    <div class="ftpro-card-label">Шаг / Всего</div>
                    <div class="ftpro-card-value accent" id="ftpro-step-val">0 / 0</div>
                    <div class="ftpro-card-sub" id="ftpro-epoch-val">Эпоха 0 / 0</div>
                </div>
                <div class="ftpro-card">
                    <div class="ftpro-card-label">Текущий Loss</div>
                    <div class="ftpro-card-value" id="ftpro-loss-val">—</div>
                    <div class="ftpro-card-sub" id="ftpro-loss-sub">avg: — | min: —</div>
                </div>
                <div class="ftpro-card">
                    <div class="ftpro-card-label">Скорость</div>
                    <div class="ftpro-card-value success" id="ftpro-speed-val">—</div>
                    <div class="ftpro-card-sub" id="ftpro-speed-sub">шагов/сек</div>
                </div>
                <div class="ftpro-card">
                    <div class="ftpro-card-label">Время / ETA</div>
                    <div class="ftpro-card-value warning" id="ftpro-eta-val">—</div>
                    <div class="ftpro-card-sub" id="ftpro-time-sub">прошло: —</div>
                </div>
            </div>

            <!-- Loss Chart -->
            <div class="ftpro-chart-container">
                <div class="ftpro-chart-header">
                    <div class="ftpro-chart-title">Training Loss</div>
                    <div class="ftpro-chart-legend">
                        <div class="ftpro-legend-item">
                            <div class="ftpro-legend-dot" style="background:#63b3ed"></div>
                            Loss
                        </div>
                        <div class="ftpro-legend-item">
                            <div class="ftpro-legend-dot" style="background:#b794f4"></div>
                            Moving Avg
                        </div>
                    </div>
                </div>
                <canvas class="ftpro-chart-canvas" id="ftpro-chart-loss"></canvas>
            </div>

            <!-- LR Chart -->
            <div class="ftpro-chart-container">
                <div class="ftpro-chart-header">
                    <div class="ftpro-chart-title">Learning Rate Schedule</div>
                </div>
                <canvas class="ftpro-chart-canvas" id="ftpro-chart-lr" style="height:180px"></canvas>
            </div>

            <!-- VRAM chart -->
            <div class="ftpro-chart-container">
                <div class="ftpro-chart-header">
                    <div class="ftpro-chart-title">VRAM Usage</div>
                </div>
                <canvas class="ftpro-chart-canvas" id="ftpro-chart-vram" style="height:180px"></canvas>
            </div>
        </div>`;
    }

    _renderConfigPanel() {
        return `
        <div class="ftpro-panel" id="ftpro-panel-config">
            <div class="ftpro-section-title">⚙️ Конфигурация тренировки</div>
            <div class="ftpro-card" id="ftpro-config-display">
                <div class="ftpro-empty-state">
                    <div class="ftpro-empty-icon">⚙️</div>
                    <div class="ftpro-empty-title">Конфигурация будет доступна</div>
                    <div class="ftpro-empty-desc">
                        Запустите тренировку, чтобы увидеть все параметры.
                        Конфигурация автоматически загрузится из текущей ноды тренировки.
                    </div>
                </div>
            </div>
        </div>`;
    }

    _renderSamplesPanel() {
        return `
        <div class="ftpro-panel" id="ftpro-panel-samples">
            <div class="ftpro-section-title">🖼️ Превью генераций</div>
            <div id="ftpro-samples-grid" class="ftpro-sample-grid">
                <div class="ftpro-empty-state">
                    <div class="ftpro-empty-icon">🖼️</div>
                    <div class="ftpro-empty-title">Нет превью-изображений</div>
                    <div class="ftpro-empty-desc">
                        Сэмплы появятся во время тренировки, если настроен sample_every_n_steps 
                        в конфигурации. Каждое изображение будет отмечено шагом и промптом.
                    </div>
                </div>
            </div>
        </div>`;
    }

    _renderDatasetPanel() {
        return `
        <div class="ftpro-panel" id="ftpro-panel-dataset">
            <div class="ftpro-section-title">📁 Датасет</div>
            <div class="ftpro-card">
                <div class="ftpro-empty-state">
                    <div class="ftpro-empty-icon">📁</div>
                    <div class="ftpro-empty-title">Менеджер датасетов</div>
                    <div class="ftpro-empty-desc">
                        Просмотр, фильтрация и редактирование caption-файлов. 
                        Статистика по разрешениям, количеству изображений и баланс классов.
                        Будет доступно в следующем обновлении.
                    </div>
                </div>
            </div>
        </div>`;
    }

    _renderPresetsPanel() {
        return `
        <div class="ftpro-panel" id="ftpro-panel-presets">
            <div class="ftpro-section-title">💾 Пресеты тренировки</div>
            <div id="ftpro-presets-list" class="ftpro-card">
                <div class="ftpro-empty-state">
                    <div class="ftpro-empty-icon">💾</div>
                    <div class="ftpro-empty-title">Сохранённые пресеты</div>
                    <div class="ftpro-empty-desc">
                        Сохраняйте конфигурации тренировки как пресеты для быстрого переключения.
                        Включает recommended presets для типовых задач (LoRA face, style, concept).
                    </div>
                </div>
            </div>
        </div>`;
    }

    // === Charts Init (only once) ===
    _initCharts() {
        if (this._chartsInitialized) return;
        
        try {
            const lossCanvas = this._elements.dashboard.querySelector('#ftpro-chart-loss');
            const lrCanvas = this._elements.dashboard.querySelector('#ftpro-chart-lr');
            const vramCanvas = this._elements.dashboard.querySelector('#ftpro-chart-vram');

            if (lossCanvas) {
                this.charts.loss = new FTProChart(lossCanvas, {
                    lineColor: '#63b3ed',
                    secondaryColor: '#b794f4',
                    fillColor: 'rgba(99, 179, 237, 0.06)',
                });
            }
            if (lrCanvas) {
                this.charts.lr = new FTProChart(lrCanvas, {
                    lineColor: '#48bb78',
                    fillColor: 'rgba(72, 187, 120, 0.06)',
                    height: 180,
                });
            }
            if (vramCanvas) {
                this.charts.vram = new FTProChart(vramCanvas, {
                    lineColor: '#ed8936',
                    fillColor: 'rgba(237, 137, 54, 0.06)',
                    height: 180,
                });
            }
            
            this._chartsInitialized = true;
        } catch (e) {
            console.error('[FTPro] Chart init error:', e);
        }
    }

    // === Event Handlers ===
    _setupEventHandlers() {
        // Status update (polling) — lightweight UI update only
        this.api.on('status_update', (data) => {
            this._state = data;
            this._updateUI(data);
        });

        // Step changed — update charts (only when data actually changes)
        this.api.on('step_update', () => {
            if (this.isOpen && this.currentTab === 'monitor') {
                this._scheduleChartUpdate();
            }
        });

        // Real-time progress (WebSocket)
        this.api.on('progress', (data) => {
            this._updateProgress(data);
        });

        // Training finished
        this.api.on('finished', (data) => {
            this._showNotification(
                data.success ? '✅ Тренировка завершена!' : '❌ Ошибка тренировки',
                data.message || `Шагов: ${data.total_steps}`
            );
        });

        // Keyboard shortcut (Ctrl+Shift+T)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.toggle();
            }
        });
    }

    // === UI Updates ===
    _updateUI(data) {
        if (!data) return;

        try {
            // Toggle button state
            const isTraining = data.status === 'training';
            this._elements.toggleBtn.classList.toggle('training', isTraining);
            const badge = this._elements.toggleBtn.querySelector('.ftpro-mini-badge');
            if (badge) {
                badge.textContent = `${Math.round(data.progress_percent || 0)}%`;
            }

            // Status pill
            if (this._elements.statusPill) {
                const pill = this._elements.statusPill;
                pill.className = `ftpro-status-pill ${data.status}`;
                const span = pill.querySelector('span');
                if (span) span.textContent = this._translateStatus(data.status);
            }

            // Progress bar
            if (this._elements.progressBar) {
            this._elements.progressBar.style.width = `${data.progress_percent || 0}%`;
        }

        // If dashboard is open, update detailed info
        if (this.isOpen && this.currentTab === 'monitor') {
            this._updateMonitorTab(data);
        }

        // Footer
        const footerTime = this._elements.dashboard?.querySelector('#ftpro-footer-time');
        if (footerTime) {
            footerTime.textContent = `Обновлено: ${new Date().toLocaleTimeString()}`;
        }
        } catch (e) {
            console.debug('[FTPro] UI update error:', e.message);
        }
    }

    _updateMonitorTab(data) {
        const dash = this._elements.dashboard;
        if (!dash) return;

        try {
            // Step
            const stepEl = dash.querySelector('#ftpro-step-val');
            if (stepEl) stepEl.textContent = `${data.step || 0} / ${data.max_steps || 0}`;

            const epochEl = dash.querySelector('#ftpro-epoch-val');
            if (epochEl) epochEl.textContent = `Эпоха ${data.epoch || 0} / ${data.max_epochs || 0}`;

            // Loss
            const lossEl = dash.querySelector('#ftpro-loss-val');
            if (lossEl) {
                const loss = data.current_loss;
                lossEl.textContent = loss != null ? loss.toFixed(6) : '—';
            }
            const lossSub = dash.querySelector('#ftpro-loss-sub');
            if (lossSub) {
                lossSub.textContent = `avg: ${data.avg_loss?.toFixed(6) || '—'} | min: ${data.min_loss?.toFixed(6) || '—'} (шаг ${data.best_loss_step || '—'})`;
            }

            // Speed
            const speedEl = dash.querySelector('#ftpro-speed-val');
            if (speedEl) speedEl.textContent = data.steps_per_second ? `${data.steps_per_second.toFixed(2)}` : '—';

            // ETA
            const etaEl = dash.querySelector('#ftpro-eta-val');
            if (etaEl) etaEl.textContent = data.eta_seconds ? this._formatTime(data.eta_seconds) : '—';
            const timeSub = dash.querySelector('#ftpro-time-sub');
            if (timeSub) timeSub.textContent = `прошло: ${this._formatTime(data.elapsed_seconds || 0)}`;
        } catch (e) {
            console.debug('[FTPro] Monitor tab update error:', e.message);
        }
        // Charts update is triggered by step_update event, NOT here
    }

    _scheduleChartUpdate() {
        // Debounce: перезапускаем таймер при каждом вызове,
        // chart update начнётся через 500ms после ПОСЛЕДНЕГО вызова
        if (this._chartUpdateTimer) clearTimeout(this._chartUpdateTimer);
        this._chartUpdateTimer = setTimeout(() => {
            this._chartUpdateTimer = null;
            this._updateCharts(); // _updateCharts имеет собственную защиту от concurrent
        }, 500);
    }

    async _updateCharts() {
        if (!this.isOpen || this._chartUpdating) return;
        this._chartUpdating = true;
        
        try {
            // Loss chart
            if (this.charts.loss) {
                const lossData = await this.api.getLoss();
                if (lossData?.data?.length > 1) {
                    const movingAvg = this._calcMovingAvg(lossData.data, 20);
                    this.charts.loss.setData(lossData.data, movingAvg);
                }
            }

            // LR chart
            if (this.charts.lr) {
                const lrData = await this.api.getLR();
                if (lrData?.data?.length > 1) {
                    this.charts.lr.setData(lrData.data);
                }
            }

            // VRAM chart
            if (this.charts.vram) {
                const vramData = await this.api.getVRAM();
                if (vramData?.data?.length > 1) {
                    const mapped = vramData.data.map(d => ({ step: d.step, value: d.used }));
                    this.charts.vram.setData(mapped);
                }
            }
        } catch (e) {
            console.debug('[FTPro] Chart data fetch error:', e.message);
        } finally {
            this._chartUpdating = false;
        }
    }

    // === Helpers ===
    _calcMovingAvg(data, window = 20) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - window + 1);
            const subset = data.slice(start, i + 1);
            const avg = subset.reduce((s, d) => s + (d.value || 0), 0) / subset.length;
            result.push({ step: data[i].step, value: avg });
        }
        return result;
    }

    _translateStatus(status) {
        const map = {
            idle: 'Ожидание',
            preparing: 'Подготовка',
            training: 'Тренировка',
            sampling: 'Генерация',
            saving: 'Сохранение',
            paused: 'Пауза',
            completed: 'Завершено',
            error: 'Ошибка',
        };
        return map[status] || status;
    }

    _formatTime(seconds) {
        if (!seconds || seconds <= 0) return '—';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}ч ${m}м`;
        if (m > 0) return `${m}м ${s}с`;
        return `${s}с`;
    }

    _updateProgress(data) {
        // Quick update from WebSocket - minimal DOM touches
        if (this._elements.progressBar && data.max_steps > 0) {
            const pct = (data.step / data.max_steps) * 100;
            this._elements.progressBar.style.width = `${pct}%`;
        }
    }

    _showNotification(title, message) {
        // Simple notification via console (can be expanded)
        console.log(`[FTPro] ${title}: ${message}`);
    }

    // === Tab Management ===
    switchTab(tabId) {
        this.currentTab = tabId;
        const dash = this._elements.dashboard;
        
        dash.querySelectorAll('.ftpro-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabId);
        });
        dash.querySelectorAll('.ftpro-panel').forEach(p => {
            p.classList.toggle('active', p.id === `ftpro-panel-${tabId}`);
        });

        // Refresh charts when switching to monitor
        if (tabId === 'monitor') {
            requestAnimationFrame(() => {
                try {
                    this._initCharts(); // Только создаёт если ещё не созданы
                    this._scheduleChartUpdate();
                } catch (e) {
                    console.debug('[FTPro] Tab switch chart error:', e.message);
                }
            });
        }
        // Load config when switching to config tab
        if (tabId === 'config') this._loadConfigPanel();
        // Load samples when switching to samples tab  
        if (tabId === 'samples') this._loadSamplesPanel();
        // Load presets
        if (tabId === 'presets') this._loadPresetsPanel();
    }

    async _loadConfigPanel() {
        try {
            const container = this._elements.dashboard.querySelector('#ftpro-config-display');
            if (!container) return;
            
            const configData = await this.api.getConfig();
            if (!configData?.config || Object.keys(configData.config).length === 0) return;

        const config = configData.config;
        let html = `<div class="ftpro-section-title">⚙️ Текущие параметры: ${configData.model_name || 'Unknown'}</div>`;
        html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
        
        for (const [key, value] of Object.entries(config)) {
            html += `
                <div style="padding:8px 12px;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.04)">
                    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:0.5px">${key}</div>
                    <div style="font-size:13px;color:#e2e8f0;margin-top:4px;word-break:break-all">${String(value)}</div>
                </div>
            `;
        }
        html += '</div>';
        container.innerHTML = html;
        } catch (e) {
            console.debug('[FTPro] Config panel load error:', e.message);
        }
    }

    async _loadSamplesPanel() {
        try {
            const grid = this._elements.dashboard.querySelector('#ftpro-samples-grid');
            if (!grid) return;

        const samplesData = await this.api.getSamples();
        if (!samplesData?.data?.length) return;

        grid.innerHTML = samplesData.data.map(s => `
            <div class="ftpro-sample-card">
                <img src="/api/fluxtrainer/sample_image/${encodeURIComponent(s.path.split('/').pop().split('\\\\').pop())}" 
                     alt="Step ${s.step}" loading="lazy"
                     onerror="this.parentElement.style.display='none'">
                <div class="ftpro-sample-meta">
                    Step ${s.step} | Ep ${s.epoch}
                    ${s.prompt ? `<br>${s.prompt.substring(0, 60)}...` : ''}
                </div>
            </div>
        `).join('');
        } catch (e) {
            console.debug('[FTPro] Samples panel load error:', e.message);
        }
    }

    async _loadPresetsPanel() {
        try {
            const container = this._elements.dashboard.querySelector('#ftpro-presets-list');
            if (!container) return;

        const presetsData = await this.api.getPresets();
        if (!presetsData?.presets?.length) return;

        let html = '<div class="ftpro-section-title">💾 Сохранённые пресеты</div>';
        presetsData.presets.forEach(p => {
            html += `
                <div style="padding:12px;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.04);margin-bottom:8px;cursor:pointer"
                     onclick="this.style.borderColor='rgba(99,179,237,0.4)'">
                    <div style="font-size:14px;font-weight:600;color:#e2e8f0">${p.name}</div>
                    <div style="font-size:11px;color:#718096;margin-top:4px">${p.description || p.model_type} | ${p.created}</div>
                </div>
            `;
        });
        container.innerHTML = html;
        } catch (e) {
            console.debug('[FTPro] Presets panel load error:', e.message);
        }
    }

    // === Open/Close ===
    toggle() {
        this.isOpen ? this.close() : this.open();
    }

    open() {
        if (this.isOpen) return;
        this.isOpen = true;
        this._elements.overlay.classList.add('open');
        this._elements.dashboard.classList.add('open');
        
        // Give browser 300ms to paint the dashboard layout BEFORE any work
        this._openTimer = setTimeout(() => {
            if (!this.isOpen) return; // might have been closed already
            try {
                this._initCharts();
                if (this._state && Object.keys(this._state).length > 0) {
                    this._updateMonitorTab(this._state);
                }
            } catch (e) {
                console.debug('[FTPro] Open chart init error:', e.message);
            }
            // Start polling AFTER charts are initialized
            this.api.startPolling(3000);
        }, 300);
    }

    close() {
        if (!this.isOpen) return;
        this.isOpen = false;
        this._elements.overlay.classList.remove('open');
        this._elements.dashboard.classList.remove('open');
        
        // Cancel opening timer if still pending
        if (this._openTimer) {
            clearTimeout(this._openTimer);
            this._openTimer = null;
        }
        // Stop polling and abort all in-flight requests
        this.api.stopPolling();
        this.api.abortAll();
        // Clear chart timers
        if (this._chartUpdateTimer) {
            clearTimeout(this._chartUpdateTimer);
            this._chartUpdateTimer = null;
        }
        this._chartUpdating = false;
    }
}


// ============================================================
// Section 5: Extension Registration
// ============================================================

let dashboard = null;

app.registerExtension({
    name: "Comfy.FluxTrainerPro.Dashboard",
    
    async setup() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
        }
        
        // Small delay to let ComfyUI finish initializing
        setTimeout(() => {
            try {
                dashboard = new FTProDashboard();
                dashboard.init();
                console.log('[FluxTrainer Pro] Dashboard initialized (Ctrl+Shift+T to toggle)');
            } catch (e) {
                console.error('[FluxTrainer Pro] Dashboard init failed:', e);
            }
        }, 1500);
    }
});
