/**
 * ================================================================
 * RAG CONTROL CONSOLE — HUD Application Logic & Orchestrator
 * ================================================================
 */

const API_BASE = `${window.location.origin}`;

// ---- DOM References ----
const form                  = document.getElementById('comm-form');
const input                 = document.getElementById('comm-input');
const chatWindow            = document.getElementById('chat-window');
const logWindow             = document.getElementById('log-window');
const overflowLogWindow     = document.getElementById('overflow-log-window');
const inspectorWindow       = document.getElementById('inspector-window');
const reconWindow           = document.getElementById('recon-window');
const btn                   = document.getElementById('comm-btn');
const stopBtn               = document.getElementById('stop-btn');
const sessionTag            = document.getElementById('session-tag');
const knowledgeCount        = document.getElementById('knowledge-count');
const activeModeBadge       = document.getElementById('active-mode-badge');

// Stats Panel
const statQ    = document.getElementById('stat-q');
const statC    = document.getElementById('stat-c');
const statT    = document.getElementById('stat-t');
const statM    = document.getElementById('stat-m');
const statCpu  = document.getElementById('stat-cpu');
const statRam  = document.getElementById('stat-ram');
const statCtx  = document.getElementById('stat-ctx');
const statTps  = document.getElementById('stat-tps');

// Budget Controls
const contextLimitSlider      = document.getElementById('context-limit-slider');
const contextLimitSliderVal   = document.getElementById('context-limit-slider-val');
const tokenUsedVal            = document.getElementById('token-used-val');
const tokenLimitValReadout    = document.getElementById('token-limit-val-readout');
const tokenProgressFill       = document.getElementById('token-progress-fill');
const tokenPercentageText     = document.getElementById('token-percentage-text');
const overflowAlertBanner     = document.getElementById('overflow-alert-banner');
const overflowIndicatorDot     = document.getElementById('overflow-indicator-dot');

// Modal Elements
const telemetryModal          = document.getElementById('telemetry-modal');
const telemetryModalBody      = document.getElementById('telemetry-modal-body');

// ---- Session State Variables ----
let sid = localStorage.getItem('station_sid') || 'SID-' + Math.random().toString(36).substr(2, 6).toUpperCase();
localStorage.setItem('station_sid', sid);
sessionTag.textContent = `SID: ${sid}`;

let contextLimit = parseInt(contextLimitSlider.value);
let abortController = null;

// ---- Explicit App State Model ----
const AppState = {
    get sid() { return sid; },
    get contextLimit() { return contextLimit; },
    get isGenerating() { return abortController !== null; },
    get abortController() { return abortController; },
    
    updateContextLimit(val) {
        contextLimit = val;
        contextLimitSliderVal.textContent = val;
        tokenLimitValReadout.textContent = val;
        const currentUsed = parseInt(tokenUsedVal.textContent) || 0;
        updateProgressBar(currentUsed, val, currentUsed > val);
    },
    
    setGenerating(generating, controller = null) {
        abortController = controller;
        btn.disabled = generating;
        if (stopBtn) stopBtn.style.display = generating ? 'inline-block' : 'none';
        addLog(generating ? "System transitioned to state: GENERATING" : "System transitioned to state: IDLE", "STATE");
    }
};

// ---- Slider & Preset Listeners ----
contextLimitSlider.addEventListener('input', (e) => {
    AppState.updateContextLimit(parseInt(e.target.value));
});

document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Toggle active style
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const val = parseInt(btn.dataset.val);
        contextLimitSlider.value = val;
        AppState.updateContextLimit(val);
        addLog(`Context window boundary adjusted to ${val} tokens via preset.`, 'SYSTEM');
    });
});

// ---- Progress Bar Sizer ----
function updateProgressBar(used, limit, isBreached) {
    const percent = Math.min(100, Math.round((used / limit) * 100));
    tokenProgressFill.style.width = `${percent}%`;
    tokenPercentageText.textContent = `${percent}% of window occupied`;
    
    // Clear status classes
    tokenProgressFill.classList.remove('warn', 'breached');
    
    if (isBreached) {
        tokenProgressFill.classList.add('breached');
    } else if (percent > 85) {
        tokenProgressFill.classList.add('warn');
    }
}

// ---- System Log Helpers ----
function addLog(msg, type = 'INFO') {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const entry = document.createElement('div');
    entry.className = `log-entry log-type-${type}`;
    entry.innerHTML = `<span class="log-time">[${time}]</span><span class="log-type">${type}</span> ${msg}`;
    logWindow.appendChild(entry);
    logWindow.scrollTop = logWindow.scrollHeight;
}

// ---- Chat Bubble Renderer ----
function addMsg(text, type = 'ai', telemetryData = null) {
    const msg = document.createElement('div');
    msg.className = `message msg-${type}`;
    
    let formattedText = text;
    if (type === 'ai') {
        try { 
            formattedText = typeof marked !== 'undefined' ? marked.parse(text) : text; 
        } catch (e) { 
            formattedText = text; 
        }
    }
    
    // Build main content
    let contentHtml = `<div class="msg-header">${type === 'user' ? 'USER_INPUT' : 'SYSTEM_OUTPUT'}</div><div class="msg-body">${formattedText}</div>`;
    
    // If AI Turn has telemetry, append footnote badge
    if (type === 'ai' && telemetryData) {
        // Parse if string (from SQLite history)
        if (typeof telemetryData === 'string') {
            try { telemetryData = JSON.parse(telemetryData); } catch (e) { telemetryData = null; }
        }
        
        if (telemetryData) {
            const hasOverflow = telemetryData.overflow_occurred === true;
            const initTkn = telemetryData.initial_tokens || 'N/A';
            const finalTkn = telemetryData.final_tokens || 'N/A';
            const limitVal = telemetryData.limit || 'N/A';
            
            if (hasOverflow) {
                msg.classList.add('msg-overflow-recovered');
            }
            
            contentHtml += `
                <div class="msg-telemetry">
                    <div class="telemetry-badges">
                        <span class="badge-item ${hasOverflow ? 'recovered' : 'nominal'}">
                            ${hasOverflow ? 'RECOVERED' : 'NOMINAL'}
                        </span>
                        <span class="badge-item">LIMIT: ${limitVal} TKN</span>
                        <span class="badge-item">FOOTPRINT: ${finalTkn} TKN</span>
                    </div>
                    <button class="telemetry-inspect-btn" onclick="viewTelemetryDetails(${escapeHtml(JSON.stringify(telemetryData))})">
                        INSPECT
                    </button>
                </div>
            `;
        }
    }
    
    msg.innerHTML = contentHtml;
    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return msg;
}

// HTML Escaping Helper for Telemetry JSON Inject
function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ---- Inspector Panels Renderer ----
function renderInspector(data) {
    const tokens  = data.stats?.exact_tokens || { prompt: 0, completion: 0, total: 0 };
    const budget  = data.stats?.budget_tracking || {};
    const latency = data.stats?.instantaneous_latency_ms || {};

    inspectorWindow.innerHTML = `
        <div class="inspector-section">
            <div class="inspector-section-hdr">EXACT TOKEN FOOTPRINT</div>
            <div class="inspector-section-body">
PROMPT: ${tokens.prompt} TKN
COMPLETION: ${tokens.completion} TKN
TOTAL: ${tokens.total} TKN

GENERATION SPEED: ${data.tps || 0} t/s
ESTIMATED COST: ${data.query_cost || '$0.00'}
            </div>
        </div>
        <div class="inspector-section">
            <div class="inspector-section-hdr">BUDGET ALLOCATION</div>
            <div class="inspector-section-body">
MEMORY OCCUPIED: ${budget.memory_tokens_used || 0} / ${budget.memory_tokens_limit || 0} TKN
KNOWLEDGE COMPRESSED: ${budget.document_tokens_used || 0} / ${budget.document_tokens_limit || 0} TKN
            </div>
        </div>
        <div class="inspector-section">
            <div class="inspector-section-hdr">PIPELINE TELEMETRY</div>
            <div class="inspector-section-body">
MODE: ${(data.stats?.mode || 'N/A').toUpperCase()}
HYBRID ALPHA: ${data.stats?.alpha || 0.5}
PEAK RE-RANK SCORE: ${data.stats?.reranker_peak_score || 0}
COMPRESSION RATIO: ${((data.stats?.compression_ratio || 0) * 100).toFixed(1)}%
EMBED GENERATION: ${latency.phase_2_embed_generation_ms || 0} ms
WEAVIATE SEARCH: ${latency.phase_2_weaviate_search_ms || 0} ms
HYDE GENERATION: ${latency.phase_1_5_hyde_ms || 0} ms
            </div>
        </div>
        <div class="inspector-section">
            <div class="inspector-section-hdr">LLM QUERY EXPANSIONS</div>
            <div class="inspector-section-body">${data.search_queries ? data.search_queries.map(q => `> ${q}`).join('\n') : 'N/A'}</div>
        </div>
        <div class="inspector-section">
            <div class="inspector-section-hdr">HYPOTHETICAL DOCUMENT (HyDE)</div>
            <div class="inspector-section-body">${data.hyde_doc || 'N/A'}</div>
        </div>
        <div class="inspector-section">
            <div class="inspector-section-hdr">RAW PROMPT INSPECTION</div>
            <div class="inspector-section-body" style="font-family: var(--font-mono); font-size: 0.65rem; max-height: 160px; overflow-y: auto; background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.03); white-space: pre-wrap; word-break: break-all;">${escapeHtml(data.raw_prompt || 'N/A')}</div>
        </div>
    `;
}

// ---- SSE Streaming Controller ----
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    addMsg(query, 'user');
    
    const mode = document.querySelector('input[name="engine-mode"]:checked').value;
    activeModeBadge.textContent = mode.replace('_', ' ').toUpperCase();
    
    addLog(`Initiating streaming request (Mode: ${mode} | Limit: ${contextLimit} TKN)`, 'REQUEST');

    // Reset/Clear UI state for query run
    reconWindow.innerHTML = '';
    inspectorWindow.innerHTML = '<div class="inspector-placeholder">Awaiting telemetry stream...</div>';
    
    // Clear and reset context overflow debugger
    overflowLogWindow.innerHTML = '';
    overflowIndicatorDot.className = 'indicator-dot nominal';
    overflowAlertBanner.className = 'overflow-banner alert-nominal';
    overflowAlertBanner.textContent = 'SYSTEM RUNNING IN NOMINAL STATE';

    const controller = new AbortController();
    AppState.setGenerating(true, controller);
    const startTime = Date.now();

    // Create placeholder AI message bubble
    const aiBubble = document.createElement('div');
    aiBubble.className = 'message msg-ai';
    aiBubble.innerHTML = `<div class="msg-header">SYSTEM_OUTPUT</div><div class="msg-body typing-cursor"></div>`;
    chatWindow.appendChild(aiBubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    const bodyContainer = aiBubble.querySelector('.msg-body');
    let accumulatedText = "";

    try {
        const response = await fetch(`${API_BASE}/query_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question: query, 
                session_id: AppState.sid, 
                mode: mode,
                context_limit: AppState.contextLimit
            }),
            signal: AppState.abortController.signal
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server Error");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Hold remaining partial line

            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith("data:")) continue;

                const jsonStr = cleanLine.substring(5).trim();
                if (!jsonStr) continue;

                let data;
                try {
                    data = JSON.parse(jsonStr);
                } catch (parseErr) {
                    console.error("SSE parse error", jsonStr, parseErr);
                    continue;
                }

                if (data.event === "thought") {
                    addLog(data.text, "THOUGHT");
                } 
                else if (data.event === "action") {
                    addLog(`Executing tool: ${data.tool}[${data.input}]`, "ACTION");
                }
                else if (data.event === "observation") {
                    addLog(`Received tool observation (${data.output.length} chars)`, "OBSERVATION");
                }
                else if (data.event === "overflow_detected") {
                    // Trigger visual warning alerts
                    overflowIndicatorDot.className = 'indicator-dot breached';
                    overflowAlertBanner.className = 'overflow-banner alert-breached';
                    overflowAlertBanner.innerHTML = `🚨 OVERFLOW DETECTED: Prompt (${data.initial} TKN) exceeds limit (${data.limit} TKN)`;
                    
                    tokenUsedVal.textContent = data.initial;
                    updateProgressBar(data.initial, data.limit, true);
                    
                    addLog(`Context overflow detected! Size: ${data.initial} TKN. Limit: ${data.limit} TKN. Running recovery...`, "WARNING");
                }
                else if (data.event === "overflow_step") {
                    // Stream lines to overflow debugger terminal shell
                    const stepDiv = document.createElement('div');
                    stepDiv.className = 'overflow-step-line';
                    
                    // Style lines based on contents
                    if (stepDiv.textContent = data.text) {
                        if (data.text.includes("🚨")) {
                            stepDiv.className += ' step-alarm';
                        } else if (data.text.includes("Phase 1")) {
                            stepDiv.className += ' step-phase1';
                        } else if (data.text.includes("Phase 2")) {
                            stepDiv.className += ' step-phase2';
                        } else if (data.text.includes("Phase 3")) {
                            stepDiv.className += ' step-phase3';
                        } else if (data.text.includes("✅")) {
                            stepDiv.className += ' step-success';
                        }
                    }
                    
                    overflowLogWindow.appendChild(stepDiv);
                    overflowLogWindow.scrollTop = overflowLogWindow.scrollHeight;
                }
                else if (data.event === "answer_chunk") {
                    accumulatedText += data.text;
                    try {
                        bodyContainer.innerHTML = typeof marked !== 'undefined' ? marked.parse(accumulatedText) : accumulatedText;
                    } catch (e) {
                        bodyContainer.textContent = accumulatedText;
                    }
                    chatWindow.scrollTop = chatWindow.scrollHeight;
                }
                else if (data.event === "error") {
                    throw new Error(data.message);
                }
                else if (data.event === "done") {
                    const latency = Date.now() - startTime;
                    addLog(`Processing complete: ${latency}ms`, 'SUCCESS');
                    bodyContainer.classList.remove('typing-cursor');

                    const stats = data.stats || {};
                    const telemetry = stats.overflow_telemetry || {};
                    const budget = stats.budget_tracking || {};

                    // Sync budget progress bar to final compiled values
                    const finalUsed = telemetry.final_tokens || (budget.memory_tokens_used + budget.document_tokens_used + 100);
                    tokenUsedVal.textContent = finalUsed;
                    updateProgressBar(finalUsed, contextLimit, finalUsed > contextLimit);

                    // Add telemetry elements to the current chat bubble
                    if (telemetry && telemetry.limit) {
                        const hasOverflow = telemetry.overflow_occurred === true;
                        if (hasOverflow) {
                            aiBubble.classList.add('msg-overflow-recovered');
                        }
                        
                        const footerDiv = document.createElement('div');
                        footerDiv.className = 'msg-telemetry';
                        footerDiv.innerHTML = `
                            <div class="telemetry-badges">
                                <span class="badge-item ${hasOverflow ? 'recovered' : 'nominal'}">
                                    ${hasOverflow ? 'RECOVERED' : 'NOMINAL'}
                                </span>
                                <span class="badge-item">LIMIT: ${telemetry.limit} TKN</span>
                                <span class="badge-item">FOOTPRINT: ${telemetry.final_tokens} TKN</span>
                            </div>
                            <button class="telemetry-inspect-btn" onclick='viewTelemetryDetails(${JSON.stringify(telemetry)}, ${JSON.stringify(budget)}, "${escapeHtml(query)}")'>
                                INSPECT
                            </button>
                        `;
                        aiBubble.appendChild(footerDiv);
                    }

                    // Render retrieved sources list in sidebar
                    if (stats.retrieved_context) {
                        reconWindow.innerHTML = '';
                        stats.retrieved_context.forEach(hit => {
                            const item = document.createElement('div');
                            item.className = 'readout-item';
                            item.innerHTML = `
                                ${escapeHtml(hit.text.substring(0, 180))}...
                                <div class="readout-meta">
                                    <span>SCORE: ${hit.score.toFixed(4)}</span>
                                    <span>SRC: ${escapeHtml(hit.source)}</span>
                                </div>
                            `;
                            reconWindow.appendChild(item);
                        });
                    } else {
                        reconWindow.innerHTML = '<div class="readout-empty">No active retrieval context.</div>';
                    }

                    // Map statistics to top performance metrics
                    statQ.textContent = stats.queries_handled || statQ.textContent;
                    if (stats.compression_ratio !== undefined) {
                        statC.textContent = Math.round((1 - stats.compression_ratio) * 100) + '%';
                    }
                    statT.textContent = latency + 'ms';
                    statM.textContent = stats.active_memories || 0;
                    if (stats.cpu_usage_percent !== undefined)    statCpu.textContent = stats.cpu_usage_percent + '%';
                    if (stats.memory_usage_percent !== undefined) statRam.textContent = stats.memory_usage_percent + '%';
                    if (stats.context_used_percent !== undefined)  statCtx.textContent = stats.context_used_percent + '%';
                    if (stats.exact_tokens) {
                        const totalTokens = stats.exact_tokens.total;
                        const durationSec = latency / 1000.0;
                        statTps.textContent = Math.round(stats.exact_tokens.completion / durationSec) + ' t/s';
                    }

                    // Render Glass Box Inspector
                    const finalData = {
                        query: query,
                        tps: statTps.textContent,
                        query_cost: stats.query_cost || (stats.exact_tokens ? `$${((stats.exact_tokens.prompt * 0.000001) + (stats.exact_tokens.completion * 0.000002)).toFixed(6)}` : "$0.00"),
                        search_queries: stats.instantaneous_latency_ms ? stats.instantaneous_latency_ms.search_queries : [query],
                        hyde_doc: stats.hyde_doc || "N/A",
                        raw_prompt: stats.raw_prompt || "N/A",
                        stats: stats
                    };
                    renderInspector(finalData);
                }
            }
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            addLog(`Pipeline generation failure: ${err.message}`, "ERROR");
            bodyContainer.innerHTML = `<span style="color: var(--accent-red); font-weight: bold;">CRITICAL ERROR:</span> ${err.message}`;
            bodyContainer.classList.remove('typing-cursor');
        }
    } finally {
        AppState.setGenerating(false);
    }
});

// ---- Stop Request Button ----
if (stopBtn) {
    stopBtn.addEventListener('click', () => {
        if (AppState.abortController) {
            AppState.abortController.abort();
            addLog("Orchestration pipeline execution stopped by client request.", "WARNING");
            
            const cursorBubble = document.querySelector('.typing-cursor');
            if (cursorBubble) {
                cursorBubble.classList.remove('typing-cursor');
                cursorBubble.innerHTML += "<br><span style='color: var(--accent-amber); font-size: 0.75rem; font-weight: bold;'>[PIPELINE ABORTED]</span>";
            }
            AppState.setGenerating(false);
        }
    });
}

// ---- Telemetry Inspection Modal Logic ----
window.viewTelemetryDetails = function(telemetry, budget, query) {
    if (!telemetry) return;
    
    // Fallback bounds
    budget = budget || telemetry.budget_tracking || {};
    query = query || telemetry.query || "User Query";
    
    let stepsHtml = '';
    if (telemetry.steps && telemetry.steps.length > 0) {
        stepsHtml = `
            <div style="margin-top: 15px;">
                <h4 style="margin-bottom: 8px; color: var(--accent-amber);">STEP-BY-STEP RECOVERY LOGS</h4>
                <div class="modal-step-list">
                    ${telemetry.steps.map(step => `<div class="modal-step-item">${escapeHtml(step)}</div>`).join('')}
                </div>
            </div>
        `;
    } else {
        stepsHtml = `
            <div style="margin-top: 15px; color: var(--text-muted); font-style: italic; text-align: center;">
                No context overflow triggered for this turn. Context window remained safe.
            </div>
        `;
    }

    let promptHtml = '';
    if (telemetry.raw_prompt) {
        promptHtml = `
            <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;">
                <h4 style="margin-bottom: 8px; color: var(--accent-cyan); font-family: var(--font-brand);">COMPILED LLM PROMPT FOOTPRINT</h4>
                <div style="font-family: var(--font-mono); font-size: 0.65rem; max-height: 180px; overflow-y: auto; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.03); padding: 8px; white-space: pre-wrap; word-break: break-all; color: var(--text-secondary); border-radius: var(--radius-sm);">
                    ${escapeHtml(telemetry.raw_prompt)}
                </div>
            </div>
        `;
    }

    telemetryModalBody.innerHTML = `
        <div style="margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;">
            <span style="color: var(--text-muted); font-size: 0.65rem; text-transform: uppercase;">QUERY TEXT:</span>
            <div style="font-weight: 500; color: #fff; margin-top: 2px;">"${escapeHtml(query)}"</div>
        </div>
        
        <table class="modal-meta-table">
            <tr>
                <td>OVERFLOW STATE</td>
                <td style="color: ${telemetry.overflow_occurred ? 'var(--accent-amber)' : 'var(--accent-green)'}">
                    ${telemetry.overflow_occurred ? 'RECOVERED (BREACH RESOLVED)' : 'NOMINAL (NO BREACH)'}
                </td>
            </tr>
            <tr>
                <td>EFFECTIVE LIMIT</td>
                <td>${telemetry.limit} TKN</td>
            </tr>
            <tr>
                <td>INITIAL FOOTPRINT</td>
                <td>${telemetry.initial_tokens} TKN</td>
            </tr>
            <tr>
                <td>SAFE COMPILED SIZE</td>
                <td style="color: var(--accent-cyan)">${telemetry.final_tokens} TKN</td>
            </tr>
            <tr>
                <td>CONVERSATION MEMORY</td>
                <td>${budget.memory_tokens_used || 0} TKN</td>
            </tr>
            <tr>
                <td>COMPRESSED DOCUMENTS</td>
                <td>${budget.document_tokens_used || 0} TKN</td>
            </tr>
        </table>
        
        ${stepsHtml}
        ${promptHtml}
    `;
    
    telemetryModal.classList.add('open');
};

window.closeTelemetryModal = function() {
    telemetryModal.classList.remove('open');
};

// Close modal on click outside content card
telemetryModal.addEventListener('click', (e) => {
    if (e.target === telemetryModal) {
        closeTelemetryModal();
    }
});

// ---- File Upload API ----
document.getElementById('drop-area').addEventListener('click', () => document.getElementById('file-in').click());
document.getElementById('file-in').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const feedback = document.getElementById('up-msg');
    feedback.style.display = 'block';
    feedback.textContent = `Uploading: ${file.name}...`;
    addLog(`Initiating file injection: ${file.name}`, 'UPLOAD');
    
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
        if (res.ok) { 
            const data = await res.json();
            feedback.textContent = `Indexed successfully!`;
            addLog(`Injection completed: ${data.message}`, "SUCCESS"); 
            refreshGlobalStats(); 
        } else {
            const err = await res.json();
            feedback.textContent = `Upload failed.`;
            addLog(`Injection failed: ${err.detail || 'HTTP Error'}`, "ERROR");
        }
    } catch (e) { 
        feedback.textContent = `Communication error.`;
        addLog(`Injection communication error: ${e.message}`, "ERROR"); 
    }
    
    setTimeout(() => { feedback.style.display = 'none'; }, 4000);
});

// ---- Background Polling ----
async function refreshGlobalStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        statQ.textContent = data.queries_handled || 0;
        statC.textContent = Math.round((1 - data.avg_compression) * 100) + '%';
        if (data.document_count !== undefined) {
            knowledgeCount.textContent = data.document_count;
        }
        if (data.cpu_usage_percent !== undefined)      statCpu.textContent = Math.round(data.cpu_usage_percent) + '%';
        if (data.memory_usage_percent !== undefined)   statRam.textContent = Math.round(data.memory_usage_percent) + '%';
    } catch (e) {}
}

// ---- Thread Restoration (Database history load) ----
async function loadHistory() {
    try {
        addLog("Restoring session conversation thread...", "SYSTEM");
        const res = await fetch(`${API_BASE}/history/${sid}`);
        if (!res.ok) return;
        
        const history = await res.json();
        chatWindow.innerHTML = '';
        
        if (history && history.length > 0) {
            let turnsRestored = 0;
            // Loop in pairs or reconstruct footer if assistant has telemetry
            for (let i = 0; i < history.length; i++) {
                const item = history[i];
                const uiRole = item.role === 'assistant' ? 'ai' : 'user';
                
                // If it is user, just render normally
                if (uiRole === 'user') {
                    addMsg(item.text, 'user');
                } else {
                    // It is assistant, pass along database saved telemetry
                    addMsg(item.text, 'ai', item.telemetry);
                    turnsRestored++;
                }
            }
            addLog(`Restored ${history.length} turns (${turnsRestored} stats-footer records).`, "SUCCESS");
        }
    } catch (e) {
        console.error(e);
        addLog("Failed to restore session history from persistent database.", "WARNING");
    }
}

// Initializers
setInterval(refreshGlobalStats, 15000);
refreshGlobalStats();
loadHistory();
AppState.updateContextLimit(parseInt(contextLimitSlider.value));
