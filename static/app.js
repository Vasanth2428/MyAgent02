/**
 * ================================================================
 * RAG CONTROL CONSOLE — Application Logic
 * ================================================================
 */

const API_BASE = `${window.location.origin}`;

// ---- DOM References ----
const form              = document.getElementById('comm-form');
const input             = document.getElementById('comm-input');
const chatWindow        = document.getElementById('chat-window');
const logWindow         = document.getElementById('log-window');
const inspectorWindow   = document.getElementById('inspector-window');
const reconWindow       = document.getElementById('recon-window');
const btn               = document.getElementById('comm-btn');
const sessionTag        = document.getElementById('session-tag');
const knowledgeCount    = document.getElementById('knowledge-count');

const statQ    = document.getElementById('stat-q');
const statC    = document.getElementById('stat-c');
const statT    = document.getElementById('stat-t');
const statM    = document.getElementById('stat-m');
const statCpu  = document.getElementById('stat-cpu');
const statRam  = document.getElementById('stat-ram');
const statCtx  = document.getElementById('stat-ctx');
const statTps  = document.getElementById('stat-tps');
const statCost = document.getElementById('stat-cost');

// ---- Session Management ----
let sid = localStorage.getItem('station_sid') || 'SID-' + Math.random().toString(36).substr(2, 6).toUpperCase();
localStorage.setItem('station_sid', sid);
sessionTag.textContent = `SID: ${sid}`;

// ---- Helpers ----
function addLog(msg, type = 'INFO') {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${time}]</span><span class="log-type">${type}</span> ${msg}`;
    logWindow.appendChild(entry);
    logWindow.scrollTop = logWindow.scrollHeight;
}

function addMsg(text, type = 'ai') {
    const msg = document.createElement('div');
    msg.className = `message msg-${type}`;
    let formattedText = text;
    if (type === 'ai') {
        try { formattedText = typeof marked !== 'undefined' ? marked.parse(text) : text; }
        catch (e) { formattedText = text; }
    }
    msg.innerHTML = `<div class="msg-header">${type === 'user' ? 'USER_INPUT' : 'SYSTEM_OUTPUT'}</div><div class="msg-body">${formattedText}</div>`;
    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return msg;
}

function safeGet(obj, path, fallback = 'N/A') {
    return path.split('.').reduce((o, k) => (o && o[k] != null) ? o[k] : fallback, obj);
}

// ---- Inspector Renderer ----
function renderInspector(data) {
    const tokens  = data.stats?.exact_tokens || { prompt: 0, completion: 0, total: 0 };
    const budget  = data.stats?.budget_tracking || {};
    const latency = data.stats?.instantaneous_latency_ms || {};

    inspectorWindow.innerHTML = `
        <div style="margin-bottom: 8px; color: #fff;"><strong>[SYSTEM] EXACT TOKEN CONSUMPTION:</strong></div>
        <div style="color: var(--text-dim); margin-bottom: 12px; background: #111; padding: 6px;">
            PROMPT: ${tokens.prompt} | COMPLETION: ${tokens.completion} | TOTAL: ${tokens.total}<br>
            PROMPT BREAKDOWN: Instructions: ~40 TKN | Memory: ${budget.memory_tokens_used || 0} TKN | Knowledge: ${budget.document_tokens_used || 0} TKN | Query: ~${Math.round((data.query || '').length / 4)} TKN<br>
            GENERATION SPEED: ${data.tps || 0} t/s | TRANSACTION COST: ${data.query_cost || '$0.00000000'}
        </div>
        <div style="margin-bottom: 8px; color: #fff;"><strong>[SYSTEM] DYNAMIC BUDGET ALLOCATION:</strong></div>
        <div style="color: var(--text-dim); margin-bottom: 12px; background: #111; padding: 6px;">
            MEMORY USED: ${budget.memory_tokens_used || 0} / ${budget.memory_tokens_limit || 0} TKN<br>
            KNOWLEDGE USED: ${budget.document_tokens_used || 0} / ${budget.document_tokens_limit || 0} TKN
        </div>
        <div style="margin-bottom: 8px; color: #fff;"><strong>[SYSTEM] PIPELINE TELEMETRY:</strong></div>
        <div style="color: var(--text-dim); margin-bottom: 12px; background: #111; padding: 6px;">
            MODE: ${(data.stats?.mode || 'N/A').toUpperCase()}<br>
            HYBRID ALPHA: ${data.stats?.alpha || 0.5}<br>
            PEAK CROSS-ENCODER SCORE: ${data.stats?.reranker_peak_score || 0}<br>
            COMPRESSION RATIO: ${((data.stats?.compression_ratio || 0) * 100).toFixed(1)}%<br>
            LOCAL EMBED LATENCY: ${latency.phase_2_embed_generation_ms || 0} ms<br>
            WEAVIATE SEARCH LATENCY: ${latency.phase_2_weaviate_search_ms || 0} ms<br>
            HYDE GENERATION LATENCY: ${latency.phase_1_5_hyde_ms || 0} ms
        </div>
        <div style="margin-bottom: 8px; color: #fff;"><strong>[PHASE 1] LLM QUERY EXPANSIONS:</strong></div>
        <div style="color: var(--text-dim); margin-bottom: 12px; background: #111; padding: 6px;">
            ${data.search_queries ? data.search_queries.map(q => `> ${q}`).join('<br>') : 'N/A'}
        </div>
        <div style="margin-bottom: 8px; color: #fff;"><strong>[PHASE 1.5] HYPOTHETICAL RESPONSE (HyDE):</strong></div>
        <div style="color: var(--text-dim); margin-bottom: 12px; background: #111; padding: 6px; white-space: pre-wrap; word-break: break-word;">
            ${data.hyde_doc || 'N/A'}
        </div>
        <div style="margin-bottom: 8px; color: #fff;"><strong>[PHASE 6] RAW GENERATION PROMPT:</strong></div>
        <div style="color: var(--text-dim); background: #111; padding: 6px; white-space: pre-wrap; word-break: break-all;">${data.raw_prompt ? data.raw_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;') : 'N/A'}</div>
    `;
}

// ---- Session / Thread Abort Controller ----
let abortController = null;
const stopBtn = document.getElementById('stop-btn');

if (stopBtn) {
    stopBtn.addEventListener('click', () => {
        if (abortController) {
            abortController.abort();
            addLog("Execution interrupted by user.", "CANCEL");
            
            // Find typing bubbles and mark interrupted
            const cursorBubble = document.querySelector('.typing-cursor');
            if (cursorBubble) {
                cursorBubble.classList.remove('typing-cursor');
                cursorBubble.innerHTML += "<br><span style='color: #ff9900; font-size: 0.75rem; font-weight: bold;'>[EXECUTION INTERRUPTED]</span>";
            }
            btn.disabled = false;
            stopBtn.style.display = 'none';
        }
    });
}

// ---- Query Submission ----
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    addMsg(query, 'user');
    const mode = document.querySelector('input[name="engine-mode"]:checked').value;
    addLog(`Query received (Mode: ${mode})`, 'REQUEST');

    btn.disabled = true;
    if (stopBtn) stopBtn.style.display = 'inline-block';
    
    // Clear dynamic readout panels
    reconWindow.innerHTML = '';
    inspectorWindow.innerHTML = '<div style="color: var(--text-dim);">Awaiting telemetry stream...</div>';

    abortController = new AbortController();
    const startTime = Date.now();

    // Create the AI chat bubble ahead of time for streaming
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
            body: JSON.stringify({ question: query, session_id: sid, mode: mode }),
            signal: abortController.signal
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
            
            // Save the last partial line back to the buffer
            buffer = lines.pop();

            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith("data:")) continue;

                const jsonStr = cleanLine.substring(5).trim();
                if (!jsonStr) continue;

                let data;
                try {
                    data = JSON.parse(jsonStr);
                } catch (parseErr) {
                    console.error("Failed to parse SSE payload", jsonStr, parseErr);
                    continue;
                }

                if (data.event === "thought") {
                    addLog(data.text, "THOUGHT");
                    
                    const entry = document.createElement('div');
                    entry.className = 'log-thought';
                    entry.textContent = `💭 ${data.text}`;
                    logWindow.appendChild(entry);
                    logWindow.scrollTop = logWindow.scrollHeight;
                } 
                else if (data.event === "action") {
                    addLog(`Executing: ${data.tool}(${data.input})`, "ACTION");
                    
                    const entry = document.createElement('div');
                    entry.className = 'log-action';
                    entry.textContent = `⚙️ Action: ${data.tool}[${data.input}]`;
                    logWindow.appendChild(entry);
                    logWindow.scrollTop = logWindow.scrollHeight;
                }
                else if (data.event === "observation") {
                    addLog(`Observation received (${data.output.length} chars)`, "OBSERVATION");
                    
                    const entry = document.createElement('div');
                    entry.className = 'log-observation';
                    entry.textContent = `📄 Observation: ${data.output}`;
                    logWindow.appendChild(entry);
                    logWindow.scrollTop = logWindow.scrollHeight;
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

                    // If we got back any raw document context inside stats, populate it
                    if (data.stats && data.stats.retrieved_context) {
                        reconWindow.innerHTML = '';
                        data.stats.retrieved_context.forEach(hit => {
                            const item = document.createElement('div');
                            item.className = 'readout-item';
                            item.innerHTML = `${hit.text.substring(0, 180)}...<br><div style="margin-top:0.4rem;font-size:0.55rem;color:var(--text-dim)">RELEVANCE: ${hit.score.toFixed(4)} | SOURCE: ${hit.source}</div>`;
                            reconWindow.appendChild(item);
                        });
                    }

                    // Render telemetry and update inspector
                    const finalData = {
                        query: query,
                        tps: data.stats?.tps || Math.round(accumulatedText.split(/\s+/).length / (latency / 1000)),
                        query_cost: data.stats?.query_cost || "$0.00",
                        search_queries: data.stats?.instantaneous_latency_ms ? data.stats.instantaneous_latency_ms.search_queries : [],
                        hyde_doc: data.stats?.hyde_doc || "",
                        raw_prompt: data.stats?.raw_prompt || "",
                        stats: data.stats || {}
                    };

                    // Update stats panel
                    if (data.stats) {
                        statQ.textContent = data.stats.queries_handled || statQ.textContent;
                        if (data.stats.compression_ratio !== undefined) {
                            statC.textContent = Math.round((1 - data.stats.compression_ratio) * 100) + '%';
                        }
                        statT.textContent = latency + 'ms';
                        statM.textContent = data.stats.active_memories || 0;
                        if (data.stats.cpu_usage_percent !== undefined)    statCpu.textContent = data.stats.cpu_usage_percent + '%';
                        if (data.stats.memory_usage_percent !== undefined) statRam.textContent = data.stats.memory_usage_percent + '%';
                    }

                    // Render Glass Box Inspector
                    renderInspector(finalData);
                }
            }
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            addLog(`Communication failure: ${err.message}`, "ERROR");
            bodyContainer.innerHTML = `<span style="color: #ff4444; font-weight: bold;">CRITICAL ERROR:</span> ${err.message}`;
            bodyContainer.classList.remove('typing-cursor');
        }
    } finally {
        btn.disabled = false;
        if (stopBtn) stopBtn.style.display = 'none';
        abortController = null;
    }
});

// ---- File Upload ----
document.getElementById('drop-area').addEventListener('click', () => document.getElementById('file-in').click());
document.getElementById('file-in').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    addLog(`File injection: ${file.name}`, 'UPLOAD');
    const fd = new FormData();
    fd.append('file', file);
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
        if (res.ok) { addLog("Injection successful", "SUCCESS"); refreshGlobalStats(); }
    } catch (e) { addLog("Injection failure", "ERROR"); }
});

// ---- Background Polling ----
async function refreshGlobalStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        statQ.textContent = data.queries_handled || 0;
        statC.textContent = Math.round((1 - data.avg_compression) * 100) + '%';
        if (data.document_count !== undefined)        knowledgeCount.textContent = data.document_count;
        if (data.cpu_usage_percent !== undefined)      statCpu.textContent = data.cpu_usage_percent + '%';
        if (data.memory_usage_percent !== undefined)   statRam.textContent = data.memory_usage_percent + '%';
    } catch (e) {}
}

async function loadHistory() {
    try {
        addLog("Restoring session thread...", "SYSTEM");
        const res = await fetch(`${API_BASE}/history/${sid}`);
        if (!res.ok) return;
        const history = await res.json();
        chatWindow.innerHTML = '';
        if (history && history.length > 0) {
            history.forEach(item => {
                const uiRole = item.role === 'assistant' ? 'ai' : 'user';
                addMsg(item.text, uiRole);
            });
            addLog(`Restored ${history.length} dialogue turns.`, "SUCCESS");
        }
    } catch (e) {
        addLog("Failed to restore thread history", "WARNING");
    }
}

setInterval(refreshGlobalStats, 15000);
refreshGlobalStats();
loadHistory();
