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

// ---- Query Submission ----
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    addMsg(query, 'user');
    const mode = document.querySelector('input[name="engine-mode"]:checked').value;
    addLog(`Query received: ${mode}`, 'REQUEST');

    btn.disabled = true;
    const startTime = Date.now();

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query, session_id: sid, mode: mode })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server Error");
        }

        const data = await response.json();
        const latency = Date.now() - startTime;

        addLog(`Processing complete: ${latency}ms`, 'SUCCESS');

        if (data.response) {
            addMsg(data.response, 'ai');

            // Source context
            reconWindow.innerHTML = '';
            if (data.retrieved_context) {
                data.retrieved_context.forEach(hit => {
                    const item = document.createElement('div');
                    item.className = 'readout-item';
                    item.innerHTML = `${hit.text.substring(0, 180)}...<br><div style="margin-top:0.4rem;font-size:0.55rem;color:var(--text-dim)">RELEVANCE: ${hit.score.toFixed(4)} | SOURCE: ${hit.source}</div>`;
                    reconWindow.appendChild(item);
                });
            }

            // Stat panel
            if (data.stats) {
                statQ.textContent = data.stats.queries_handled;
                statC.textContent = Math.round((1 - data.stats.compression_ratio) * 100) + '%';
                statT.textContent = latency + 'ms';
                statM.textContent = data.stats.active_memories || 0;
                if (data.stats.cpu_usage_percent !== undefined)    statCpu.textContent = data.stats.cpu_usage_percent + '%';
                if (data.stats.memory_usage_percent !== undefined) statRam.textContent = data.stats.memory_usage_percent + '%';
                if (data.tps !== undefined)        statTps.textContent = data.tps + ' t/s';
                if (data.query_cost !== undefined)  statCost.textContent = data.query_cost;

                if (data.stats.exact_tokens) {
                    statCtx.textContent = data.stats.context_used_percent + '% (' + data.stats.exact_tokens.prompt + ' TKN)';
                } else if (data.stats.context_used_percent !== undefined) {
                    statCtx.textContent = data.stats.context_used_percent + '%';
                }
            }

            // Glass Box Inspector
            renderInspector(data);
        }
    } catch (err) {
        addLog("Communication failure", "ERROR");
        addMsg(`CRITICAL ERROR: ${err.message}`, "ai");
    } finally {
        btn.disabled = false;
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
