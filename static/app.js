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
const statQ         = document.getElementById('stat-q');
const statC         = document.getElementById('stat-c');
const statT         = document.getElementById('stat-t');
const statM         = document.getElementById('stat-m');
const statCost      = document.getElementById('stat-cost');
const statGrounding = document.getElementById('stat-grounding');
const statHits      = document.getElementById('stat-hits');
const statOverflows = document.getElementById('stat-overflows');

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

// Toast Container
const toastContainer          = document.getElementById('toast-container');

// Scroll-to-Bottom FAB
const scrollBottomFab         = document.getElementById('scroll-bottom-fab');

// ---- Session State Variables ----
// sid is managed by ConversationManager; initialise to last used or a fresh ID
let sid = localStorage.getItem('station_sid') || 'SID-' + Math.random().toString(36).substr(2, 8).toUpperCase();
localStorage.setItem('station_sid', sid);
if (sessionTag) sessionTag.textContent = `SID: ${sid}`;

let contextLimit = parseInt(contextLimitSlider.value);
let abortController = null;
let isInParallelMode = false;
let sessionOverflows = 0;

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

// ---- Mode Selector Listeners ----
document.querySelectorAll('input[name="engine-mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.checked) {
            const mode = e.target.value;
            activeModeBadge.textContent = mode === 'agentic' ? 'COOPERATIVE MULTI-AGENT' : mode.replace('_', ' ').toUpperCase();
            addLog(`Processing mode changed to ${mode === 'agentic' ? 'COOPERATIVE MULTI-AGENT' : mode.replace('_', ' ').toUpperCase()}`, "SYSTEM");
        }
    });
});

// ---- Custom Dropdown Mode Selector ----
const modeDropdownContainer = document.querySelector('.mode-dropdown-container');
const modeDropdownBtn = document.getElementById('mode-dropdown-btn');
const modeDropdownItems = document.querySelectorAll('.mode-dropdown-item');

if (modeDropdownBtn && modeDropdownContainer) {
    modeDropdownBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modeDropdownContainer.classList.toggle('open');
        const isOpen = modeDropdownContainer.classList.contains('open');
        modeDropdownBtn.setAttribute('aria-expanded', isOpen);
    });

    document.addEventListener('click', (e) => {
        if (modeDropdownContainer && !modeDropdownContainer.contains(e.target)) {
            modeDropdownContainer.classList.remove('open');
            modeDropdownBtn.setAttribute('aria-expanded', 'false');
        }
    });

    modeDropdownItems.forEach(item => {
        item.addEventListener('click', () => {
            const modeVal = item.dataset.value;
            const targetRadio = document.getElementById(`mode-${modeVal}-radio`);
            if (targetRadio) {
                targetRadio.checked = true;
                targetRadio.dispatchEvent(new Event('change'));
            }

            // Sync active visual class
            modeDropdownItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            modeDropdownContainer.classList.remove('open');
            modeDropdownBtn.setAttribute('aria-expanded', 'false');
        });
    });
}

// ---- Slider & Preset Listeners ----
contextLimitSlider.addEventListener('input', (() => {
    let debounceTimer = null;
    return (e) => {
        const val = parseInt(e.target.value);
        // Immediate visual feedback
        contextLimitSliderVal.textContent = val;
        tokenLimitValReadout.textContent = val;
        // Debounce the heavier update
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            AppState.updateContextLimit(val);
        }, 60);
    };
})());

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

// ---- Toast Notification System ----
function showToast(message, type = 'info') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    // Auto-remove after animation completes
    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 4000);
}

// ---- Keyboard Shortcut (Enter to submit, Shift+Enter for newline) ----
if (input) {
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            if (!AppState.isGenerating && input.value.trim()) {
                form.dispatchEvent(new Event('submit', { cancelable: true }));
            }
        }
    });
}

// ---- Keyboard Shortcut (Ctrl+Enter globally) ----
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!AppState.isGenerating && input.value.trim()) {
            form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    }
});

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
    
    // Parse telemetryData if it's a string
    if (type === 'ai' && typeof telemetryData === 'string') {
        try { telemetryData = JSON.parse(telemetryData); } catch (e) { telemetryData = null; }
    }
    
    // Build thinking accordion if agent steps are in telemetry history
    let thinkingHtml = '';
    if (type === 'ai' && telemetryData && telemetryData.debug_info && telemetryData.debug_info.actions_taken && telemetryData.debug_info.actions_taken.length > 0) {
        const actions = telemetryData.debug_info.actions_taken;
        let stepsLinesHtml = '';
        actions.forEach(act => {
            if (act.thought) {
                stepsLinesHtml += `<div class="step-thought-log">🧠 Thought: ${escapeHtml(act.thought)}</div>`;
            }
            if (act.tool) {
                stepsLinesHtml += `<div class="step-action-log">🔧 Action: Call ${escapeHtml(act.tool)} with argument "${escapeHtml(act.input)}"</div>`;
            }
            if (act.observation) {
                stepsLinesHtml += `<div class="step-obs-log">🔍 Observation: ${escapeHtml(act.observation.substring(0, 150))}...</div>`;
            }
        });
        
        thinkingHtml = `
            <details class="thinking-accordion" open>
                <summary class="thinking-summary">
                    <span style="color: var(--accent-emerald); font-weight: bold; font-size: 0.8rem; margin-right: 0.2rem;">✓</span>
                    <span class="thinking-status">Thought process completed</span>
                </summary>
                <div class="thinking-details">
                    ${stepsLinesHtml}
                </div>
            </details>
        `;
    }
    
    // Build main content
    let contentHtml = `<div class="msg-header">${type === 'user' ? 'USER_INPUT' : 'SYSTEM_OUTPUT'}</div>${thinkingHtml}<div class="msg-body">${formattedText}</div>`;
    
    // If AI Turn has telemetry, append footnote badge
    if (type === 'ai' && telemetryData) {
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
                <button type="button" class="telemetry-inspect-btn">
                    INSPECT
                </button>
            </div>
        `;
    }
    
    msg.innerHTML = contentHtml;

    if (type === 'ai' && telemetryData) {
        const inspectBtn = msg.querySelector('.telemetry-inspect-btn');
        if (inspectBtn) {
            inspectBtn.addEventListener('click', () => {
                viewTelemetryDetails(telemetryData, telemetryData.budget_tracking, telemetryData.query);
            });
        }
    }

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
    const budget  = data.stats?.budget_tracking || {};
    const latency = data.stats?.instantaneous_latency_ms || {};

    // Retrieve the traces we built in real-time
    const stepsListContainer = document.getElementById('agent-steps-list');
    let stepsHtml = '';
    if (stepsListContainer) {
        // Ensure all are marked complete now
        const activeSteps = stepsListContainer.querySelectorAll('.agent-step-card.active');
        activeSteps.forEach(card => {
            card.classList.remove('active');
            card.classList.add('completed');
            const statusBadge = card.querySelector('.step-status-badge');
            if (statusBadge) {
                statusBadge.textContent = 'COMPLETED';
                statusBadge.className = 'step-status-badge status-completed';
            }
        });
        stepsHtml = stepsListContainer.innerHTML;
    } else {
        stepsHtml = '<div style="color: var(--text-muted); font-style: italic;">No execution trace available.</div>';
    }

    let debugSection = '';
    if (data.stats?.debug_info) {
        const dbg = data.stats.debug_info;
        const goalsHtml = dbg.goals_set.length > 0 
            ? dbg.goals_set.map((g, i) => `<div class="agent-goal-item">${i+1}. ${escapeHtml(g)}</div>`).join('')
            : '<div style="color: var(--accent-amber); margin-left: 8px;">No goals recorded</div>';
        
        debugSection = `
        <div class="inspector-section">
            <div class="inspector-section-hdr">AGENTIC EXECUTION TRACE</div>
            <div class="inspector-section-body" style="font-family: var(--font-mono); font-size: 0.7rem; display: flex; flex-direction: column; gap: 8px;">
                <div style="margin-bottom: 6px;">LLM CALLS: ${dbg.llm_calls}</div>
                <div style="margin-bottom: 6px;">GOALS SET:</div>
                ${goalsHtml}
                <div style="margin-top: 6px; margin-bottom: 4px;">PROCESS DETAILS:</div>
                <div class="agent-step-list-final" style="display: flex; flex-direction: column; gap: 8px;">${stepsHtml}</div>
            </div>
        </div>`;
    }

    inspectorWindow.innerHTML = `
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
GROUNDING SCORE: ${(data.stats?.grounding_score || 0).toFixed(3)} (heuristic only)
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
        ${debugSection}
    `;
}

// ---- Real-Time Agentic Loop Processing Monitor ----
function addOrUpdateStep(stepType, name, detail = '') {
    const listContainer = document.getElementById('agent-steps-list');
    if (!listContainer) return;
    
    // Clear empty message
    const emptyMsg = listContainer.querySelector('.empty-steps-msg');
    if (emptyMsg) emptyMsg.remove();
    
    if (stepType === 'node') {
        const isParallelNode = isInParallelMode && name !== 'aggregate_parallel_results_node' && name !== 'supervisor_node';
        
        // Mark previous active steps as completed, unless we are entering/running parallel branches
        const activeSteps = listContainer.querySelectorAll('.agent-step-card.active');
        activeSteps.forEach(card => {
            const isCardParallel = card.classList.contains('parallel-branch');
            const shouldComplete = !isParallelNode || !isCardParallel;
            
            if (shouldComplete) {
                card.classList.remove('active');
                card.classList.add('completed');
                const statusBadge = card.querySelector('.step-status-badge');
                if (statusBadge) {
                    statusBadge.textContent = 'COMPLETED';
                    statusBadge.className = 'step-status-badge status-completed';
                }
            }
        });
        
        const friendlyNames = {
            'supervisor_node': 'Routing Supervisor [Cooperative Planner]',
            'rag_worker_node': 'RAG Specialist [Knowledge Retrieval]',
            'web_worker_node': 'Web Search Specialist [Internet Queries]',
            'utility_worker_node': 'Utility Specialist [Computations & Logic]',
            'scraper_worker_node': 'Scraper Specialist [URL Extraction]',
            'critic_worker_node': 'Critic Specialist [Fact-Check & Audit]',
            'report_worker_node': 'Report Specialist [Document Generation]',
            'synthesizer_node': 'Response Synthesizer [Final Fusion]',
            'aggregate_parallel_results_node': 'Result Aggregator [Cooperative Join]',
            'early_exit_check': 'Early Exit Validation',
            'early_exit_execute': 'Fast Path Response',
            'overflow_recovery': 'Context Overflow Safeguard',
            'reasoning': 'ReAct Agent Reasoning',
            'execute_formatting_error': 'Format Correction Handler',
            'execute_tool': 'Tool Execution Core',
            'synthesis': 'Final Answer Synthesis',
            'streaming_final_answer': 'Streaming Assistant Output',
            'WAITING_FOR_REASONING': 'LLM Inference Reasoning',
            'WAITING_FOR_ACTION': 'Pipeline Gating & Routing',
            'EXECUTING_TOOL': 'Information Extraction & Search',
            'WAITING_FOR_FINAL_ANSWER': 'Compiling Context Data',
            'STREAMING_FINAL_RESPONSE': 'Generating Final Response Stream'
        };
        const displayName = friendlyNames[name] || name;
        
        const card = document.createElement('div');
        let cardClass = 'agent-step-card active';
        let parallelBadge = '';
        if (isParallelNode) {
            cardClass += ' parallel-branch';
            parallelBadge = '<span class="step-parallel-badge">COOPERATIVE PARALLEL</span>';
        }
        card.className = cardClass;
        card.id = `step-node-${name}-${Date.now()}`; // unique id to prevent clash
        card.innerHTML = `
            <div class="step-header">
                <span class="step-icon-dot"></span>
                <span class="step-name">${escapeHtml(displayName)} ${parallelBadge}</span>
                <span class="step-status-badge status-active">ACTIVE</span>
            </div>
            <div class="step-details" style="display: none;"></div>
        `;
        listContainer.appendChild(card);
        listContainer.scrollTop = listContainer.scrollHeight;
    } else {
        // Find the last step (current active one)
        const lastCard = listContainer.lastElementChild;
        if (lastCard && lastCard.classList.contains('agent-step-card')) {
            const detailsDiv = lastCard.querySelector('.step-details');
            if (detailsDiv) {
                detailsDiv.style.display = 'block';
                const logDiv = document.createElement('div');
                if (stepType === 'thought') {
                    logDiv.className = 'step-thought-log';
                    logDiv.innerHTML = `<span class="step-detail-label">THOUGHT:</span> ${escapeHtml(detail)}`;
                } else if (stepType === 'action') {
                    logDiv.className = 'step-action-log';
                    logDiv.innerHTML = `<span class="step-detail-label">ACTION:</span> Executing <strong>${escapeHtml(name)}</strong> with: <code>${escapeHtml(detail)}</code>`;
                } else if (stepType === 'observation') {
                    logDiv.className = 'step-obs-log';
                    logDiv.innerHTML = `<span class="step-detail-label">OBSERVATION:</span> ${escapeHtml(detail.substring(0, 150))}${detail.length > 150 ? '...' : ''}`;
                }
                detailsDiv.appendChild(logDiv);
            }
        } else {
            // Fallback: If no step card exists (e.g. initial phases), add a simple log line
            const line = document.createElement('div');
            line.className = 'step-simple-log';
            line.innerHTML = `&gt; ${escapeHtml(detail || name)}`;
            listContainer.appendChild(line);
        }
        listContainer.scrollTop = listContainer.scrollHeight;
    }
}

// ---- SSE Streaming Controller ----
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    input.style.height = 'auto'; // Reset textarea height on submit
    addMsg(query, 'user');
    
    // Auto-title from the first message
    setTimeout(() => {
        if (typeof ConversationManager !== 'undefined') {
            ConversationManager.autoTitleFromFirstMessage(query);
        }
    }, 600);
    
    const mode = document.querySelector('input[name="engine-mode"]:checked').value;
    activeModeBadge.textContent = mode === 'agentic' ? 'COOPERATIVE MULTI-AGENT' : mode.replace('_', ' ').toUpperCase();
    isInParallelMode = false; // Reset parallel tracking state
    
    addLog(`Initiating streaming request (Mode: ${mode} | Limit: ${contextLimit} TKN)`, 'REQUEST');

    // Reset/Clear UI state for query run
    reconWindow.innerHTML = '';
    inspectorWindow.innerHTML = `
        <div class="inspector-section">
            <div class="inspector-section-hdr">PROCESSING MONITOR (REAL TIME)</div>
            <div class="inspector-section-body" style="font-family: var(--font-mono); font-size: 0.7rem;">
                <div class="agent-step-list" id="agent-steps-list" style="max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;">
                    <div class="empty-steps-msg" style="color: var(--text-muted); font-style: italic;">Awaiting pipeline initiation...</div>
                </div>
            </div>
        </div>
    `;
    
    // Clear and reset context overflow debugger
    overflowLogWindow.innerHTML = '';
    overflowIndicatorDot.className = 'indicator-dot nominal';
    overflowAlertBanner.className = 'overflow-banner alert-nominal';
    overflowAlertBanner.textContent = 'SYSTEM RUNNING IN NOMINAL STATE';

    let inlineThinkingAccordion = null;
    let inlineThinkingDetails = null;
    let thinkingStart = null;

    // Streaming rendering & Throttled scheduler state
    let accumulatedText = "";
    let lastRenderTime = 0;
    let renderThrottleTimeout = null;
    let streamTextContainer = null;

    const maybeScrollToBottom = () => {
        const threshold = 100;
        const isNearBottom = chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight <= threshold;
        if (isNearBottom) {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    };

    const updateStreamUI = (isFinal = false) => {
        if (isFinal) {
            if (renderThrottleTimeout) {
                clearTimeout(renderThrottleTimeout);
                renderThrottleTimeout = null;
            }
            try {
                bodyContainer.innerHTML = typeof marked !== 'undefined' ? marked.parse(accumulatedText) : accumulatedText;
            } catch (e) {
                bodyContainer.textContent = accumulatedText;
            }
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return;
        }

        // Ensure stream text node container exists
        if (!streamTextContainer) {
            streamTextContainer = document.createElement('span');
            streamTextContainer.className = 'raw-text-stream';
            bodyContainer.innerHTML = '';
            bodyContainer.appendChild(streamTextContainer);
        }

        const now = Date.now();
        const renderInterval = 80; // 80ms throttle (approx 12fps)

        const performRender = () => {
            streamTextContainer.textContent = accumulatedText;
            maybeScrollToBottom();
            lastRenderTime = Date.now();
            renderThrottleTimeout = null;
        };

        if (now - lastRenderTime >= renderInterval) {
            if (renderThrottleTimeout) {
                clearTimeout(renderThrottleTimeout);
                renderThrottleTimeout = null;
            }
            performRender();
        } else if (!renderThrottleTimeout) {
            renderThrottleTimeout = setTimeout(performRender, renderInterval - (now - lastRenderTime));
        }
    };

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

                const ensureAccordion = () => {
                    if (!inlineThinkingAccordion) {
                        thinkingStart = Date.now();
                        inlineThinkingAccordion = document.createElement('details');
                        inlineThinkingAccordion.className = 'thinking-accordion';
                        inlineThinkingAccordion.setAttribute('open', '');
                        inlineThinkingAccordion.innerHTML = `
                            <summary class="thinking-summary">
                                <span class="thinking-spinner"></span>
                                <span class="thinking-status">Thinking...</span>
                            </summary>
                            <div class="thinking-details"></div>
                        `;
                        bodyContainer.parentNode.insertBefore(inlineThinkingAccordion, bodyContainer);
                        inlineThinkingDetails = inlineThinkingAccordion.querySelector('.thinking-details');
                    }
                };

                if (data.event === "node_start") {
                    if (data.node === "aggregate_parallel_results_node") {
                        isInParallelMode = false;
                    }
                    addOrUpdateStep("node", data.node);
                }
                else if (data.event === "thought") {
                    addLog(data.text, "THOUGHT");
                    addOrUpdateStep("thought", "", data.text);
                    ensureAccordion();
                    const line = document.createElement('div');
                    line.className = 'step-thought-log';
                    line.innerHTML = `🧠 Thought: ${escapeHtml(data.text)}`;
                    inlineThinkingDetails.appendChild(line);
                    inlineThinkingDetails.scrollTop = inlineThinkingDetails.scrollHeight;
                } 
                else if (data.event === "action") {
                    if (data.tool === "Parallel Dispatch") {
                        isInParallelMode = true;
                    }
                    
                    let toolName = data.tool;
                    if (toolName.startsWith("Route to ")) {
                        const target = toolName.replace("Route to ", "") + "_node";
                        const mapping = {
                            'supervisor_node': 'Routing Supervisor [Cooperative Planner]',
                            'rag_worker_node': 'RAG Specialist [Knowledge Retrieval]',
                            'web_worker_node': 'Web Search Specialist [Internet Queries]',
                            'utility_worker_node': 'Utility Specialist [Computations & Logic]',
                            'scraper_worker_node': 'Scraper Specialist [URL Extraction]',
                            'critic_worker_node': 'Critic Specialist [Fact-Check & Audit]',
                            'report_worker_node': 'Report Specialist [Document Generation]',
                            'synthesizer_node': 'Response Synthesizer [Final Fusion]',
                            'aggregate_parallel_results_node': 'Result Aggregator [Cooperative Join]'
                        };
                        const friendlyTarget = mapping[target] || target.replace("_node", "");
                        toolName = `Delegate to ${friendlyTarget}`;
                    }
                    
                    addLog(`Executing tool: ${data.tool}[${data.input}]`, "ACTION");
                    addOrUpdateStep("action", toolName, data.input);
                    ensureAccordion();
                    const line = document.createElement('div');
                    line.className = 'step-action-log';
                    line.innerHTML = `🔧 Action: ${escapeHtml(toolName)} with argument "${escapeHtml(data.input)}"`;
                    inlineThinkingDetails.appendChild(line);
                    inlineThinkingDetails.scrollTop = inlineThinkingDetails.scrollHeight;
                }
                else if (data.event === "observation") {
                    addLog(`Received tool observation (${data.output.length} chars)`, "OBSERVATION");
                    addOrUpdateStep("observation", "", data.output);
                    ensureAccordion();
                    const line = document.createElement('div');
                    line.className = 'step-obs-log';
                    line.innerHTML = `🔍 Observation: ${escapeHtml(data.output.substring(0, 150))}${data.output.length > 150 ? '...' : ''}`;
                    inlineThinkingDetails.appendChild(line);
                    inlineThinkingDetails.scrollTop = inlineThinkingDetails.scrollHeight;
                }
                else if (data.event === "blocked_tool") {
                    // Show approval UI for blocked file operations
                    addLog("File operation blocked pending approval: " + data.filepath, "WARNING");
                    ensureAccordion();
                    
                    const approvalDiv = document.createElement("div");
                    approvalDiv.id = "approval-panel-" + Date.now();
                    approvalDiv.className = "step-approval-log";
                    approvalDiv.style.cssText = "margin-top: 10px; padding: 12px; background: rgba(251, 191, 36, 0.1); border: 1px solid orange; border-radius: 6px;";
                    
                    const approvalMsg = document.createElement("div");
                    approvalMsg.innerHTML = "<span style=\"color: orange; font-weight: bold;\">⚠️ " + data.tool + " on '" + data.filepath + "'</span>";
                    approvalDiv.appendChild(approvalMsg);
                    
                    const btnContainer = document.createElement("div");
                    btnContainer.style.cssText = "display: flex; gap: 8px; margin-top: 8px;";
                    
                    const approveBtn = document.createElement("button");
                    approveBtn.textContent = "✔ Approve";
                    approveBtn.style.cssText = "padding: 6px 14px; background: #16a34a; border: none; border-radius: 4px; color: #fff; font-weight: bold; cursor: pointer; pointer-events: auto; z-index: 100;";
                    
                    const rejectBtn = document.createElement("button");
                    rejectBtn.textContent = "✖ Reject";
                    rejectBtn.style.cssText = "padding: 6px 14px; background: #dc2626; border: none; border-radius: 4px; color: #fff; font-weight: bold; cursor: pointer; pointer-events: auto; z-index: 100;";

                    // Capture current accordion refs before the async resume stream replaces them
                    const capturedAccordion = inlineThinkingAccordion;
                    const capturedDetails = inlineThinkingDetails;
                    const capturedBodyContainer = bodyContainer;
                    const capturedAiBubble = aiBubble;
                    const capturedMode = mode;

                    const disableBtns = () => {
                        approveBtn.disabled = true;
                        rejectBtn.disabled = true;
                        approveBtn.style.opacity = '0.5';
                        rejectBtn.style.opacity = '0.5';
                    };

                    approveBtn.onclick = function() {
                        disableBtns();
                        sendApproval(true, data.filepath, data.tool, capturedAccordion, capturedDetails, capturedBodyContainer, capturedAiBubble, capturedMode);
                    };
                    rejectBtn.onclick = function() {
                        disableBtns();
                        sendApproval(false, data.filepath, data.tool, capturedAccordion, capturedDetails, capturedBodyContainer, capturedAiBubble, capturedMode);
                    };
                    
                    btnContainer.appendChild(approveBtn);
                    btnContainer.appendChild(rejectBtn);
                    approvalDiv.appendChild(btnContainer);
                    inlineThinkingDetails.appendChild(approvalDiv);
                    inlineThinkingDetails.scrollTop = inlineThinkingDetails.scrollHeight;
                }
                else if (data.event === "waiting_for_approval") {
                    // Stream has ended but we are waiting — keep UI in generating state
                    // (do NOT call setGenerating(false) — the finally block below will
                    // normally do it, so we must prevent that by marking a flag)
                    addLog("Pipeline paused — awaiting user approval for: " + data.filepath, "WARNING");
                    // Mark on bodyContainer so the finally block knows not to disable
                    bodyContainer.dataset.waitingForApproval = "true";
                    bodyContainer.classList.remove('typing-cursor');
                    if (!accumulatedText) {
                        bodyContainer.innerHTML = `<span style="color: orange; font-style: italic;">⏳ Workflow paused — approve or reject the file operation above to continue.</span>`;
                    }
                }
                else if (data.event === "state_change") {
                    addLog(`State: ${data.state}`, "SYSTEM");
                    if (mode !== "agentic") {
                        addOrUpdateStep("node", data.state);
                    }
                }
                 else if (data.event === "overflow_detected") {
                     // Increment overflow counter
                     sessionOverflows++;
                     if (statOverflows) statOverflows.textContent = sessionOverflows;

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
                     if (inlineThinkingAccordion) {
                         const elapsed = ((Date.now() - thinkingStart) / 1000).toFixed(1);
                         const statusSpan = inlineThinkingAccordion.querySelector('.thinking-status');
                         const spinner = inlineThinkingAccordion.querySelector('.thinking-spinner');
                         if (statusSpan) statusSpan.textContent = `Thought process completed (${elapsed}s)`;
                         if (spinner) {
                             spinner.className = '';
                             spinner.textContent = '✓';
                             spinner.style.color = 'var(--accent-emerald)';
                             spinner.style.fontWeight = 'bold';
                             spinner.style.fontSize = '0.8rem';
                             spinner.style.marginRight = '0.2rem';
                         }
                         // Keep the accordion expanded to show all turns inline
                         // inlineThinkingAccordion.removeAttribute('open');
                         inlineThinkingAccordion = null;
                         inlineThinkingDetails = null;
                     }
                     accumulatedText += data.text;
                     updateStreamUI(false);
                 }
                 else if (data.event === "error") {
                     throw new Error(data.message);
                 }
                 else if (data.event === "done") {
                     updateStreamUI(true);
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
                             <button type="button" class="telemetry-inspect-btn">
                                 INSPECT
                             </button>
                         `;
                         const inspectBtn = footerDiv.querySelector('.telemetry-inspect-btn');
                         if (inspectBtn) {
                             inspectBtn.addEventListener('click', () => {
                                 viewTelemetryDetails(telemetry, budget, query);
                             });
                         }
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

                     // Update new RAG metrics in HUD
                     const calculatedCost = stats.query_cost || (stats.exact_tokens ? `$${((stats.exact_tokens.prompt * 0.000001) + (stats.exact_tokens.completion * 0.000002)).toFixed(6)}` : "$0.00");
                     if (statCost) statCost.textContent = calculatedCost;

                     const calculatedGrounding = stats.grounding_score !== undefined ? stats.grounding_score.toFixed(3) : "0.000";
                     if (statGrounding) statGrounding.textContent = calculatedGrounding;

                     const retrievedCount = stats.retrieved_context ? stats.retrieved_context.length : 0;
                     if (statHits) statHits.textContent = retrievedCount;

                     if (statOverflows) statOverflows.textContent = sessionOverflows;

                     const calculatedTps = stats.exact_tokens ? Math.round(stats.exact_tokens.completion / (latency / 1000.0)) + ' t/s' : '0 t/s';

                     // Render Glass Box Inspector
                     const finalData = {
                         query: query,
                         tps: calculatedTps,
                         query_cost: calculatedCost,
                         search_queries: (stats.instantaneous_latency_ms && stats.instantaneous_latency_ms.search_queries) ? stats.instantaneous_latency_ms.search_queries : [query],
                         hyde_doc: stats.hyde_doc || "N/A",
                         raw_prompt: stats.raw_prompt || "N/A",
                         stats: stats
                     };
                     renderInspector(finalData);

                     // Refresh session list to update message count and timestamp
                     if (typeof ConversationManager !== 'undefined') {
                         ConversationManager.loadSessions();
                     }
                 }
            }
        }
    } catch (err) {
        if (inlineThinkingAccordion) {
            const spinner = inlineThinkingAccordion.querySelector('.thinking-spinner');
            const statusSpan = inlineThinkingAccordion.querySelector('.thinking-status');
            if (spinner) {
                spinner.className = '';
                spinner.textContent = '✗';
                spinner.style.color = 'var(--accent-rose)';
            }
            if (statusSpan) statusSpan.textContent = 'Thought process failed/interrupted';
            inlineThinkingAccordion.removeAttribute('open');
            inlineThinkingAccordion = null;
        }
        if (err.name !== 'AbortError') {
            addLog(`Pipeline generation failure: ${err.message}`, "ERROR");
            bodyContainer.innerHTML = `<span style="color: var(--accent-red); font-weight: bold;">CRITICAL ERROR:</span> ${err.message}`;
            bodyContainer.classList.remove('typing-cursor');
        }
    } finally {
        // If the workflow is paused waiting for user approval, do NOT reset the
        // generating state — the buttons need to remain interactive.
        if (!bodyContainer.dataset.waitingForApproval) {
            AppState.setGenerating(false);
        }
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

// ---- Send Approval Function ----
// Accepts captured UI refs so it can inject resumed-workflow SSE events
// directly into the existing AI bubble / accordion without spawning a new message.
async function sendApproval(approved, filepath, tool,
                            capturedAccordion, capturedDetails,
                            capturedBodyContainer, capturedAiBubble, capturedMode) {
    try {
        const res = await fetch(API_BASE + '/approve_changes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: AppState.sid,
                approve: approved
            })
        });
        const data = await res.json();
        if (!approved) {
            addLog('Changes rejected', 'WARNING');
            if (capturedBodyContainer) {
                capturedBodyContainer.dataset.waitingForApproval = '';
                capturedBodyContainer.innerHTML = `<span style="color: var(--accent-amber); font-style: italic;">❌ File operation rejected. Workflow stopped.</span>`;
            }
            AppState.setGenerating(false);
            return;
        }

        addLog('Changes approved — resuming workflow...', 'SUCCESS');

        // ---- Open a resume SSE stream and feed events into the existing bubble ----
        try {
            const resumeRes = await fetch(`${API_BASE}/resume_stream/${AppState.sid}`, {
                method: 'GET',
                signal: AppState.abortController ? AppState.abortController.signal : undefined
            });
            if (!resumeRes.ok) throw new Error('Resume stream failed: ' + resumeRes.status);

            const reader = resumeRes.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            // Re-show the accordion spinner
            if (capturedAccordion) {
                const spinner = capturedAccordion.querySelector('.thinking-spinner');
                const statusSpan = capturedAccordion.querySelector('.thinking-status');
                if (spinner) { spinner.className = 'thinking-spinner'; spinner.textContent = ''; spinner.style.color = ''; }
                if (statusSpan) statusSpan.textContent = 'Resuming workflow...';
                capturedAccordion.setAttribute('open', '');
            }
            if (capturedBodyContainer) {
                capturedBodyContainer.dataset.waitingForApproval = '';
                capturedBodyContainer.innerHTML = '';
            }

            let resumeAccumulated = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const clean = line.trim();
                    if (!clean.startsWith('data:')) continue;
                    let evt;
                    try { evt = JSON.parse(clean.substring(5).trim()); } catch { continue; }

                    if (evt.event === 'node_start') {
                        addLog('Resumed node: ' + evt.node, 'STATE');
                    } else if (evt.event === 'thought') {
                        if (capturedDetails) {
                            const l = document.createElement('div');
                            l.className = 'step-thought-log';
                            l.innerHTML = `🧠 ${escapeHtml(evt.text)}`;
                            capturedDetails.appendChild(l);
                            capturedDetails.scrollTop = capturedDetails.scrollHeight;
                        }
                    } else if (evt.event === 'observation') {
                        if (capturedDetails) {
                            const l = document.createElement('div');
                            l.className = 'step-obs-log';
                            l.innerHTML = `🔍 ${escapeHtml((evt.output || '').substring(0, 150))}`;
                            capturedDetails.appendChild(l);
                            capturedDetails.scrollTop = capturedDetails.scrollHeight;
                        }
                    } else if (evt.event === 'answer_chunk') {
                        resumeAccumulated += evt.text;
                        if (capturedBodyContainer) {
                            capturedBodyContainer.textContent = resumeAccumulated;
                            chatWindow.scrollTop = chatWindow.scrollHeight;
                        }
                    } else if (evt.event === 'blocked_tool') {
                        // Another approval needed — show buttons again inside captured details
                        if (capturedDetails) {
                            addLog('Another file operation needs approval: ' + evt.filepath, 'WARNING');
                            const aDiv = document.createElement('div');
                            aDiv.style.cssText = 'margin-top:10px;padding:12px;background:rgba(251,191,36,0.1);border:1px solid orange;border-radius:6px;';
                            aDiv.innerHTML = `<span style="color:orange;font-weight:bold;">⚠️ ${escapeHtml(evt.tool)} on '${escapeHtml(evt.filepath)}'</span>`;
                            const bc = document.createElement('div');
                            bc.style.cssText = 'display:flex;gap:8px;margin-top:8px;';
                            const ab = document.createElement('button');
                            ab.textContent = '✔ Approve';
                            ab.style.cssText = 'padding:6px 14px;background:#16a34a;border:none;border-radius:4px;color:#fff;font-weight:bold;cursor:pointer;';
                            const rb = document.createElement('button');
                            rb.textContent = '✖ Reject';
                            rb.style.cssText = 'padding:6px 14px;background:#dc2626;border:none;border-radius:4px;color:#fff;font-weight:bold;cursor:pointer;';
                            const dis = () => { ab.disabled = true; rb.disabled = true; ab.style.opacity='0.5'; rb.style.opacity='0.5'; };
                            ab.onclick = () => { dis(); sendApproval(true, evt.filepath, evt.tool, capturedAccordion, capturedDetails, capturedBodyContainer, capturedAiBubble, capturedMode); };
                            rb.onclick = () => { dis(); sendApproval(false, evt.filepath, evt.tool, capturedAccordion, capturedDetails, capturedBodyContainer, capturedAiBubble, capturedMode); };
                            bc.appendChild(ab); bc.appendChild(rb); aDiv.appendChild(bc);
                            capturedDetails.appendChild(aDiv);
                            capturedDetails.scrollTop = capturedDetails.scrollHeight;
                        }
                        if (capturedBodyContainer) {
                            capturedBodyContainer.dataset.waitingForApproval = 'true';
                            if (!resumeAccumulated) capturedBodyContainer.innerHTML = `<span style="color:orange;font-style:italic;">⏳ Workflow paused — approve or reject the file operation above to continue.</span>`;
                        }
                        return; // wait for next approval; do not call setGenerating(false)
                    } else if (evt.event === 'done') {
                        if (resumeAccumulated && capturedBodyContainer) {
                            try {
                                capturedBodyContainer.innerHTML = typeof marked !== 'undefined' ? marked.parse(resumeAccumulated) : resumeAccumulated;
                            } catch { capturedBodyContainer.textContent = resumeAccumulated; }
                        } else if (capturedBodyContainer && !resumeAccumulated) {
                            capturedBodyContainer.innerHTML = `<span style="color:var(--accent-emerald);">✅ Workflow completed successfully.</span>`;
                        }
                        if (capturedAccordion) {
                            const sp = capturedAccordion.querySelector('.thinking-spinner');
                            const st = capturedAccordion.querySelector('.thinking-status');
                            if (sp) { sp.className=''; sp.textContent='✓'; sp.style.color='var(--accent-emerald)'; sp.style.fontWeight='bold'; }
                            if (st) st.textContent = 'Workflow resumed and completed';
                        }
                        addLog('Resumed workflow completed.', 'SUCCESS');
                    } else if (evt.event === 'error') {
                        addLog('Resume stream error: ' + evt.message, 'ERROR');
                        if (capturedBodyContainer) capturedBodyContainer.innerHTML = `<span style="color:var(--accent-red);">❌ Error during resume: ${escapeHtml(evt.message)}</span>`;
                    }
                }
            }
        } catch (streamErr) {
            addLog('Failed to open resume stream: ' + streamErr.message, 'ERROR');
        } finally {
            AppState.setGenerating(false);
        }
    } catch (e) {
        addLog('Failed to send approval: ' + e.message, 'ERROR');
        AppState.setGenerating(false);
    }
}


// Close modal on click outside content card
telemetryModal.addEventListener('click', (e) => {
    if (e.target === telemetryModal) {
        closeTelemetryModal();
    }
});

// ---- File Upload API ----
document.getElementById('navbar-upload-btn').addEventListener('click', () => document.getElementById('file-in').click());
document.getElementById('file-in').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const feedback = document.getElementById('up-msg');
    feedback.style.display = 'block';
    feedback.textContent = `Uploading: ${file.name}...`;
    feedback.className = 'upload-feedback uploading';
    addLog(`Initiating file injection: ${file.name}`, 'UPLOAD');
    showToast(`Uploading ${file.name}...`, 'info');
    
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
        if (res.ok) { 
            const data = await res.json();
            feedback.textContent = `Indexed successfully!`;
            feedback.className = 'upload-feedback success';
            addLog(`Injection completed: ${data.message}`, "SUCCESS");
            showToast(`✓ ${data.message}`, 'success');
            refreshGlobalStats(); 
        } else {
            const err = await res.json();
            feedback.textContent = `Upload failed.`;
            feedback.className = 'upload-feedback error';
            addLog(`Injection failed: ${err.detail || 'HTTP Error'}`, "ERROR");
            showToast(`✗ Upload failed: ${err.detail || 'Error'}`, 'error');
        }
    } catch (e) { 
        feedback.textContent = `Communication error.`;
        feedback.className = 'upload-feedback error';
        addLog(`Injection communication error: ${e.message}`, "ERROR");
        showToast(`✗ Network error during upload`, 'error');
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
    } catch (e) {}
}

// ---- Thread Restoration (Database history load) ----
async function loadHistory() {
    try {
        addLog("Restoring session conversation thread...", "SYSTEM");
        const res = await fetch(`${API_BASE}/history/${sid}`);
        if (!res.ok) {
            removeSkeleton();
            return;
        }
        
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
        } else {
            removeSkeleton();
        }
    } catch (e) {
        console.error(e);
        addLog("Failed to restore session history from persistent database.", "WARNING");
        removeSkeleton();
    }
}

function removeSkeleton() {
    const skeleton = document.getElementById('chat-skeleton');
    if (skeleton) skeleton.remove();
}

// Initializers
let statsInterval = null;

function startStatsPolling() {
    if (!statsInterval) {
        statsInterval = setInterval(refreshGlobalStats, 15000);
    }
}

function stopStatsPolling() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopStatsPolling();
        addLog("Tab hidden. Suspending stats polling.", "SYSTEM");
    } else {
        startStatsPolling();
        refreshGlobalStats();
        addLog("Tab visible. Resumed stats polling.", "SYSTEM");
    }
});

startStatsPolling();
refreshGlobalStats();
loadHistory();
AppState.updateContextLimit(parseInt(contextLimitSlider.value));

// ---- HUD Panel Toggle ----
const hudToggleBtn = document.getElementById('hud-toggle-btn');
const mainGrid = document.querySelector('.main-grid');

if (hudToggleBtn && mainGrid) {
    // Helper to update button label text without losing the SVG icon
    const updateButtonLabel = (visibleOrCollapsed, textVal) => {
        hudToggleBtn.title = textVal;
    };

    // Start collapsed on all viewports by default
    mainGrid.classList.add('hud-collapsed');
    updateButtonLabel(true, 'Show HUD');
    hudToggleBtn.classList.add('hud-hidden-state');

    hudToggleBtn.addEventListener('click', () => {
        const width = window.innerWidth;
        if (width <= 1200) {
            // On small viewports, toggle hud-visible class
            mainGrid.classList.toggle('hud-visible');
            const isVisible = mainGrid.classList.contains('hud-visible');
            hudToggleBtn.classList.toggle('hud-hidden-state', !isVisible);
            updateButtonLabel(isVisible, isVisible ? 'Hide HUD' : 'Show HUD');
            addLog(isVisible ? "System HUD visible in responsive overlay." : "System HUD hidden in responsive overlay.", "SYSTEM");
        } else {
            // On large viewports, toggle hud-collapsed class
            mainGrid.classList.toggle('hud-collapsed');
            const collapsed = mainGrid.classList.contains('hud-collapsed');
            hudToggleBtn.classList.toggle('hud-hidden-state', collapsed);
            updateButtonLabel(collapsed, collapsed ? 'Show HUD' : 'Collapse HUD');
            addLog(collapsed ? "System HUD collapsed." : "System HUD expanded.", "SYSTEM");
        }
    });

    // Handle window resize dynamically to sync classes
    window.addEventListener('resize', () => {
        const width = window.innerWidth;
        if (width > 1200) {
            mainGrid.classList.remove('hud-visible');
            const collapsed = mainGrid.classList.contains('hud-collapsed');
            updateButtonLabel(collapsed, collapsed ? 'Show HUD' : 'Collapse HUD');
            hudToggleBtn.classList.toggle('hud-hidden-state', collapsed);
        } else {
            const isVisible = mainGrid.classList.contains('hud-visible');
            updateButtonLabel(isVisible, isVisible ? 'Hide HUD' : 'Show HUD');
            hudToggleBtn.classList.toggle('hud-hidden-state', !isVisible);
        }
    });
}

// ---- Scroll-to-Bottom FAB ----
if (scrollBottomFab && chatWindow) {
    chatWindow.addEventListener('scroll', () => {
        const scrollFromBottom = chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight;
        scrollBottomFab.classList.toggle('visible', scrollFromBottom > 150);
    });

    scrollBottomFab.addEventListener('click', () => {
        chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
    });
}

// ---- Drag & Drop File Upload ----
const fileDragOverlay = document.getElementById('file-drag-overlay');
const fileInput = document.getElementById('file-in');

if (fileDragOverlay) {
    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter++;
        fileDragOverlay.classList.add('drag-active');
    });

    window.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter--;
        if (dragCounter === 0) {
            fileDragOverlay.classList.remove('drag-active');
        }
    });

    window.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    window.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter = 0;
        fileDragOverlay.classList.remove('drag-active');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const validTypes = ['.pdf', '.txt'];
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!validTypes.includes(ext)) {
                showToast(`Unsupported file type: ${ext}. Use .pdf or .txt`, 'error');
                addLog(`Rejected drag-drop: unsupported file type ${ext}`, 'WARNING');
                return;
            }
            // Trigger the same upload flow as the file input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

// ---- Tab Switching Event Listeners (Developer Console) ----
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        const parent = btn.closest('.dev-tabs');
        if (!parent) return;

        parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        parent.querySelectorAll('.tab-content-panel').forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPanel = document.getElementById(tabId);
        if (targetPanel) {
            targetPanel.classList.add('active');
            // Auto-scroll any log containers inside to bottom when tab becomes visible
            targetPanel.querySelectorAll('.log-container').forEach(el => {
                el.scrollTop = el.scrollHeight;
            });
        }
    });
});

// ---- Textarea Auto-Resizer ----
if (input) {
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// ---- Right Drawer Auto-Close (Click Outside) ----
document.addEventListener('click', (e) => {
    const sidebarRight = document.querySelector('.sidebar-right');
    const hudToggleBtn = document.getElementById('hud-toggle-btn');
    if (!sidebarRight || !hudToggleBtn || !mainGrid) return;

    // Check if sidebar is currently visible/expanded
    const isResponsiveVisible = mainGrid.classList.contains('hud-visible');
    const isDesktopExpanded = !mainGrid.classList.contains('hud-collapsed');

    // If it's not open in any way, nothing to do
    if (!isResponsiveVisible && !isDesktopExpanded) return;

    // Check if the click target is inside the sidebar, the toggle button, or the telemetry modal
    const clickedInsideSidebar = sidebarRight.contains(e.target);
    const clickedToggleBtn = hudToggleBtn.contains(e.target);
    const clickedModal = e.target.closest('#telemetry-modal') || e.target.closest('.telemetry-inspect-btn');

    if (!clickedInsideSidebar && !clickedToggleBtn && !clickedModal) {
        if (window.innerWidth <= 1200) {
            mainGrid.classList.remove('hud-visible');
            hudToggleBtn.classList.add('hud-hidden-state');
            hudToggleBtn.title = 'Show HUD';
            addLog("System HUD hidden by clicking outside.", "SYSTEM");
        } else {
            mainGrid.classList.add('hud-collapsed');
            hudToggleBtn.classList.add('hud-hidden-state');
            hudToggleBtn.title = 'Show HUD';
            addLog("System HUD collapsed by clicking outside.", "SYSTEM");
        }
    }
});


// ================================================================
// CONVERSATION MANAGER
// ================================================================

const ConversationManager = (() => {
    const convList     = document.getElementById('conv-list');
    const convEmpty    = document.getElementById('conv-list-empty');
    const newChatBtn   = document.getElementById('new-chat-btn');

    // ---- helpers ----
    function relativeTime(dateStr) {
        const now  = Date.now();
        const then = new Date(dateStr + (dateStr.endsWith('Z') ? '' : 'Z')).getTime();
        const diff = Math.floor((now - then) / 1000);
        if (diff < 60)       return 'just now';
        if (diff < 3600)     return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400)    return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }

    function setActiveSid(newSid) {
        sid = newSid;
        localStorage.setItem('station_sid', newSid);
        if (sessionTag) sessionTag.textContent = `SID: ${newSid}`;
    }

    // ---- render ----
    function renderList(sessions) {
        if (!convList) return;
        // Clear dynamic items (keep empty placeholder)
        convList.querySelectorAll('.conv-item').forEach(el => el.remove());

        if (!sessions || sessions.length === 0) {
            if (convEmpty) convEmpty.style.display = 'block';
            return;
        }
        if (convEmpty) convEmpty.style.display = 'none';

        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = 'conv-item' + (session.session_id === sid ? ' active' : '');
            item.dataset.sid = session.session_id;

            item.innerHTML = `
                <div class="conv-item-text">
                    <div class="conv-title" title="${session.title}">${session.title}</div>
                    <div class="conv-meta">${session.message_count} msg &middot; ${relativeTime(session.updated_at)}</div>
                </div>
                <div class="conv-actions">
                    <button class="conv-action-btn rename-btn" title="Rename">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    <button class="conv-action-btn danger delete-btn" title="Delete">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                            <path d="M10 11v6"/><path d="M14 11v6"/>
                            <path d="M9 6V4h6v2"/>
                        </svg>
                    </button>
                </div>
            `;

            convList.appendChild(item);
        });
    }

    // ---- load sessions ----
    async function loadSessions() {
        try {
            const res = await fetch(`${API_BASE}/sessions`);
            if (!res.ok) return;
            const sessions = await res.json();
            renderList(sessions);
        } catch (e) {
            console.warn('Could not load sessions:', e);
        }
    }

    // ---- switch session ----
    async function switchSession(newSid) {
        setActiveSid(newSid);
        // Highlight active item
        document.querySelectorAll('.conv-item').forEach(el => {
            el.classList.toggle('active', el.dataset.sid === newSid);
        });
        // Clear chat and reload history
        if (chatWindow) {
            chatWindow.innerHTML = '<div id="chat-skeleton" class="chat-skeleton" style="display:none"></div>';
        }
        
        // Reset stats
        sessionOverflows = 0;
        if (statOverflows) statOverflows.textContent = '0';
        if (statCost) statCost.textContent = '$0.00';
        if (statGrounding) statGrounding.textContent = '0.000';
        if (statHits) statHits.textContent = '0';
        if (statT) statT.textContent = '0ms';
        if (statM) statM.textContent = '0';

        try {
            const res  = await fetch(`${API_BASE}/history/${newSid}`);
            const hist = await res.json();
            hist.forEach(entry => {
                if (entry.role === 'user') {
                    addMsg(entry.text, 'user');
                } else if (entry.role === 'assistant') {
                    addMsg(entry.text, 'ai', entry.telemetry);
                }
            });
            if (chatWindow) chatWindow.scrollTop = chatWindow.scrollHeight;
        } catch (e) {
            console.warn('Could not load history:', e);
        }
    }

    // ---- create new session ----
    async function createNewChat() {
        try {
            const res  = await fetch(`${API_BASE}/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: 'New Chat' })
            });
            const data = await res.json();
            setActiveSid(data.session_id);
            // Clear chat window
            if (chatWindow) {
                chatWindow.innerHTML = '<div id="chat-skeleton" class="chat-skeleton" style="display:none"></div>';
            }
            
            // Reset stats
            sessionOverflows = 0;
            if (statOverflows) statOverflows.textContent = '0';
            if (statCost) statCost.textContent = '$0.00';
            if (statGrounding) statGrounding.textContent = '0.000';
            if (statHits) statHits.textContent = '0';
            if (statT) statT.textContent = '0ms';
            if (statM) statM.textContent = '0';

            await loadSessions();
            // Focus input
            if (input) input.focus();
        } catch (e) {
            console.error('Failed to create session:', e);
        }
    }

    // ---- auto-title from first message ----
    async function autoTitleFromFirstMessage(question) {
        // Only title if current title is generic
        const activeItem = document.querySelector(`.conv-item[data-sid="${sid}"]`);
        if (!activeItem) return;
        const titleEl = activeItem.querySelector('.conv-title');
        if (!titleEl || (titleEl.textContent !== 'New Chat' && titleEl.textContent !== sid)) return;
        const title = question.trim().slice(0, 45) + (question.length > 45 ? '...' : '');
        try {
            await fetch(`${API_BASE}/sessions/${sid}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title })
            });
            titleEl.textContent = title;
            titleEl.title = title;
        } catch (e) { /* silent */ }
    }

    // ---- inline rename ----
    function startInlineRename(item, session) {
        const titleEl = item.querySelector('.conv-title');
        const currentTitle = titleEl.textContent;
        const inp = document.createElement('input');
        inp.className = 'conv-rename-input';
        inp.value = currentTitle;
        titleEl.replaceWith(inp);
        inp.focus();
        inp.select();

        const commit = async () => {
            const newTitle = inp.value.trim() || currentTitle;
            const span = document.createElement('div');
            span.className = 'conv-title';
            span.textContent = newTitle;
            span.title = newTitle;
            inp.replaceWith(span);
            if (newTitle !== currentTitle) {
                try {
                    await fetch(`${API_BASE}/sessions/${session.session_id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title: newTitle })
                    });
                } catch (e) { console.error('Rename failed:', e); }
            }
        };
        inp.addEventListener('blur', commit);
        inp.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); inp.blur(); }
            if (e.key === 'Escape') { inp.value = currentTitle; inp.blur(); }
        });
    }

    // ---- delete session ----
    async function deleteSession(delSid, item) {
        if (!confirm(`Delete this conversation? This cannot be undone.`)) return;
        try {
            const res = await fetch(`${API_BASE}/sessions/${delSid}`, { method: 'DELETE' });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server responded with status ${res.status}`);
            }
            item.remove();
            // If we deleted the active session, create a new one
            if (delSid === sid) {
                await createNewChat();
            }
            // Show empty state if no items left
            if (!convList.querySelector('.conv-item')) {
                if (convEmpty) convEmpty.style.display = 'block';
            }
            showToast("Conversation deleted successfully", "success");
        } catch (e) {
            console.error('Delete failed:', e);
            showToast(`✗ Delete failed: ${e.message}`, "error");
        }
    }

    // ---- wire up new chat button ----
    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewChat);
    }

    // ---- event delegation for conversation items ----
    if (convList) {
        convList.addEventListener('click', (e) => {
            const renameBtn = e.target.closest('.rename-btn');
            const deleteBtn = e.target.closest('.delete-btn');
            const item = e.target.closest('.conv-item');
            
            if (!item) return;
            const itemSid = item.dataset.sid;
            
            if (renameBtn) {
                e.stopPropagation();
                const titleVal = item.querySelector('.conv-title').textContent;
                startInlineRename(item, { session_id: itemSid, title: titleVal });
            } else if (deleteBtn) {
                e.stopPropagation();
                deleteSession(itemSid, item);
            } else {
                if (itemSid !== sid) {
                    switchSession(itemSid);
                }
            }
        });
    }

    // ---- init: ensure current session is registered ----
    async function init() {
        // Register current session so it appears in the list
        try {
            await fetch(`${API_BASE}/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid, title: 'New Chat' })
            });
        } catch (e) { /* already exists — that's fine */ }
        await loadSessions();
    }

    init();

    return { loadSessions, autoTitleFromFirstMessage };
})();

// Title auto-setting is handled directly inside the form submit listener.
