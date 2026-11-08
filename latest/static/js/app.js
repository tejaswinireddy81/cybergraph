let adminToken = localStorage.getItem('cybergraph_admin_token') || '';
let csrfToken = '';

async function fetchCsrfToken() {
    try {
        const res = await fetch('/api/auth/csrf');
        if (res.ok) {
            const data = await res.json();
            if (data.csrf_token) {
                csrfToken = data.csrf_token;
            }
        }
    } catch(e) {
        console.warn('Could not fetch CSRF token:', e);
    }
}
fetchCsrfToken();
        if (adminToken) {
            localStorage.setItem('cybergraph_admin_token', adminToken);
        }

        function promptAdminToken() {
            const current = localStorage.getItem('cybergraph_admin_token') || '';
            const t = prompt('Enter CyberGraph Admin Access Token (from server startup console):', current);
            if (t !== null) {
                localStorage.setItem('cybergraph_admin_token', t.trim());
                adminToken = t.trim();
                alert('[CYBERGRAPH AUTH] Admin token saved successfully.');
                updateDeviceAccessSection();
            }
        }

        async function postAdmin(url, bodyObj = {}) {
            let token = localStorage.getItem('cybergraph_admin_token') || adminToken;
            if (!csrfToken) {
                await fetchCsrfToken();
            }
            let res;
            const headers = {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken || '',
            };
            if (token) {
                headers['X-Admin-Token'] = token;
                headers['Authorization'] = 'Bearer ' + token;
            }

            try {
                res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(bodyObj)
                });
            } catch(e) {
                throw e;
            }

            if (res.status === 403) {
                const promptToken = prompt('[SECURITY 403 FORBIDDEN]\nEnter Admin Access Token to authenticate this action:');
                if (promptToken) {
                    localStorage.setItem('cybergraph_admin_token', promptToken.trim());
                    adminToken = promptToken.trim();
                    headers['X-Admin-Token'] = adminToken;
                    headers['Authorization'] = 'Bearer ' + adminToken;
                    res = await fetch(url, {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify(bodyObj)
                    });
                }
            }
            return res;
        }

        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        const sparkCanvas = document.getElementById('sparkline-canvas');
        const sctx = sparkCanvas.getContext('2d');

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            sparkCanvas.width = sparkCanvas.parentElement.clientWidth;
            sparkCanvas.height = 45;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        let nodes = [], edges = [], particles = [];
        let nodePosMap = {};
        let riskHistory = [0.05];

        function ipToNum(ip) {
            const clean = String(ip).replace(/^(EXT|GW|PENDING):/, '');
            const parts = clean.split('.').map(Number);
            if (parts.length === 4 && !parts.some(isNaN)) {
                return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3];
            }
            return clean;
        }

        function sortNodes(nodeList) {
            return [...nodeList].sort((a, b) => {
                const numA = ipToNum(a.label);
                const numB = ipToNum(b.label);
                if (typeof numA === 'number' && typeof numB === 'number') {
                    return numA - numB;
                }
                return String(a.label).localeCompare(String(b.label));
            });
        }

        let graphScale = 1.0;
        let graphOffsetX = 0;
        let graphOffsetY = 0;
        let isPanning = false;
        let panStartX = 0;
        let panStartY = 0;
        let hasMoved = false;

        function zoomGraph(factor) {
            graphScale = Math.max(0.4, Math.min(3.0, graphScale * factor));
        }

        function resetGraphView() {
            graphScale = 1.0;
            graphOffsetX = 0;
            graphOffsetY = 0;
        }

        
        // ─── DASHBOARD CHARTS & ANALYTICS STATE ───
        const alertsTimeCanvas = document.getElementById('chart-alerts-time');
        const actx = alertsTimeCanvas ? alertsTimeCanvas.getContext('2d') : null;
        const topAttackersCanvas = document.getElementById('chart-top-attackers');
        const tactx = topAttackersCanvas ? topAttackersCanvas.getContext('2d') : null;

        const alertsTimeHistory = [];
        const MAX_TIME_SLOTS = 15;
        let lastL1 = 0, lastL2 = 0, lastL3 = 0;

        function resizeAnalyticsCanvases() {
            if (alertsTimeCanvas && alertsTimeCanvas.parentElement) {
                alertsTimeCanvas.width = alertsTimeCanvas.parentElement.clientWidth || 300;
                alertsTimeCanvas.height = 160;
            }
            if (topAttackersCanvas && topAttackersCanvas.parentElement) {
                topAttackersCanvas.width = topAttackersCanvas.parentElement.clientWidth || 300;
                topAttackersCanvas.height = 160;
            }
        }
        window.addEventListener('resize', resizeAnalyticsCanvases);
        setTimeout(resizeAnalyticsCanvases, 100);

        function drawAlertsOverTimeChart(data) {
            if (!actx || !alertsTimeCanvas) return;
            const isLight = document.body.classList.contains('light-theme');
            const w = alertsTimeCanvas.width;
            const h = alertsTimeCanvas.height;

            actx.clearRect(0, 0, w, h);

            // Compute delta alerts
            const curL1 = data?.stats?.level1_normal || 0;
            const curL2 = data?.stats?.level2_suspicious || 0;
            const curL3 = data?.stats?.level3_critical || 0;

            const deltaL1 = Math.max(0, curL1 - lastL1);
            const deltaL2 = Math.max(0, curL2 - lastL2);
            const deltaL3 = Math.max(0, curL3 - lastL3);

            lastL1 = curL1; lastL2 = curL2; lastL3 = curL3;

            alertsTimeHistory.push({
                l1: deltaL1,
                l2: deltaL2,
                l3: deltaL3,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            });

            if (alertsTimeHistory.length > MAX_TIME_SLOTS) {
                alertsTimeHistory.shift();
            }

            // Draw grid & axis
            actx.strokeStyle = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)';
            actx.lineWidth = 1;
            for (let y = 30; y < h - 25; y += 30) {
                actx.beginPath();
                actx.moveTo(35, y);
                actx.lineTo(w - 15, y);
                actx.stroke();
            }

            const maxVal = Math.max(5, ...alertsTimeHistory.map(d => d.l1 + d.l2 + d.l3));
            const chartH = h - 60;
            const slotW = (w - 55) / MAX_TIME_SLOTS;

            alertsTimeHistory.forEach((slot, i) => {
                const x = 40 + i * slotW;
                const barW = Math.max(6, slotW - 6);

                const h1 = (slot.l1 / maxVal) * chartH;
                const h2 = (slot.l2 / maxVal) * chartH;
                const h3 = (slot.l3 / maxVal) * chartH;

                let curY = h - 30;

                // L1 Normal (Green)
                if (h1 > 0) {
                    actx.fillStyle = '#00ff9d';
                    actx.fillRect(x, curY - h1, barW, h1);
                    curY -= h1;
                }
                // L2 Suspicious (Amber)
                if (h2 > 0) {
                    actx.fillStyle = '#ffb700';
                    actx.fillRect(x, curY - h2, barW, h2);
                    curY -= h2;
                }
                // L3 Critical (Crimson)
                if (h3 > 0) {
                    actx.fillStyle = '#ff0055';
                    actx.fillRect(x, curY - h3, barW, h3);
                }
            });

            // Axis labels & legend
            actx.fillStyle = isLight ? '#475569' : '#94a3b8';
            actx.font = '9px JetBrains Mono, monospace';
            actx.fillText('Alerts', 4, 15);
            actx.fillText(`Peak: ${maxVal}`, w - 75, 15);

            const sumEl = document.getElementById('alerts-time-summary');
            if (sumEl) sumEl.innerText = `L3: ${curL3} | L2: ${curL2} | L1: ${curL1}`;
        }

        function drawTopAttackersChart(evals) {
            if (!tactx || !topAttackersCanvas) return;
            const isLight = document.body.classList.contains('light-theme');
            const w = topAttackersCanvas.width;
            const h = topAttackersCanvas.height;

            tactx.clearRect(0, 0, w, h);

            // Aggregate evals by source host
            const counts = {};
            const maxRisks = {};
            (evals || []).forEach(ev => {
                const src = ev.source_host || 'Unknown';
                if (src.includes('127.0.0.1') || src.includes('localhost')) return;
                counts[src] = (counts[src] || 0) + 1;
                maxRisks[src] = Math.max(maxRisks[src] || 0, Number(ev.risk_score || 0));
            });

            const top = Object.entries(counts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4);

            if (top.length === 0) {
                tactx.fillStyle = isLight ? '#059669' : '#00ff9d';
                tactx.font = '10px JetBrains Mono, monospace';
                tactx.fillText('✓ Network Clean — No Threat Actors', 20, h / 2);
                return;
            }

            const maxC = top[0][1] || 1;
            const rowH = 26;

            top.forEach(([ip, cnt], i) => {
                const y = 20 + i * rowH;
                const barMaxW = w - 140;
                const barW = Math.max(12, (cnt / maxC) * barMaxW);
                const risk = maxRisks[ip] || 0;
                const barColor = risk >= 0.75 ? '#ff0055' : risk >= 0.35 ? '#ffb700' : '#00f0ff';

                // IP Label
                tactx.fillStyle = isLight ? '#0f172a' : '#f8fafc';
                tactx.font = '10px JetBrains Mono, monospace';
                tactx.fillText(ip.replace('HOST:', ''), 10, y + 12);

                // Bar
                tactx.fillStyle = barColor;
                tactx.fillRect(95, y + 2, barW, 12);

                // Count badge
                tactx.fillStyle = isLight ? '#64748b' : '#94a3b8';
                tactx.fillText(`${cnt} hits (${risk.toFixed(2)})`, 105 + barW, y + 12);
            });

            const sumEl = document.getElementById('top-attackers-summary');
            if (sumEl) sumEl.innerText = `${top.length} Active Actor(s)`;
        }

        const MITRE_MATRIX_DEF = [
            { id: 'T1046', name: 'Network Service Discovery', tactic: 'Discovery' },
            { id: 'T1110', name: 'Brute Force Auth', tactic: 'Credential Access' },
            { id: 'T1498', name: 'Network Denial of Service', tactic: 'Impact' },
            { id: 'T1071', name: 'Standard App Protocol', tactic: 'Command & Control' },
            { id: 'T1190', name: 'Exploit Public-Facing App', tactic: 'Initial Access' },
            { id: 'T1021', name: 'Remote Services / SMB', tactic: 'Lateral Movement' },
        ];

        function renderMitreHeatmap(evals) {
            const container = document.getElementById('mitre-heatmap-container');
            if (!container) return;

            const counts = {};
            (evals || []).forEach(ev => {
                (ev.mitre_mappings || []).forEach(m => {
                    const tid = m.technique_id;
                    if (tid) counts[tid] = (counts[tid] || 0) + 1;
                });
            });

            container.innerHTML = MITRE_MATRIX_DEF.map(tech => {
                const hits = counts[tech.id] || 0;
                let bg = 'rgba(15, 23, 42, 0.6)';
                let border = 'rgba(51, 65, 85, 0.5)';
                let badgeColor = 'var(--text-muted)';

                if (hits > 0 && hits <= 2) {
                    bg = 'rgba(255, 183, 0, 0.12)';
                    border = 'var(--warning-amber)';
                    badgeColor = 'var(--warning-amber)';
                } else if (hits > 2) {
                    bg = 'rgba(255, 0, 85, 0.2)';
                    border = 'var(--critical-crimson)';
                    badgeColor = 'var(--critical-crimson)';
                }

                return `
                    <div style="background:${bg}; border:1px solid ${border}; border-radius:6px; padding:6px 8px; display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                <strong style="color:var(--mitre-blue); font-size:10px;">${tech.id}</strong>
                                <span style="background:${badgeColor}; color:#000; font-size:8px; font-weight:700; padding:1px 4px; border-radius:3px;">${hits} hits</span>
                            </div>
                            <div style="color:var(--text-main); font-size:9px; line-height:1.2;">${tech.name}</div>
                        </div>
                        <div style="color:var(--text-muted); font-size:8px; margin-top:4px; text-transform:uppercase;">${tech.tactic}</div>
                    </div>
                `;
            }).join('');
        }

        function updateAllCharts(data) {
            drawAlertsOverTimeChart(data);
            drawTopAttackersChart(data?.evals || currentEvals || []);
            renderMitreHeatmap(data?.evals || currentEvals || []);
        }

        function drawSparkline() {
            sctx.clearRect(0, 0, sparkCanvas.width, sparkCanvas.height);
            if (riskHistory.length < 2) return;

            sctx.beginPath();
            const step = sparkCanvas.width / (riskHistory.length - 1);
            riskHistory.forEach((r, i) => {
                const x = i * step;
                const y = sparkCanvas.height - (r * sparkCanvas.height * 0.8 + 4);
                if (i === 0) sctx.moveTo(x, y);
                else sctx.lineTo(x, y);
            });
            sctx.strokeStyle = '#00f0ff';
            sctx.lineWidth = 2;
            sctx.stroke();
        }

        function drawStage() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(graphOffsetX, graphOffsetY);
            ctx.scale(graphScale, graphScale);

            const marginX = Math.max(130, Math.min(200, canvas.width * 0.28));
            const hostX = marginX;
            const serviceX = canvas.width - marginX;

            // Bipartite Column Guides
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(hostX, 0); ctx.lineTo(hostX, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(serviceX, 0); ctx.lineTo(serviceX, canvas.height); ctx.stroke();
            ctx.setLineDash([]);

            // Column Header Titles
            ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
            ctx.font = '700 9px "Space Grotesk", sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('HOST / LAN ENDPOINTS', hostX - 16, 22);
            ctx.textAlign = 'left';
            ctx.fillText('SERVICES / TARGET PORTS', serviceX + 16, 22);

            // Draw Restrained Curved / Straight Edges
            edges.forEach(e => {
                let edgeColor = 'rgba(0, 255, 157, 0.35)';
                let lineWidth = 1.5;

                if (e.color === 'critical' || e.risk >= 0.75) {
                    edgeColor = 'rgba(255, 0, 85, 0.85)';
                    lineWidth = 2.5;
                } else if (e.color === 'suspicious' || e.risk >= 0.35) {
                    edgeColor = 'rgba(255, 183, 0, 0.65)';
                    lineWidth = 2;
                }

                // Underlay glow for critical edges
                if (e.risk >= 0.75 || e.color === 'critical') {
                    ctx.beginPath();
                    ctx.moveTo(e.x1, e.y1);
                    ctx.quadraticCurveTo(e.controlX, e.controlY, e.x2, e.y2);
                    ctx.strokeStyle = 'rgba(255, 0, 85, 0.2)';
                    ctx.lineWidth = lineWidth + 4;
                    ctx.stroke();
                }

                ctx.beginPath();
                ctx.moveTo(e.x1, e.y1);
                ctx.quadraticCurveTo(e.controlX, e.controlY, e.x2, e.y2);
                ctx.strokeStyle = edgeColor;
                ctx.lineWidth = lineWidth;
                ctx.stroke();
            });

            // Flow Particles along Bezier / Straight Curves
            particles.forEach(p => {
                p.progress += p.speed;
                if (p.progress >= 1.0) p.progress = 0.0;
                const t = p.progress;
                const invT = 1 - t;
                const currX = invT * invT * p.x1 + 2 * invT * t * p.controlX + t * t * p.x2;
                const currY = invT * invT * p.y1 + 2 * invT * t * p.controlY + t * t * p.y2;

                ctx.beginPath();
                ctx.arc(currX, currY, p.risk >= 0.75 ? 4.5 : 3, 0, Math.PI * 2);
                let pColor = '#00f0ff';
                if (p.color === 'critical' || p.risk >= 0.75) pColor = '#ff0055';
                else if (p.color === 'suspicious' || p.risk >= 0.35) pColor = '#ffb700';
                ctx.fillStyle = pColor;
                ctx.shadowColor = pColor;
                ctx.shadowBlur = 6;
                ctx.fill();
                ctx.shadowBlur = 0;
            });

            // Draw Nodes
            nodes.forEach(n => {
                const isCritical = (n.color === 'critical' || n.threat_level === 'CRITICAL');
                const isSuspicious = (n.color === 'suspicious' || n.threat_level === 'SUSPICIOUS' || n.type === 'PENDING');

                // Node radius: HOST=8, GATEWAY=8, PENDING=8, EXTERNAL=7, SERVICE=6
                const radius = (n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'PENDING') ? 8 : (n.type === 'EXTERNAL' ? 7 : 6);

                // Pulsing / glowing outer ring for Critical or Suspicious HOST / PENDING nodes
                if (n.type === 'HOST' || n.type === 'PENDING') {
                    if (isCritical) {
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, radius + 5, 0, Math.PI * 2);
                        ctx.strokeStyle = 'rgba(255, 0, 85, 0.8)';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    } else if (isSuspicious) {
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, radius + 4, 0, Math.PI * 2);
                        ctx.strokeStyle = 'rgba(255, 183, 0, 0.6)';
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                    }
                }

                let nodeColor;
                if (n.type === 'SERVICE') {
                    nodeColor = '#b026ff';
                } else if (n.type === 'GATEWAY') {
                    nodeColor = '#818cf8';
                } else if (n.type === 'PENDING') {
                    nodeColor = '#ffb700'; // Amber for pending
                } else if (n.type === 'EXTERNAL') {
                    if (isCritical) nodeColor = '#ff0055';
                    else if (isSuspicious) nodeColor = '#ffb700';
                    else nodeColor = '#ff8c00';
                } else {
                    // HOST
                    if (isCritical) nodeColor = '#ff0055';
                    else if (isSuspicious) nodeColor = '#ffb700';
                    else nodeColor = '#00f0ff';
                }

                if (n.type === 'GATEWAY') {
                    // Distinct diamond/square icon for Gateway
                    ctx.beginPath();
                    ctx.rect(n.x - radius, n.y - radius, radius * 2, radius * 2);
                    ctx.fillStyle = nodeColor;
                    ctx.shadowColor = nodeColor;
                    ctx.shadowBlur = 8;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    ctx.strokeStyle = '#c7d2fe';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                    ctx.fillStyle = nodeColor;
                    ctx.shadowColor = nodeColor;
                    ctx.shadowBlur = (isCritical || isSuspicious || n.type === 'HOST') ? 10 : 4;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }

                // Labels
                ctx.font = '600 11px "JetBrains Mono", monospace';
                if (n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'PENDING' || n.type === 'EXTERNAL') {
                    ctx.textAlign = 'right';
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#070a11';

                    let displayLabel = n.label;
                    if (n.type === 'GATEWAY') {
                        displayLabel = n.label.replace(/^GW:/, '') + ' [Gateway]';
                    } else if (n.type === 'PENDING') {
                        displayLabel = n.label.replace(/^PENDING:/, '') + ' [Pending]';
                    } else if (n.type === 'EXTERNAL') {
                        displayLabel = n.label.replace(/^EXT:/, '');
                    }

                    ctx.strokeText(displayLabel, n.x - 16, n.y + 4);

                    let textColor = '#f8fafc';
                    if (n.type === 'GATEWAY') textColor = '#c7d2fe';
                    else if (n.type === 'PENDING') textColor = '#ffe066';
                    else if (isCritical) textColor = '#ff4d7d';
                    else if (isSuspicious) textColor = '#ffe066';

                    ctx.fillStyle = textColor;
                    ctx.fillText(displayLabel, n.x - 16, n.y + 4);
                } else {
                    ctx.textAlign = 'left';
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#070a11';
                    ctx.strokeText(n.label, n.x + 16, n.y + 4);
                    ctx.fillStyle = '#e2e8f0';
                    ctx.fillText(n.label, n.x + 16, n.y + 4);
                }
            });

            ctx.restore();
            requestAnimationFrame(drawStage);
        }
        requestAnimationFrame(drawStage);

        canvas.addEventListener('mousedown', (e) => {
            isPanning = true;
            hasMoved = false;
            panStartX = e.clientX - graphOffsetX;
            panStartY = e.clientY - graphOffsetY;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isPanning) return;
            hasMoved = true;
            graphOffsetX = e.clientX - panStartX;
            graphOffsetY = e.clientY - panStartY;
        });

        window.addEventListener('mouseup', () => {
            isPanning = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.87;
            zoomGraph(factor);
        }, { passive: false });

        let currentInspectedIp = null;

        canvas.addEventListener('click', (e) => {
            if (hasMoved) return;
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left - graphOffsetX) / graphScale;
            const mouseY = (e.clientY - rect.top - graphOffsetY) / graphScale;

            nodes.forEach(n => {
                const dist = Math.hypot(n.x - mouseX, n.y - mouseY);
                if (dist <= 22) {
                    let displayAddr = n.label;
                    if (n.type === 'GATEWAY') displayAddr = n.label.replace(/^GW:/, '') + ' (Default Gateway)';
                    else if (n.type === 'PENDING') displayAddr = n.label.replace(/^PENDING:/, '') + ' (Pending Authorization)';
                    else if (n.type === 'EXTERNAL') displayAddr = n.label.replace(/^EXT:/, '');

                    currentInspectedIp = displayAddr.split(' ')[0].replace(/[^0-9a-zA-Z.:_-]/g, '');
                    document.getElementById('insp-title').innerText = `INSPECTING: ${displayAddr}`;
                    document.getElementById('insp-type').innerText = n.type;
                    document.getElementById('insp-addr').innerText = displayAddr;
                    document.getElementById('insp-level').innerText = n.threat_level || 'NORMAL';
                    const inspStatus = document.getElementById('insp-status');
                    if (inspStatus) inspStatus.innerText = 'MONITORED & SECURED';
                    document.getElementById('inspector-modal').style.display = 'block';
                }
            });
        });

        function closeInspector() {
            document.getElementById('inspector-modal').style.display = 'none';
        }

        function inspectorBlockCurrentNode() {
            if (currentInspectedIp) {
                executeRealBlock(currentInspectedIp);
            }
        }

        function inspectorUnblockCurrentNode() {
            if (currentInspectedIp) {
                unblockIp(currentInspectedIp);
            }
        }

        function escapeHtml(str) {
            if (str === null || str === undefined) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function sanitizeIp(str) {
            if (!str) return '';
            return String(str).replace(/[^0-9a-zA-Z.:_-]/g, '');
        }

        let pendingBlockIp = null;

        function executeRealBlock(targetIp) {
            const clean = sanitizeIp(targetIp);
            if (!clean) return;
            pendingBlockIp = clean;
            const modal = document.getElementById('firewall-confirm-modal');
            if (!modal) {
                if (!confirm(`Are you sure you want to block IP ${clean} in the OS Firewall?`)) return;
                performRealBlock(clean);
                return;
            }
            document.getElementById('fw-confirm-ip').innerText = clean;
            const isWin = navigator.platform.toLowerCase().includes('win');
            const cmd = isWin 
                ? `netsh advfirewall firewall add rule name="Block-GNN-${clean}" dir=in action=block remoteip=${clean}`
                : `iptables -A INPUT -s ${clean} -j DROP`;
            document.getElementById('fw-confirm-cmd').innerText = cmd;
            const statusBox = document.getElementById('fw-confirm-status');
            statusBox.style.display = 'none';
            statusBox.innerText = '';
            const btn = document.getElementById('btn-do-real-block');
            btn.disabled = false;
            btn.innerText = 'Execute Real OS Block';
            modal.style.display = 'flex';
        }

        function closeFirewallModal() {
            const modal = document.getElementById('firewall-confirm-modal');
            if (modal) modal.style.display = 'none';
            pendingBlockIp = null;
        }

        async function confirmRealBlock() {
            if (!pendingBlockIp) return;
            const clean = pendingBlockIp;
            const btn = document.getElementById('btn-do-real-block');
            const statusBox = document.getElementById('fw-confirm-status');
            btn.disabled = true;
            btn.innerText = 'Applying Rule...';
            
            try {
                const res = await postAdmin('/api/execute_real_block', { ip: clean });
                const data = await res.json();
                statusBox.style.display = 'block';
                if (!res.ok) {
                    statusBox.style.background = 'rgba(255, 0, 85, 0.2)';
                    statusBox.style.color = 'var(--critical-crimson)';
                    statusBox.innerText = `[FIREWALL ACTION ERROR ${res.status}]: ${data.message || 'Block action failed'}`;
                    btn.disabled = false;
                    btn.innerText = 'Retry Block';
                } else {
                    statusBox.style.background = 'rgba(0, 255, 157, 0.2)';
                    statusBox.style.color = 'var(--emerald-green)';
                    statusBox.innerText = `[SUCCESS]: ${data.message || 'Rule applied successfully.'} (Mode: ${data.execution_mode || 'ENFORCED'})`;
                    btn.innerText = '✓ Block Applied';
                    setTimeout(() => {
                        closeFirewallModal();
                    }, 1200);
                }
            } catch(e) {
                statusBox.style.display = 'block';
                statusBox.style.background = 'rgba(255, 0, 85, 0.2)';
                statusBox.style.color = 'var(--critical-crimson)';
                statusBox.innerText = `[NETWORK/AUTH ERROR]: ${e.message}`;
                btn.disabled = false;
                btn.innerText = 'Retry Block';
            }
        }

        async function performRealBlock(clean) {
            try {
                const res = await postAdmin('/api/execute_real_block', { ip: clean });
                const data = await res.json();
                if (!res.ok) {
                    alert(`[FIREWALL ACTION ERROR ${res.status}]:\n${data.message || 'Block action failed'}`);
                    return;
                }
                alert(`[ADMIN FIREWALL RESPONSE]:\n${data.message || data.status}`);
            } catch (err) {
                alert(`[FIREWALL NETWORK FAILURE]:\nUnable to reach server: ${err.message}`);
            }
        }

        async function openBlockHistoryModal() {
            const modal = document.getElementById('block-history-modal');
            if (modal) {
                modal.style.display = 'flex';
                await refreshBlockHistory();
            }
        }

        function closeBlockHistoryModal() {
            const modal = document.getElementById('block-history-modal');
            if (modal) modal.style.display = 'none';
        }

        async function refreshBlockHistory() {
            try {
                let token = localStorage.getItem('cybergraph_admin_token') || adminToken;
                const headers = {};
                if (token) {
                    headers['Authorization'] = 'Bearer ' + token;
                    headers['X-Admin-Token'] = token;
                }
                const res = await fetch('/api/blocks/history', { headers: headers });
                if (!res.ok) {
                    const el = document.getElementById('blocked-ips-container');
                    if (el) el.innerHTML = `<span style="color:var(--warning-amber);">Failed to load blocks (HTTP ${res.status}). Authenticate as admin.</span>`;
                    return;
                }
                const data = await res.json();
                const allBlocked = data.all_blocked || [];
                const countEl = document.getElementById('blocked-count');
                if (countEl) countEl.innerText = allBlocked.length;

                const container = document.getElementById('blocked-ips-container');
                if (container) {
                    if (allBlocked.length === 0) {
                        container.innerHTML = '<span style="color:var(--text-muted);">No blocked IPs currently active.</span>';
                    } else {
                        container.innerHTML = allBlocked.map(ip => {
                            const clean = sanitizeIp(ip);
                            const isReal = (data.real_blocked || []).includes(clean);
                            const badge = isReal 
                                ? '<span style="color:var(--critical-crimson); font-weight:700;">[OS FIREWALL]</span>' 
                                : '<span style="color:var(--warning-amber);">[SIMULATED ACL]</span>';
                            return `
                                <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 8px; border-bottom:1px solid rgba(255,255,255,0.05);">
                                    <div><strong>${clean}</strong> ${badge}</div>
                                    <button class="cyber-btn unblock-btn" style="width:auto; margin:0; padding:3px 8px; font-size:9px;" data-action="unblock-hist" data-ip="${clean}">Unblock</button>
                                </div>
                            `;
                        }).join('');
                    }
                }

                const tbody = document.getElementById('fw-audit-tbody');
                if (tbody) {
                    const hist = data.history || [];
                    if (hist.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted); text-align:center;">No actions logged yet.</td></tr>';
                    } else {
                        tbody.innerHTML = hist.slice().reverse().map(h => {
                            const ts = h.timestamp ? new Date(h.timestamp * 1000).toLocaleTimeString() : '-';
                            const actColor = h.action === 'BLOCK' ? 'var(--critical-crimson)' : 'var(--emerald-green)';
                            return `
                                <tr>
                                    <td>${ts}</td>
                                    <td style="color:${actColor}; font-weight:700;">${escapeHtml(h.action)}</td>
                                    <td>${escapeHtml(h.ip)}</td>
                                    <td>${escapeHtml(h.execution_mode || '-')}</td>
                                    <td>${escapeHtml(h.status)}</td>
                                </tr>
                            `;
                        }).join('');
                    }
                }
            } catch(e) {
                console.warn('Error fetching block history:', e);
            }
        }

        async function unblockFromHistory(ip) {
            await unblockIp(ip);
            await refreshBlockHistory();
        }

        async function unblockIp(targetIp) {
            const clean = sanitizeIp(targetIp);
            if (!clean) return;
            if (!confirm(`Are you sure you want to unblock IP ${clean}?`)) return;
            try {
                const res = await postAdmin('/api/unblock_ip', { ip: clean });
                const data = await res.json();
                if (!res.ok) {
                    alert(`[FIREWALL UNBLOCK ERROR ${res.status}]:\n${data.message || 'Unblock action failed'}`);
                    return;
                }
                alert(`[ADMIN UNBLOCK RESULT]:\n${data.message || data.status}`);
                const histModal = document.getElementById('block-history-modal');
                if (histModal && histModal.style.display === 'flex') {
                    await refreshBlockHistory();
                }
            } catch (err) {
                alert(`[UNBLOCK NETWORK FAILURE]:\nUnable to reach server: ${err.message}`);
            }
        }

        async function updateDeviceAccessSection() {
            try {
                const res = await fetch('/api/device_requests');
                if (!res.ok) {
                    console.warn(`Device requests fetch failed: HTTP ${res.status}`);
                    return;
                }
                const data = await res.json();
                const authContainer = document.getElementById('auth-device-list');
                const pendContainer = document.getElementById('pending-device-list');
                if (!authContainer || !pendContainer) return;

                const isAdmin = !!data.is_admin;

                let authHtml = '';
                if (data.authorized_devices && data.authorized_devices.length > 0) {
                    data.authorized_devices.forEach(dev => {
                        let actionHtml = '';
                        const safeDevId = escapeHtml(dev.device_id || '');
                        const safeIp = escapeHtml(dev.client_ip || '');
                        if (dev.is_server) {
                            actionHtml = '<span style="color:var(--neon-cyan); font-size:8px; font-weight:700;">SERVER</span>';
                        } else if (isAdmin) {
                            actionHtml = `<button class="cyber-btn" style="padding:2px 6px; font-size:8px; background:var(--critical-crimson); color:white; width:auto; margin:0;" data-action="revoke-dev" data-dev-id="${safeDevId}">REVOKE</button>`;
                        }
                        const devLabel = escapeHtml(dev.is_server ? 'CyberGraph Server' : (dev.device_name || 'Approved Device'));
                        authHtml += `
                            <div style="display:flex; justify-content:space-between; align-items:center; padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <div>
                                    <span style="color:var(--emerald-green); font-weight:700;">${safeIp}</span>
                                    <span style="color:var(--text-muted); font-size:8px;"> (${devLabel})</span>
                                </div>
                                <div>
                                    ${actionHtml}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    authHtml = '<div style="font-size:9px; color:var(--text-muted);">No authorized devices</div>';
                }
                authContainer.innerHTML = authHtml;

                let pendHtml = '';
                if (data.pending_requests && data.pending_requests.length > 0) {
                    data.pending_requests.forEach(req => {
                        let actionHtml = '';
                        const safeReqId = escapeHtml(req.request_id || '');
                        const safeIp = escapeHtml(req.client_ip || '');
                        if (isAdmin) {
                            actionHtml = `<button class="cyber-btn" style="padding:2px 6px; font-size:8px; background:var(--emerald-green); color:black; width:auto; margin:0;" data-action="approve-dev" data-req-id="${safeReqId}">APPROVE</button>`;
                        }
                        pendHtml += `
                            <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <div>
                                    <span style="color:var(--warning-amber); font-weight:700;">${safeIp}</span>
                                    <span style="color:var(--text-muted); font-size:8px;"> PENDING</span>
                                </div>
                                <div>
                                    ${actionHtml}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    pendHtml = '<div style="font-size:9px; color:var(--text-muted);">No pending requests</div>';
                }
                pendContainer.innerHTML = pendHtml;
            } catch(e) {
                console.error("Device access UI error", e);
            }
        }

        async function approveDevice(reqId) {
            try {
                const res = await postAdmin('/api/approve_device', { request_id: reqId });
                if (!res.ok) console.warn(`Approve device returned HTTP ${res.status}`);
            } catch(e) {
                console.error("Approve device error", e);
            }
            updateDeviceAccessSection();
            fetchStreamStep();
        }

        async function revokeDevice(devId) {
            try {
                const res = await postAdmin('/api/revoke_device', { device_id: devId });
                if (!res.ok) console.warn(`Revoke device returned HTTP ${res.status}`);
            } catch(e) {
                console.error("Revoke device error", e);
            }
            updateDeviceAccessSection();
            fetchStreamStep();
        }

        let currentEvals = [];
        const acknowledgedEvents = new Set();
        const falsePositiveIps = new Set();

        function acknowledgeAlert(btn, ip) {
            const card = btn.closest('.threat-card');
            if (card) {
                card.style.opacity = '0.4';
                card.style.filter = 'grayscale(0.8)';
                btn.textContent = 'Acknowledged';
                btn.disabled = true;
            }
        }

        function markFalsePositive(btn, ip) {
            if (!confirm(`Mark IP ${ip} as False Positive? Future alerts will be suppressed.`)) return;
            falsePositiveIps.add(ip);
            const card = btn.closest('.threat-card');
            if (card) {
                card.style.display = 'none';
            }
        }

        function renderFilteredAlerts() {
            const searchVal = (document.getElementById('alert-search')?.value || '').toLowerCase().trim();
            const levelVal = document.getElementById('alert-level-filter')?.value || 'ALL';
            const feed = document.getElementById('alert-feed');
            if (!feed) return;

            feed.innerHTML = '';
            let visibleCount = 0;

            currentEvals.forEach(ev => {
                if (falsePositiveIps.has(ev.source_host) || falsePositiveIps.has(ev.target_service)) {
                    return;
                }

                if (levelVal !== 'ALL') {
                    if (!ev.level || !ev.level.includes(levelVal)) return;
                }

                if (searchVal) {
                    const textContent = `${ev.source_host} ${ev.target_service} ${ev.summary} ${ev.level}`.toLowerCase();
                    const mitreMatch = (ev.mitre_mappings || []).some(m => `${m.technique_id} ${m.technique_name}`.toLowerCase().includes(searchVal));
                    if (!textContent.includes(searchVal) && !mitreMatch) return;
                }

                visibleCount++;
                const isNormal = ev.level && ev.level.includes('Level 1');
                const card = document.createElement('div');
                card.className = `threat-card ${isNormal ? 'lvl-1' : (ev.level.includes('Level 2') ? 'lvl-2' : 'lvl-3')}`;

                const safeLvl = escapeHtml(ev.level || 'Unknown');
                const safeBadge = escapeHtml(ev.badge_color || '#10b981');
                const safeSrc = escapeHtml(ev.source_host || '-');
                const safeDst = escapeHtml(ev.target_service || '-');
                const safeSummary = escapeHtml(ev.summary || '');
                const safeRisk = Number(ev.risk_score || 0).toFixed(4);

                let html = `
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <strong>${safeLvl}</strong>
                        <span style="background:${safeBadge}; color:#000; padding:2px 6px; border-radius:4px; font-weight:700;">Risk: ${safeRisk}</span>
                    </div>
                    <div><strong>Src:</strong> ${safeSrc} &rarr; <strong>Dst:</strong> ${safeDst}</div>
                    <div style="margin-top:4px; color:var(--text-muted);">${safeSummary}</div>
                `;

                if (!isNormal && ev.xai_explanation && ev.xai_explanation.why_this_attack_reasons) {
                    const reasons = ev.xai_explanation.why_this_attack_reasons;
                    html += `
                        <div class="why-attack-box">
                            <div class="why-title">WHY THIS ATTACK? (SHAP & SALIENCY)</div>
                            ${reasons.map(r => `<div>${escapeHtml(r)}</div>`).join('')}
                        </div>
                    `;
                }

                if (!isNormal && ev.bipartite_attack_chain) {
                    const safeChain = ev.bipartite_attack_chain.map(c => escapeHtml(c)).join(' &rarr; ');
                    html += `<div style="margin-top:6px; font-family:'JetBrains Mono'; font-size:10px; color:#ff99bb;"><strong>Attack Chain:</strong><br>${safeChain}</div>`;

                if (!isNormal && ev.xai_explanation) {
                    const xai = ev.xai_explanation;
                    const topShap = xai.shap_top_features || [];
                    const attributions = xai.feature_attributions || {};
                    let shapItems = [];
                    if (topShap.length > 0) {
                        shapItems = topShap.slice(0, 4).map(item => ({
                            name: Array.isArray(item) ? item[0] : (item.name || 'Feature'),
                            val: Array.isArray(item) ? item[1] : (item.val || 0),
                        }));
                    } else if (Object.keys(attributions).length > 0) {
                        shapItems = Object.entries(attributions)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 4)
                            .map(([name, val]) => ({ name, val }));
                    }

                    if (shapItems.length > 0) {
                        html += `
                            <div style="margin-top:6px; background:rgba(0,0,0,0.5); border:1px solid rgba(0, 240, 255, 0.25); border-radius:6px; padding:6px 8px;">
                                <div style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; color:var(--neon-cyan); margin-bottom:4px; display:flex; justify-content:space-between;">
                                    <span>SHAP TOP-FEATURE DRIVERS</span>
                                    <span style="color:var(--text-muted); font-size:8px;">XAI SALIENCY</span>
                                </div>
                                ${shapItems.map(item => {
                                    const fName = escapeHtml(String(item.name || ''));
                                    const fValNum = Number(item.val || 0);
                                    const fValStr = (fValNum >= 0 ? '+' : '') + fValNum.toFixed(4);
                                    const pct = Math.min(100, Math.max(12, Math.abs(fValNum) * 100));
                                    const barColor = fValNum >= 0 ? 'var(--critical-crimson)' : 'var(--emerald-green)';
                                    return `
                                        <div style="margin:2px 0; font-family:'JetBrains Mono',monospace; font-size:9px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:1px;">
                                                <span>${fName}</span>
                                                <span style="color:${barColor}; font-weight:700;">${fValStr}</span>
                                            </div>
                                            <div style="width:100%; height:3px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                                                <div style="width:${pct}%; height:100%; background:${barColor};"></div>
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        `;
                    }
                }

                }

                if (!isNormal && ev.mitre_mappings && ev.mitre_mappings.length > 0) {
                    html += `<div class="mitre-section">`;
                    html += `<div style="color:var(--mitre-blue); font-weight:700; margin-bottom:4px;">MITRE ATT&CK Techniques Detected:</div>`;
                    ev.mitre_mappings.forEach(m => {
                        const safeTechId = escapeHtml(m.technique_id || '');
                        const safeTechName = escapeHtml(m.technique_name || '');
                        const safeConf = (Number(m.confidence || 0) * 100).toFixed(1);
                        const safeRationale = escapeHtml(m.rationale || '');
                        const safeTactic = escapeHtml(m.tactic || '');

                        html += `<div style="margin:4px 0;">`;
                        html += `<span class="mitre-badge">${safeTechId}</span> `;
                        html += `<span style="color:var(--text-main);">${safeTechName}</span> `;
                        html += `<span class="mitre-confidence">${safeConf}%</span>`;
                        html += `</div>`;
                        if (safeRationale) {
                            html += `<div class="mitre-rationale">${safeRationale}</div>`;
                        }
                        if (safeTactic) {
                            html += `<div style="color:var(--text-muted); font-size:8px;">Tactic: ${safeTactic}</div>`;
                        }
                    });
                    html += `</div>`;
                }

                if (!isNormal && ev.recommended_firewall_action) {
                    const fw = ev.recommended_firewall_action;
                    const ipClean = sanitizeIp(fw.target_ip || '');
                    const safePreview = escapeHtml(fw.windows_netsh_rule_preview || '');
                    html += `
                        <div style="margin-top:8px;">
                            <strong style="color:var(--critical-crimson);">Recommended Block Action (Simulated Response):</strong>
                            <div class="cmd-box">${safePreview}</div>
                            <button class="cyber-btn real-fw-btn" data-action="real-block" data-ip="${ipClean}">Admin Approval: Execute Real OS Firewall Rule</button>
                            <button class="cyber-btn unblock-btn" data-action="unblock" data-ip="${ipClean}">Admin Action: Unblock IP</button>
                            <div style="display:flex; gap:6px; margin-top:8px;">
                                <button class="cyber-btn" style="padding:4px 8px; font-size:9px; background:rgba(255,255,255,0.08); color:white; width:auto; margin:0;" data-action="ack" data-ip="${ipClean}">✓ Acknowledge</button>
                                <button class="cyber-btn" style="padding:4px 8px; font-size:9px; background:rgba(245,158,11,0.2); border:1px solid var(--warning-amber); color:var(--warning-amber); width:auto; margin:0;" data-action="fp" data-ip="${ipClean}">⚠ False Positive</button>
                            </div>
                        </div>
                    `;
                }

                card.innerHTML = html;
                feed.appendChild(card);
            });

            const cntEl = document.getElementById('threat-feed-count');
            if (cntEl) cntEl.innerText = `${visibleCount} events`;
        }

        let isPolling = false;
        let lastDeviceStateVersion = -1;
        let lastDevicePollTime = 0;

        function handleStreamData(data) {
            if (!data || !data.stats) return;
            if (isPolling) return;
            isPolling = true;

            let data;
            try {
                const res = await fetch('/api/stream_step');
                if (!res.ok) {
                    console.warn(`Stream step HTTP ${res.status}`);
                    return;
                }
                data = await res.json();
            } catch(e) {
                console.error("Stream step network error:", e);
                return;
            } finally {
                isPolling = false;
            }

        }

        async function fetchStreamStep() {
            if (isPolling) return;
            isPolling = true;

            try {
                const res = await fetch('/api/stream_step');
                if (!res.ok) {
                    showConnectionBanner();
                    return;
                }
                hideConnectionBanner();
                const data = await res.json();
                handleStreamData(data);
            } catch(e) {
                console.error("Stream step network error:", e);
                showConnectionBanner();
            } finally {
                isPolling = false;
            }
        }

        function _handleStreamDataInner(data) {
            // Avoid unnecessary repeated device-access calls:
            // Only update device section if state_version changed or periodically after 10s
            const nowTime = Date.now();
            if (data.state_version !== lastDeviceStateVersion || (nowTime - lastDevicePollTime) > 10000) {
                lastDeviceStateVersion = data.state_version;
                lastDevicePollTime = nowTime;
                updateDeviceAccessSection();
            }

            if (data.state_version) {
                document.getElementById('version-tag').innerText = `STATE V: ${data.state_version}`;
            }

            document.getElementById('val-total').innerText = data.stats.total_flows;
            document.getElementById('val-l1').innerText = data.stats.level1_normal;
            document.getElementById('val-l2').innerText = data.stats.level2_suspicious;
            document.getElementById('val-l3').innerText = data.stats.level3_critical;
            document.getElementById('val-hosts').innerText = data.stats.active_hosts;
            document.getElementById('val-services').innerText = data.stats.active_services;
            document.getElementById('val-risk').innerText = data.stats.current_risk.toFixed(4);

            const dsBadge = document.getElementById('ds-badge');
            const graphTitle = document.getElementById('graph-panel-title');
            dsBadge.innerText = `● DATA SOURCE: ${data.stats.current_data_source}`;

            if (data.stats.live_buffer_count !== undefined) {
                document.getElementById('cap-buf').innerText = `${data.stats.live_buffer_count} Packets`;
            }

            const btnAtk = document.getElementById('btn-attack');
            const btnNorm = document.getElementById('btn-normal');
            const btnOff = document.getElementById('btn-offline');

            if (data.mode === 'attack') {
                if (btnAtk) btnAtk.style.boxShadow = '0 0 15px rgba(255, 0, 85, 0.6)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            } else if (data.mode === 'offline') {
                if (btnOff) btnOff.style.boxShadow = '0 0 15px rgba(14, 165, 233, 0.8)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
            } else {
                if (btnNorm) btnNorm.style.boxShadow = '0 0 15px rgba(0, 255, 157, 0.6)';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            }

            function setGraphTitle(accentColor, titleText) {
                if (!graphTitle) return;
                graphTitle.textContent = ' ' + titleText;
                const acc = document.createElement('div');
                acc.className = 'title-accent';
                if (accentColor) acc.style.background = accentColor;
                graphTitle.prepend(acc);
            }

            if (data.stats.current_data_source.includes('LIVE')) {
                if (data.stats.current_data_source.includes('DISCONNECTED') || data.stats.current_data_source.includes('NO PHYSICAL')) {
                    dsBadge.style.color = 'var(--warning-amber)';
                    dsBadge.style.borderColor = 'var(--warning-amber)';
                    setGraphTitle('var(--warning-amber)', 'Live Wi-Fi Standby (No Packets Captured)');
                } else {
                    dsBadge.style.color = 'var(--emerald-green)';
                    dsBadge.style.borderColor = 'var(--emerald-green)';
                    setGraphTitle('var(--emerald-green)', 'Live Wi-Fi Bipartite Graph Radar (HOST ↔ SERVICE)');
                }
            } else if (data.stats.current_data_source.includes('ATTACK')) {
                dsBadge.style.color = 'var(--critical-crimson)';
                dsBadge.style.borderColor = 'var(--critical-crimson)';
                setGraphTitle('var(--critical-crimson)', 'Replay Attack Bipartite Graph Radar (HOST ↔ SERVICE)');
            } else {
                dsBadge.style.color = 'var(--neon-cyan)';
                dsBadge.style.borderColor = 'var(--neon-cyan)';
                setGraphTitle('', 'Offline Bipartite Graph Radar (NF-UNSW-NB15)');
            }

            if (data.latencies) {
                document.getElementById('lat-cap').innerText = data.latencies.capture.toFixed(2);
                document.getElementById('lat-prep').innerText = data.latencies.feature_prep.toFixed(2);
                document.getElementById('lat-gnn').innerText = data.latencies.gnn_infer.toFixed(2);
                document.getElementById('lat-tot').innerText = data.latencies.total.toFixed(2);
                document.getElementById('val-lat').innerText = `${data.latencies.total.toFixed(2)} ms`;
            }

            if (data.pipeline_metrics) {
                const elXgb = document.getElementById('m-xgb');
                if (elXgb) elXgb.innerText = data.pipeline_metrics.xgb_prob.toFixed(4);
                const elAe = document.getElementById('m-ae');
                if (elAe) elAe.innerText = data.pipeline_metrics.ae_score.toFixed(4);
                const elFus = document.getElementById('m-fus');
                if (elFus) elFus.innerText = data.pipeline_metrics.fusion_score.toFixed(4);
                const elGnn = document.getElementById('m-gnn');
                if (elGnn) elGnn.innerText = data.pipeline_metrics.gnn_risk.toFixed(4);
                const elFin = document.getElementById('m-fin');
                if (elFin) elFin.innerText = data.pipeline_metrics.final_tier;
            }

            if (data.stats.risk_history) {
                riskHistory = data.stats.risk_history;
                drawSparkline();
            }
            updateAllCharts(data);

            currentEvals = data.evals || [];
            renderFilteredAlerts();

            const cntEl = document.getElementById('threat-feed-count');
            if (cntEl) cntEl.innerText = `${visibleCount} events`;
        }

        let isPolling = false;
        let lastDeviceStateVersion = -1;
        let lastDevicePollTime = 0;

        function handleStreamData(data) {
            if (!data || !data.stats) return;
            if (isPolling) return;
            isPolling = true;

            let data;
            try {
                const res = await fetch('/api/stream_step');
                if (!res.ok) {
                    console.warn(`Stream step HTTP ${res.status}`);
                    return;
                }
                data = await res.json();
            } catch(e) {
                console.error("Stream step network error:", e);
                return;
            } finally {
                isPolling = false;
            }

        }

        async function fetchStreamStep() {
            if (isPolling) return;
            isPolling = true;

            try {
                const res = await fetch('/api/stream_step');
                if (!res.ok) {
                    showConnectionBanner();
                    return;
                }
                hideConnectionBanner();
                const data = await res.json();
                handleStreamData(data);
            } catch(e) {
                console.error("Stream step network error:", e);
                showConnectionBanner();
            } finally {
                isPolling = false;
            }
        }

        function _handleStreamDataInner(data) {
            // Avoid unnecessary repeated device-access calls:
            // Only update device section if state_version changed or periodically after 10s
            const nowTime = Date.now();
            if (data.state_version !== lastDeviceStateVersion || (nowTime - lastDevicePollTime) > 10000) {
                lastDeviceStateVersion = data.state_version;
                lastDevicePollTime = nowTime;
                updateDeviceAccessSection();
            }

            if (data.state_version) {
                document.getElementById('version-tag').innerText = `STATE V: ${data.state_version}`;
            }

            document.getElementById('val-total').innerText = data.stats.total_flows;
            document.getElementById('val-l1').innerText = data.stats.level1_normal;
            document.getElementById('val-l2').innerText = data.stats.level2_suspicious;
            document.getElementById('val-l3').innerText = data.stats.level3_critical;
            document.getElementById('val-hosts').innerText = data.stats.active_hosts;
            document.getElementById('val-services').innerText = data.stats.active_services;
            document.getElementById('val-risk').innerText = data.stats.current_risk.toFixed(4);

            const dsBadge = document.getElementById('ds-badge');
            const graphTitle = document.getElementById('graph-panel-title');
            dsBadge.innerText = `● DATA SOURCE: ${data.stats.current_data_source}`;

            if (data.stats.live_buffer_count !== undefined) {
                document.getElementById('cap-buf').innerText = `${data.stats.live_buffer_count} Packets`;
            }

            const btnAtk = document.getElementById('btn-attack');
            const btnNorm = document.getElementById('btn-normal');
            const btnOff = document.getElementById('btn-offline');

            if (data.mode === 'attack') {
                if (btnAtk) btnAtk.style.boxShadow = '0 0 15px rgba(255, 0, 85, 0.6)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            } else if (data.mode === 'offline') {
                if (btnOff) btnOff.style.boxShadow = '0 0 15px rgba(14, 165, 233, 0.8)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
            } else {
                if (btnNorm) btnNorm.style.boxShadow = '0 0 15px rgba(0, 255, 157, 0.6)';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            }

            function setGraphTitle(accentColor, titleText) {
                if (!graphTitle) return;
                graphTitle.textContent = ' ' + titleText;
                const acc = document.createElement('div');
                acc.className = 'title-accent';
                if (accentColor) acc.style.background = accentColor;
                graphTitle.prepend(acc);
            }

            if (data.stats.current_data_source.includes('LIVE')) {
                if (data.stats.current_data_source.includes('DISCONNECTED') || data.stats.current_data_source.includes('NO PHYSICAL')) {
                    dsBadge.style.color = 'var(--warning-amber)';
                    dsBadge.style.borderColor = 'var(--warning-amber)';
                    setGraphTitle('var(--warning-amber)', 'Live Wi-Fi Standby (No Packets Captured)');
                } else {
                    dsBadge.style.color = 'var(--emerald-green)';
                    dsBadge.style.borderColor = 'var(--emerald-green)';
                    setGraphTitle('var(--emerald-green)', 'Live Wi-Fi Bipartite Graph Radar (HOST ↔ SERVICE)');
                }
            } else if (data.stats.current_data_source.includes('ATTACK')) {
                dsBadge.style.color = 'var(--critical-crimson)';
                dsBadge.style.borderColor = 'var(--critical-crimson)';
                setGraphTitle('var(--critical-crimson)', 'Replay Attack Bipartite Graph Radar (HOST ↔ SERVICE)');
            } else {
                dsBadge.style.color = 'var(--neon-cyan)';
                dsBadge.style.borderColor = 'var(--neon-cyan)';
                setGraphTitle('', 'Offline Bipartite Graph Radar (NF-UNSW-NB15)');
            }

            if (data.latencies) {
                document.getElementById('lat-cap').innerText = data.latencies.capture.toFixed(2);
                document.getElementById('lat-prep').innerText = data.latencies.feature_prep.toFixed(2);
                document.getElementById('lat-gnn').innerText = data.latencies.gnn_infer.toFixed(2);
                document.getElementById('lat-tot').innerText = data.latencies.total.toFixed(2);
                document.getElementById('val-lat').innerText = `${data.latencies.total.toFixed(2)} ms`;
            }

            if (data.pipeline_metrics) {
                const elXgb = document.getElementById('m-xgb');
                if (elXgb) elXgb.innerText = data.pipeline_metrics.xgb_prob.toFixed(4);
                const elAe = document.getElementById('m-ae');
                if (elAe) elAe.innerText = data.pipeline_metrics.ae_score.toFixed(4);
                const elFus = document.getElementById('m-fus');
                if (elFus) elFus.innerText = data.pipeline_metrics.fusion_score.toFixed(4);
                const elGnn = document.getElementById('m-gnn');
                if (elGnn) elGnn.innerText = data.pipeline_metrics.gnn_risk.toFixed(4);
                const elFin = document.getElementById('m-fin');
                if (elFin) elFin.innerText = data.pipeline_metrics.final_tier;
            }

            if (data.stats.risk_history) {
                riskHistory = data.stats.risk_history;
                drawSparkline();
            }

            const feed = document.getElementById('alert-feed');
            feed.innerHTML = '';

            data.evals.forEach(ev => {
                const isNormal = ev.level && ev.level.includes('Level 1');
                const card = document.createElement('div');
                card.className = `threat-card ${isNormal ? 'lvl-1' : (ev.level.includes('Level 2') ? 'lvl-2' : 'lvl-3')}`;

                const safeLvl = escapeHtml(ev.level || 'Unknown');
                const safeBadge = escapeHtml(ev.badge_color || '#10b981');
                const safeSrc = escapeHtml(ev.source_host || '-');
                const safeDst = escapeHtml(ev.target_service || '-');
                const safeSummary = escapeHtml(ev.summary || '');
                const safeRisk = Number(ev.risk_score || 0).toFixed(4);

                let html = `
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <strong>${safeLvl}</strong>
                        <span style="background:${safeBadge}; color:#000; padding:2px 6px; border-radius:4px; font-weight:700;">Risk: ${safeRisk}</span>
                    </div>
                    <div><strong>Src:</strong> ${safeSrc} &rarr; <strong>Dst:</strong> ${safeDst}</div>
                    <div style="margin-top:4px; color:var(--text-muted);">${safeSummary}</div>
                `;

                if (!isNormal && ev.xai_explanation && ev.xai_explanation.why_this_attack_reasons) {
                    const reasons = ev.xai_explanation.why_this_attack_reasons;
                    html += `
                        <div class="why-attack-box">
                            <div class="why-title">WHY THIS ATTACK? (SHAP & SALIENCY)</div>
                            ${reasons.map(r => `<div>${escapeHtml(r)}</div>`).join('')}
                        </div>
                    `;
                }

                if (!isNormal && ev.bipartite_attack_chain) {
                    const safeChain = ev.bipartite_attack_chain.map(c => escapeHtml(c)).join(' &rarr; ');
                    html += `<div style="margin-top:6px; font-family:'JetBrains Mono'; font-size:10px; color:#ff99bb;"><strong>Attack Chain:</strong><br>${safeChain}</div>`;

                if (!isNormal && ev.xai_explanation) {
                    const xai = ev.xai_explanation;
                    const topShap = xai.shap_top_features || [];
                    const attributions = xai.feature_attributions || {};
                    let shapItems = [];
                    if (topShap.length > 0) {
                        shapItems = topShap.slice(0, 4).map(item => ({
                            name: Array.isArray(item) ? item[0] : (item.name || 'Feature'),
                            val: Array.isArray(item) ? item[1] : (item.val || 0),
                        }));
                    } else if (Object.keys(attributions).length > 0) {
                        shapItems = Object.entries(attributions)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 4)
                            .map(([name, val]) => ({ name, val }));
                    }

                    if (shapItems.length > 0) {
                        html += `
                            <div style="margin-top:6px; background:rgba(0,0,0,0.5); border:1px solid rgba(0, 240, 255, 0.25); border-radius:6px; padding:6px 8px;">
                                <div style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; color:var(--neon-cyan); margin-bottom:4px; display:flex; justify-content:space-between;">
                                    <span>SHAP TOP-FEATURE DRIVERS</span>
                                    <span style="color:var(--text-muted); font-size:8px;">XAI SALIENCY</span>
                                </div>
                                ${shapItems.map(item => {
                                    const fName = escapeHtml(String(item.name || ''));
                                    const fValNum = Number(item.val || 0);
                                    const fValStr = (fValNum >= 0 ? '+' : '') + fValNum.toFixed(4);
                                    const pct = Math.min(100, Math.max(12, Math.abs(fValNum) * 100));
                                    const barColor = fValNum >= 0 ? 'var(--critical-crimson)' : 'var(--emerald-green)';
                                    return `
                                        <div style="margin:2px 0; font-family:'JetBrains Mono',monospace; font-size:9px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:1px;">
                                                <span>${fName}</span>
                                                <span style="color:${barColor}; font-weight:700;">${fValStr}</span>
                                            </div>
                                            <div style="width:100%; height:3px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                                                <div style="width:${pct}%; height:100%; background:${barColor};"></div>
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        `;
                    }
                }

                }

                if (!isNormal && ev.mitre_mappings && ev.mitre_mappings.length > 0) {
                    html += `<div class="mitre-section">`;
                    html += `<div style="color:var(--mitre-blue); font-weight:700; margin-bottom:4px;">MITRE ATT&CK Techniques Detected:</div>`;
                    ev.mitre_mappings.forEach(m => {
                        const safeTechId = escapeHtml(m.technique_id || '');
                        const safeTechName = escapeHtml(m.technique_name || '');
                        const safeConf = (Number(m.confidence || 0) * 100).toFixed(1);
                        const safeRationale = escapeHtml(m.rationale || '');
                        const safeTactic = escapeHtml(m.tactic || '');

                        html += `<div style="margin:4px 0;">`;
                        html += `<span class="mitre-badge">${safeTechId}</span> `;
                        html += `<span style="color:var(--text-main);">${safeTechName}</span> `;
                        html += `<span class="mitre-confidence">${safeConf}%</span>`;
                        html += `</div>`;
                        if (safeRationale) {
                            html += `<div class="mitre-rationale">${safeRationale}</div>`;
                        }
                        if (safeTactic) {
                            html += `<div style="color:var(--text-muted); font-size:8px;">Tactic: ${safeTactic}</div>`;
                        }
                    });
                    html += `</div>`;
                }

                if (!isNormal && ev.recommended_firewall_action) {
                    const fw = ev.recommended_firewall_action;
                    const ipClean = sanitizeIp(fw.target_ip || '');
                    const safePreview = escapeHtml(fw.windows_netsh_rule_preview || '');
                    html += `
                        <div style="margin-top:8px;">
                            <strong style="color:var(--critical-crimson);">Recommended Block Action (Simulated Response):</strong>
                            <div class="cmd-box">${safePreview}</div>
                            <button class="cyber-btn real-fw-btn" data-action="real-block" data-ip="${ipClean}">Admin Approval: Execute Real OS Firewall Rule</button>
                            <button class="cyber-btn unblock-btn" data-action="unblock" data-ip="${ipClean}">Admin Action: Unblock IP</button>
                        </div>
                    `;
                }

                card.innerHTML = html;
                feed.appendChild(card);
            });

            // PERFECT DYNAMIC BIPARTITE GRAPH FIT
            // Left column: HOST + GATEWAY + EXTERNAL nodes  |  Right column: SERVICE nodes
            const hostNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'HOST'));
            const gatewayNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'GATEWAY'));
            const externalNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'EXTERNAL'));
            const serviceNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'SERVICE'));

            // All left-side nodes combined (HOSTs first, then GATEWAY, then EXTERNALs)
            const leftNodes = [...hostNodes, ...gatewayNodes, ...externalNodes];

            const marginX = Math.max(140, Math.min(220, canvas.width * 0.28));
            const hostX = marginX;
            const serviceX = canvas.width - marginX;

            const topPad = 60;
            const bottomPad = 60;
            const availH = canvas.height - topPad - bottomPad;

            const lCount = leftNodes.length;
            const sCount = serviceNodes.length;

            const newNodePosMap = {};

            // Distribute all left-column nodes (HOST + GATEWAY + EXTERNAL) vertically with generous padding
            leftNodes.forEach((n, idx) => {
                const y = lCount <= 1 ? canvas.height / 2 : topPad + idx * (availH / (lCount - 1));
                newNodePosMap[n.label] = { x: hostX, y: y };
            });

            serviceNodes.forEach((s, idx) => {
                const y = sCount <= 1 ? canvas.height / 2 : topPad + idx * (availH / (sCount - 1));
                newNodePosMap[s.label] = { x: serviceX, y: y };
            });

            nodePosMap = newNodePosMap;

            nodes = data.graph_nodes.map(n => ({
                label: n.label,
                type: n.type,
                threat_level: n.threat_level || 'NORMAL',
                color: n.color || 'normal',
                x: nodePosMap[n.label] ? nodePosMap[n.label].x : ((n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'EXTERNAL') ? hostX : serviceX),
                y: nodePosMap[n.label] ? nodePosMap[n.label].y : canvas.height / 2,
            }));

            // Restrained Edge Routing: Prefer mostly straight lines; small ±12px offset only for multiple overlapping edges between same pair
            const edgePairCounts = {};
            const edgePairIndices = {};

            data.graph_edges.forEach(e => {
                const pairKey = `${e.source_label}->${e.target_label}`;
                edgePairCounts[pairKey] = (edgePairCounts[pairKey] || 0) + 1;
            });

            edges = data.graph_edges.map(e => {
                const pairKey = `${e.source_label}->${e.target_label}`;
                const idx = edgePairIndices[pairKey] || 0;
                edgePairIndices[pairKey] = idx + 1;
                const total = edgePairCounts[pairKey];

                const srcPos = nodePosMap[e.source_label] || { x: hostX, y: canvas.height / 2 };
                const dstPos = nodePosMap[e.target_label] || { x: serviceX, y: canvas.height / 2 };

                const midX = (srcPos.x + dstPos.x) / 2;
                const midY = (srcPos.y + dstPos.y) / 2;

                // Restrained curve offset: 0 for single edges (straight line!), ±14px max for duplicate parallel edges
                let curveOffset = 0;
                if (total > 1) {
                    const rawOffset = (idx - (total - 1) / 2) * 14;
                    curveOffset = Math.max(-18, Math.min(18, rawOffset));
                }

                return {
                    x1: srcPos.x, y1: srcPos.y,
                    x2: dstPos.x, y2: dstPos.y,
                    controlX: midX,
                    controlY: midY + curveOffset,
                    risk: e.risk, color: e.color || 'normal', level: e.level || 'NORMAL'
                };
            });

            particles = edges.map(e => ({
                x1: e.x1, y1: e.y1, x2: e.x2, y2: e.y2,
                controlX: e.controlX, controlY: e.controlY,
                risk: e.risk, color: e.color,
                progress: Math.random(), speed: 0.004 + e.risk * 0.012
            }));
        }

        document.getElementById('btn-normal').addEventListener('click', async () => {
            await postAdmin('/api/set_mode', { mode: 'normal' });
            await fetchStreamStep();
        });
        const btnOfflineEl = document.getElementById('btn-offline');
        if (btnOfflineEl) {
            btnOfflineEl.addEventListener('click', async () => {
                await postAdmin('/api/set_mode', { mode: 'offline' });
                await fetchStreamStep();
            });
        }
        document.getElementById('btn-attack').addEventListener('click', async () => {
            await postAdmin('/api/set_mode', { mode: 'attack' });
            await fetchStreamStep();
        });


        let sseActive = false;
        let sseSource = null;

        function showConnectionBanner() {
            const b = document.getElementById('conn-banner');
            if (b) b.style.display = 'block';
        }

        function hideConnectionBanner() {
            const b = document.getElementById('conn-banner');
            if (b) b.style.display = 'none';
        }

        function initSSE() {
            if (!window.EventSource) return;
            try {
                if (sseSource) sseSource.close();
                sseSource = new EventSource('/api/stream_events');
                sseSource.addEventListener('update', (e) => {
                    sseActive = true;
                    hideConnectionBanner();
                    try {
                        const data = JSON.parse(e.data);
                        handleStreamData(data);
                    } catch(err) {
                        console.error("SSE parse error", err);
                    }
                });
                sseSource.addEventListener('ping', () => {
                    sseActive = true;
                    hideConnectionBanner();
                });
                sseSource.onerror = () => {
                    sseActive = false;
                    showConnectionBanner();
                };
            } catch(e) {
                sseActive = false;
            }
        }

        function scheduleNextPoll() {
            setTimeout(async () => {
                if (!sseActive) {
                    await fetchStreamStep();
                }
                scheduleNextPoll();
            }, 850);
        }

        initSSE();
        fetchStreamStep().finally(() => {
            scheduleNextPoll();
        });
// Centralized CSP-Compliant Event Delegation
document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.getAttribute('data-action');
    const ip = btn.getAttribute('data-ip');
    const devId = btn.getAttribute('data-dev-id');
    const reqId = btn.getAttribute('data-req-id');

    if (action === 'real-block') executeRealBlock(ip);
    else if (action === 'unblock') unblockIp(ip);
    else if (action === 'unblock-hist') unblockFromHistory(ip);
    else if (action === 'ack') acknowledgeAlert(btn, ip);
    else if (action === 'fp') markFalsePositive(btn, ip);
    else if (action === 'revoke-dev') revokeDevice(devId);
    else if (action === 'approve-dev') approveDevice(reqId);
});


        // ─── ADMIN AUTH MODAL HANDLERS ───
        async function openAdminAuthModal() {
            const modal = document.getElementById('admin-auth-modal');
            if (!modal) return;
            modal.style.display = 'flex';

            const statusBadge = document.getElementById('admin-session-badge');
            const ipBadge = document.getElementById('admin-client-ip');
            const logoutBtn = document.getElementById('btn-admin-logout');
            const statusDiv = document.getElementById('admin-auth-status');
            if (statusDiv) statusDiv.style.display = 'none';

            try {
                const res = await fetch('/api/auth/status');
                if (res.ok) {
                    const data = await res.json();
                    if (statusBadge) {
                        statusBadge.innerText = data.is_authenticated_admin ? 'ADMINISTRATOR (ACTIVE)' : 'VISITOR (VIEW ONLY)';
                        statusBadge.style.color = data.is_authenticated_admin ? 'var(--emerald-green)' : 'var(--warning-amber)';
                    }
                    if (ipBadge) ipBadge.innerText = data.remote_addr || '127.0.0.1';
                    if (logoutBtn) logoutBtn.style.display = data.is_authenticated_admin ? 'inline-block' : 'none';
                }
            } catch(e) {
                console.warn('Status fetch error:', e);
            }
        }

        function closeAdminAuthModal() {
            const modal = document.getElementById('admin-auth-modal');
            if (modal) modal.style.display = 'none';
        }

        async function doAdminLogin() {
            const input = document.getElementById('admin-token-input');
            const statusDiv = document.getElementById('admin-auth-status');
            const token = input ? input.value.trim() : '';

            if (!token) {
                if (statusDiv) {
                    statusDiv.style.display = 'block';
                    statusDiv.style.background = 'rgba(255,0,85,0.2)';
                    statusDiv.style.color = 'var(--critical-crimson)';
                    statusDiv.innerText = 'Please enter an admin token.';
                }
                return;
            }

            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token: token })
                });
                const data = await res.json();
                if (res.ok && data.status === 'SUCCESS') {
                    if (data.csrf_token) csrfToken = data.csrf_token;
                    localStorage.setItem('cybergraph_admin_token', token);
                    adminToken = token;

                    if (statusDiv) {
                        statusDiv.style.display = 'block';
                        statusDiv.style.background = 'rgba(0,255,157,0.2)';
                        statusDiv.style.color = 'var(--emerald-green)';
                        statusDiv.innerText = '✓ Authenticated successfully as Administrator!';
                    }
                    setTimeout(() => {
                        closeAdminAuthModal();
                        updateDeviceAccessSection();
                        fetchBlockHistory();
                    }, 800);
                } else {
                    if (statusDiv) {
                        statusDiv.style.display = 'block';
                        statusDiv.style.background = 'rgba(255,0,85,0.2)';
                        statusDiv.style.color = 'var(--critical-crimson)';
                        statusDiv.innerText = 'Error: ' + (data.message || 'Authentication failed');
                    }
                }
            } catch(e) {
                if (statusDiv) {
                    statusDiv.style.display = 'block';
                    statusDiv.style.background = 'rgba(255,0,85,0.2)';
                    statusDiv.style.color = 'var(--critical-crimson)';
                    statusDiv.innerText = 'Network error: ' + e.message;
                }
            }
        }

        async function doAdminLogout() {
            try {
                await fetch('/api/auth/logout', { method: 'POST' });
            } catch(e) {}
            localStorage.removeItem('cybergraph_admin_token');
            adminToken = '';
            closeAdminAuthModal();
            updateDeviceAccessSection();
            alert('Logged out from Administrator session.');
        }

        // Global Escape Key Accessibility Listener
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeAdminAuthModal();
                closeBlockHistoryModal();
                closeFirewallModal();
                closeInspector();
            }
        });

function initStaticEventListeners() {
    const btnOpenBlockHist = document.getElementById('btn-open-block-history');
    if (btnOpenBlockHist) btnOpenBlockHist.addEventListener('click', openBlockHistoryModal);

    const btnCloseAuthX = document.getElementById('btn-close-admin-auth-x');
    if (btnCloseAuthX) btnCloseAuthX.addEventListener('click', closeAdminAuthModal);

    const btnCloseAuth = document.getElementById('btn-close-admin-auth');
    if (btnCloseAuth) btnCloseAuth.addEventListener('click', closeAdminAuthModal);

    const btnDoLogin = document.getElementById('btn-do-admin-login');
    if (btnDoLogin) btnDoLogin.addEventListener('click', doAdminLogin);

    const btnDoLogout = document.getElementById('btn-admin-logout');
    if (btnDoLogout) btnDoLogout.addEventListener('click', doAdminLogout);


    const btnAdminAuth = document.getElementById('btn-admin-auth');
    if (btnAdminAuth) btnAdminAuth.addEventListener('click', openAdminAuthModal);

    const btnExportReport = document.getElementById('btn-export-report');
    if (btnExportReport) btnExportReport.addEventListener('click', () => window.open('/api/export_report'));

    const btnThemeToggle = document.getElementById('btn-theme-toggle');
    if (btnThemeToggle) {
        if (localStorage.getItem('cybergraph_theme') === 'light') {
            document.body.classList.add('light-theme');
        }
        btnThemeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('cybergraph_theme', isLight ? 'light' : 'dark');
        });
    }

    const btnZoomIn = document.getElementById('btn-zoom-in');
    if (btnZoomIn) btnZoomIn.addEventListener('click', () => zoomGraph(1.2));

    const btnZoomOut = document.getElementById('btn-zoom-out');
    if (btnZoomOut) btnZoomOut.addEventListener('click', () => zoomGraph(0.8));

    const btnZoomReset = document.getElementById('btn-zoom-reset');
    if (btnZoomReset) btnZoomReset.addEventListener('click', resetGraphView);

    const alertSearch = document.getElementById('alert-search');
    if (alertSearch) alertSearch.addEventListener('input', renderFilteredAlerts);

    const alertFilter = document.getElementById('alert-level-filter');
    if (alertFilter) alertFilter.addEventListener('change', renderFilteredAlerts);

    const btnCloseInsp = document.getElementById('btn-close-inspector');
    if (btnCloseInsp) btnCloseInsp.addEventListener('click', closeInspector);

    const btnInspBlock = document.getElementById('insp-btn-block');
    if (btnInspBlock) btnInspBlock.addEventListener('click', inspectorBlockCurrentNode);

    const btnInspUnblock = document.getElementById('insp-btn-unblock');
    if (btnInspUnblock) btnInspUnblock.addEventListener('click', inspectorUnblockCurrentNode);

    const btnCloseFwModal = document.getElementById('btn-close-fw-modal');
    if (btnCloseFwModal) btnCloseFwModal.addEventListener('click', closeFirewallModal);

    const btnCancelFw = document.getElementById('btn-cancel-fw-block');
    if (btnCancelFw) btnCancelFw.addEventListener('click', closeFirewallModal);

    const btnDoRealBlock = document.getElementById('btn-do-real-block');
    if (btnDoRealBlock) btnDoRealBlock.addEventListener('click', confirmRealBlock);

    const btnCloseHistX = document.getElementById('btn-close-block-history-x');
    if (btnCloseHistX) btnCloseHistX.addEventListener('click', closeBlockHistoryModal);

    const btnCloseHist = document.getElementById('btn-close-block-history');
    if (btnCloseHist) btnCloseHist.addEventListener('click', closeBlockHistoryModal);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initStaticEventListeners);
} else {
    initStaticEventListeners();
}
