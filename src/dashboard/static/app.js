function fmt(n) {
  if (n === null || n === undefined) return '-';
  return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
}
function fmtAge(sec) {
  if (sec === null || sec === undefined) return T('js.never');
  if (sec < 60) return T('js.ago_s', {n: sec});
  if (sec < 3600) return T('js.ago_m', {n: Math.floor(sec/60)});
  return T('js.ago_h', {n: Math.floor(sec/3600)});
}
const $ = id => document.getElementById(id);
const setText = (id, val) => { const el = $(id); if (el) el.innerText = val; };

// ── Heartbeat pill (navbar, every page) ──────────────────────────────────────
// Green under 2 min, amber under 10, red past that: the poller writes on a
// 15-minute bar but touches the file far more often than that.
function paintHeartbeat(hb) {
  const pill = $('hbPill'), txt = $('hbText');
  if (!pill || !txt) return;
  const age = (hb || {}).age_sec;
  let cls = 'hb-bad';
  if (age !== null && age !== undefined) {
    if (age < 120) cls = 'hb-ok';
    else if (age < 600) cls = 'hb-warn';
  }
  pill.className = 'hb-pill ms-3 ' + cls;
  txt.innerText = T('js.poller') + ' ' + fmtAge(age);
}

// ── Overview live refresh ────────────────────────────────────────────────────
if (window.location.pathname === '/') {
  const refresh = () => {
    fetch('/api/refresh').then(r => r.json()).then(data => {
      if (data.account) {
        setText('acct-equity', fmt(data.account.equity));
        const ap = $('acct-day-pnl');
        if (ap) {
          ap.innerText = fmt(data.account.day_pnl);
          ap.className = 'stat-value mono ' + (data.account.day_pnl >= 0 ? 'pnl-pos' : 'pnl-neg');
        }
        setText('acct-trades', data.account.trades);
        setText('acct-commission', fmt(data.account.commission));
        setText('acct-asof', data.account.as_of);
      }
      if (data.position) {
        setText('pos-side', data.position.side);
        setText('pos-qty', data.position.qty);
        setText('pos-avg', data.position.avg_price);
        setText('pos-unrealized', data.position.unrealized);
      }
      const ld = data.last_decision, ldDir = $('last-decision-dir');
      if (ld && ldDir) {
        ldDir.innerText = ld.direction || '-';
        ldDir.className = 'stat-value ' + (ld.direction === 'LONG' ? 'pnl-pos'
                                         : ld.direction === 'SHORT' ? 'pnl-neg' : 'text-white');
      }
      if (ld) {
        let when = ld.bar_ts;
        try { when = new Date(Number(ld.bar_ts) * 1000).toLocaleTimeString(); } catch (e) { /* raw ts */ }
        setText('last-decision-time', when);
      }
      paintHeartbeat(data.heartbeat);
    }).catch(() => {});
  };
  refresh();
  setInterval(refresh, 15000);

  // Equity curve — inline here rather than on a page of its own.
  fetch('/api/equity').then(r => r.json()).then(data => {
    const cv = $('equityChart');
    if (!cv) return;
    if (!data.length) { const e = $('equityEmpty'); if (e) e.hidden = false; return; }
    const ctx = cv.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 280);
    grad.addColorStop(0, 'rgba(31,111,235,.35)');
    grad.addColorStop(1, 'rgba(31,111,235,0)');

    // High-water mark; the shaded gap down to the equity line is the drawdown.
    let peak = -Infinity;
    const values = data.map(d => d.balance);
    const hwm = values.map(v => { peak = Math.max(peak, v); return peak; });

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => d.ts),
        datasets: [
          { label: T('js.equity'), data: values, borderColor: '#1f6feb', backgroundColor: grad,
            borderWidth: 2, pointRadius: data.length > 20 ? 0 : 3, fill: true, tension: .3, order: 1 },
          { label: 'High-Water Mark', data: hwm, borderColor: 'rgba(210,153,34,.55)',
            borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, fill: '-1',
            backgroundColor: 'rgba(248,81,73,.08)', tension: .3, order: 2 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#121824', borderColor: '#26324d', borderWidth: 1,
            callbacks: {
              label: c => {
                if (c.datasetIndex === 1) return T('js.peak') + ': ' + fmt(c.parsed.y);
                const dd = c.parsed.y - hwm[c.dataIndex];
                const out = [T('js.equity') + ': ' + fmt(c.parsed.y)];
                if (dd < -0.01) out.push(T('js.drawdown') + ': ' + (dd / hwm[c.dataIndex] * 100).toFixed(2) + '%');
                return out;
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#9da7b3', maxTicksLimit: 8, maxRotation: 0 }, grid: { color: 'rgba(38,50,77,.5)' } },
          y: { ticks: { color: '#9da7b3', callback: v => '$' + v.toLocaleString('en-US') }, grid: { color: 'rgba(38,50,77,.5)' } },
        }
      }
    });
  }).catch(() => {});
}

// Other pages still want the heartbeat pill to be live.
if (window.location.pathname !== '/') {
  const beat = () => fetch('/api/refresh').then(r => r.json())
                       .then(d => paintHeartbeat(d.heartbeat)).catch(() => {});
  beat();
  setInterval(beat, 30000);
}
