function fmt(n) {
  if (n === null || n === undefined) return '-';
  return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
}
function fmtAge(sec) {
  if (sec === null || sec === undefined) return 'never';
  if (sec < 60) return sec + 's ago';
  if (sec < 3600) return Math.floor(sec/60) + 'm ago';
  return Math.floor(sec/3600) + 'h ago';
}

if (window.location.pathname === '/') {
  const refresh = () => {
    fetch('/api/refresh').then(r => r.json()).then(data => {
      // Account
      if (data.account) {
        const ae = document.getElementById('acct-equity'); if (ae) ae.innerText = fmt(data.account.equity);
        const ap = document.getElementById('acct-day-pnl'); if (ap) {
          ap.innerText = fmt(data.account.day_pnl);
          ap.className = 'stat-value ' + (data.account.day_pnl >= 0 ? 'pnl-pos' : 'pnl-neg');
        }
        const at = document.getElementById('acct-trades'); if (at) at.innerText = data.account.trades;
        const ac = document.getElementById('acct-commission'); if (ac) ac.innerText = fmt(data.account.commission);
        const ao = document.getElementById('acct-asof'); if (ao) ao.innerText = 'As of ' + data.account.as_of;
      }
      // Position
      if (data.position) {
        const ps = document.getElementById('pos-side'); if (ps) ps.innerText = data.position.side;
        const pq = document.getElementById('pos-qty'); if (pq) pq.innerText = data.position.qty;
        const pa = document.getElementById('pos-avg'); if (pa) pa.innerText = data.position.avg_price;
        const pu = document.getElementById('pos-unrealized'); if (pu) pu.innerText = data.position.unrealized;
      }
      // Last decision
      const ld = data.last_decision;
      const ldDir = document.getElementById('last-decision-dir');
      const ldTime = document.getElementById('last-decision-time');
      if (ld && ldDir) {
        ldDir.innerText = ld.direction || '-';
        ldDir.className = 'stat-value ' + (ld.direction === 'LONG' ? 'pnl-pos' : ld.direction === 'SHORT' ? 'pnl-neg' : 'text-white');
      }
      if (ld && ldTime) {
        try { ldTime.innerText = new Date(Number(ld.bar_ts) * 1000).toLocaleTimeString(); } catch (e) { ldTime.innerText = ld.bar_ts; }
      }
      // Heartbeat
      const hb = data.heartbeat || {};
      const dot = document.getElementById('hb-dot');
      const txt = document.getElementById('hb-text');
      if (dot && txt) {
        const age = hb.age_sec;
        let cls = 'hb-bad';
        if (age !== null && age !== undefined) {
          if (age < 120) cls = 'hb-ok';
          else if (age < 600) cls = 'hb-warn';
        }
        dot.className = 'heartbeat-dot ' + cls;
        txt.innerText = 'poller: ' + fmtAge(age);
      }
    }).catch(() => {});
  };
  refresh();
  setInterval(refresh, 15000);
}

if (window.location.pathname === '/trades') {
  let currentPage = 1;
  function loadTrades(page) {
    fetch('/api/trades?page=' + page).then(r => r.json()).then(data => {
      const list = document.getElementById('trades-list');
      if (!list) return;
      list.innerHTML = '';
      data.trades.forEach(t => {
        const div = document.createElement('div');
        div.className = 'card mb-2';
        div.innerHTML = '<div class="card-body"><b>' + t.ts + '</b> ' + t.side + ' ' + t.qty + ' ' + t.symbol + ' @ ' + (t.fill_price || 'N/A') + '</div>';
        list.appendChild(div);
      });
      const pn = document.getElementById('page-num'); if (pn) pn.innerText = page;
    }).catch(() => {});
  }
  const prev = document.getElementById('prev-page');
  const next = document.getElementById('next-page');
  if (prev) prev.addEventListener('click', () => { if (currentPage > 1) { currentPage--; loadTrades(currentPage); } });
  if (next) next.addEventListener('click', () => { currentPage++; loadTrades(currentPage); });
  loadTrades(1);
}
