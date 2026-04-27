if (window.location.pathname === '/') {
    function fmt(n) { return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}); }
    setInterval(() => {
        fetch('/api/refresh').then(r => r.json()).then(data => {
            // Position card
            document.getElementById('pos-qty').innerText = 'Qty: ' + data.position.qty;
            document.getElementById('pos-side').innerText = 'Side: ' + data.position.side;
            document.getElementById('pos-avg').innerText = 'Avg Price: ' + data.position.avg_price;
            document.getElementById('pos-unrealized').innerText = 'Unrealized PnL: ' + data.position.unrealized;
            document.getElementById('pos-realized').innerText = 'Today Realized: ' + data.today_realized;
            // Account card
            if (data.account) {
                document.getElementById('acct-equity').innerText = 'Equity: ' + fmt(data.account.equity);
                document.getElementById('acct-day-pnl').innerText = 'Day P&L: ' + fmt(data.account.day_pnl);
                document.getElementById('acct-trades').innerText = 'Trades today: ' + data.account.trades;
                document.getElementById('acct-commission').innerText = 'Commission: ' + fmt(data.account.commission);
                document.getElementById('acct-asof').innerHTML = '<small>As of ' + data.account.as_of + '</small>';
            }
            // Last decision badge
            const ld = data.last_decision;
            const el = document.getElementById('last-decision');
            if (el && ld) {
                const ts = new Date(ld.bar_ts * 1000).toLocaleTimeString();
                el.innerText = 'Last: ' + (ld.direction || '—') + ' @ ' + ts;
                el.style.color = ld.direction === 'LONG' ? '#4caf50' : ld.direction === 'SHORT' ? '#f44336' : '#aaa';
            }
        });
    }, 15000);
}

if (window.location.pathname === '/trades') {
    let currentPage = 1;
    function loadTrades(page) {
        fetch('/api/trades?page=' + page).then(r => r.json()).then(data => {
            const list = document.getElementById('trades-list');
            list.innerHTML = '';
            data.trades.forEach(t => {
                const div = document.createElement('div');
                div.innerHTML = `<b>${t.ts}</b> ${t.side} ${t.qty} ${t.symbol} @ ${t.fill_price || 'N/A'}`;
                const ul = document.createElement('ul');
                t.llm_calls.forEach(lc => {
                    const li = document.createElement('li');
                    li.textContent = `${lc.model}: ${lc.raw_response}`;
                    ul.appendChild(li);
                });
                div.appendChild(ul);
                list.appendChild(div);
            });
            document.getElementById('page-num').innerText = page;
        });
    }
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadTrades(currentPage);
        }
    });
    document.getElementById('next-page').addEventListener('click', () => {
        currentPage++;
        loadTrades(currentPage);
    });
    loadTrades(1);
}
