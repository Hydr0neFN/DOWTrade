if (window.location.pathname === '/') {
    setInterval(() => {
        fetch('/api/refresh').then(r => r.json()).then(data => {
            document.getElementById('pos-qty').innerText = 'Qty: ' + data.position.qty;
            document.getElementById('pos-side').innerText = 'Side: ' + data.position.side;
            document.getElementById('pos-avg').innerText = 'Avg Price: ' + data.position.avg_price;
            document.getElementById('pos-unrealized').innerText = 'Unrealized PnL: ' + data.position.unrealized;
            document.getElementById('pos-realized').innerText = 'Today Realized: ' + data.today_realized;
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
