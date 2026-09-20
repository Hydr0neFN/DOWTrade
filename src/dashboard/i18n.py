"""Bilingual (English / Traditional Chinese) strings for both trading dashboards.

Keep this file byte-identical in ~/trader/dashboard and
DOWTrade/trading-bot/src/dashboard — it is the single source of UI wording.

Language is chosen per request: an explicit choice (cookie, set by the navbar
toggle) wins, otherwise the browser's Accept-Language, which follows the OS
language setting. Templates call the three helpers registered as Jinja globals:

    t(key, **fmt)   plain text in the active language — title=, tooltips, sentences
    L(key)          zh with the English term inline after it, dimmed
    Lb(key)         zh with the English term stacked underneath (tight columns)

In English mode L and Lb are just the English string: the gloss only exists to
help a zh reader map a label back to the API/log vocabulary, which is English.

zh-TW wording follows Taiwanese finance convention (損益 / 部位 / 未實現 /
委託 / 成交 / 權益 / 停損), not mainland vocabulary.
"""

from markupsafe import Markup, escape

DEFAULT_LANG = "en"
LANGS = ("en", "zh")
COOKIE = "dash_lang"

# key: (English, 繁體中文). A zh value of None means "this term stays English"
# — model names, status tokens and anything that must match a log or JSON field.
STRINGS = {
    # ── chrome ───────────────────────────────────────────────────────────────
    "nav.overview":          ("Overview", "總覽"),
    "nav.history":           ("History", "交易歷史"),
    "nav.decisions":         ("Decisions", "決策紀錄"),
    "lang.toggle_title":     ("Switch language", "切換語言"),
    "footer.trader":         ("Paper trading only", "僅模擬交易"),
    "footer.dow":            ("Simulated fills only", "僅模擬成交"),
    "footer.dow_mym":        ("MYM futures", "MYM 期貨"),
    "footer.dow_safety":     ("execution gated in Python, not by an LLM",
                              "執行由 Python 把關，非由 LLM 決定"),
    "footer.models":         ("live models", "實際使用模型"),
    "footer.models_none":    ("no model activity logged yet", "尚無模型呼叫紀錄"),

    # ── stat cards ───────────────────────────────────────────────────────────
    "stat.portfolio_value":  ("Portfolio Value", "投組價值"),
    "stat.total_pnl":        ("Total P&L", "總損益"),
    "stat.day_pnl":          ("Day P&L", "當日損益"),
    "stat.last_run":         ("Last Run", "最新執行"),
    "stat.equity":           ("Equity", "帳戶權益"),
    "stat.position":         ("Position", "持有部位"),
    "stat.last_decision":    ("Last Decision", "最新決策"),
    "sub.cash":              ("{x} cash", "現金 {x}"),
    "sub.net_fees":          ("{net} net · {fees} sim fees", "淨額 {net} · 模擬費用 {fees}"),
    "sub.open_positions":    ("{n} open position(s)", "{n} 個未平倉部位"),
    "sub.approved":          ("{n} approved", "{n} 筆通過"),
    "sub.vetoed":            ("{n} vetoed", "{n} 筆否決"),
    "sub.trades_comm":       ("{n} trade(s) · {x} comm", "{n} 筆交易 · 手續費 {x}"),
    "sub.as_of":             ("as of {date}", "截至 {date}"),
    "sub.avg_upnl":          ("avg {price} · uPnL {x}", "均價 {price} · 未實現 {x}"),

    # ── section headers ──────────────────────────────────────────────────────
    "sec.portfolio_chart":   ("Portfolio Value Over Time", "投資組合價值走勢"),
    "sec.open_positions":    ("Open Positions", "未平倉部位"),
    "sec.equity_curve":      ("Equity Curve", "權益曲線"),
    "sec.equity_curve_hint": ("last 30 sessions", "近 30 個交易日"),
    "sec.recent_orders":     ("Recent Orders", "近期委託"),
    "sec.recent_orders_hint":("last {n}", "最近 {n} 筆"),
    "sec.trade_history":     ("Trade History", "交易歷史紀錄"),
    "sec.decision_chain":    ("Decision Chain", "決策歷程"),
    "sec.daily_journal":     ("Daily Journal", "每日交易日誌"),

    # ── table columns ────────────────────────────────────────────────────────
    "col.ticker":            ("Ticker", "標的"),
    "col.side":              ("Side", "買賣"),
    "col.qty":               ("Qty", "數量"),
    "col.avg_entry":         ("Avg Entry", "進場均價"),
    "col.current":           ("Current", "現價"),
    "col.trend_7d":          ("Trend 7d", "7日走勢"),
    "col.market_value":      ("Market Value", "市值"),
    "col.allocation":        ("Allocation", "部位佔比"),
    "col.unrealized_pnl":    ("Unrealized P&L", "未實現損益"),
    "col.time":              ("Time", "時間"),
    "col.action":            ("Action", "動作"),
    "col.verdict":           ("Verdict", "審核"),
    "col.price":             ("Price", "價格"),
    "col.confidence":        ("Conf.", "信心度"),
    "col.fill":              ("Fill", "成交價"),
    "col.symbol":            ("Symbol", "商品代碼"),
    "col.status":            ("Status", "狀態"),

    # ── filters ──────────────────────────────────────────────────────────────
    "filter.run":            ("Run", "執行批次"),
    "filter.ticker":         ("Ticker", "標的"),
    "filter.action":         ("Action", "動作"),
    "filter.type":           ("Type", "類型"),
    "filter.direction":      ("Direction", "方向"),
    "filter.outcome":        ("Outcome", "審查結果"),
    "filter.date":           ("Date", "日期"),
    "filter.from":           ("From", "起始日"),
    "filter.to":             ("To", "結束日"),
    "filter.all":            ("All", "全部"),
    "filter.apply":          ("Filter", "篩選"),
    "filter.reset":          ("Reset", "重設"),
    "filter.actions_only":   ("Actions only", "僅顯示動作"),
    "filter.buy_analysis":   ("Buy Analysis", "買進分析"),
    "filter.exit_review":    ("Exit Review", "出場審查"),
    "filter.approved":       ("Approved", "已通過"),
    "filter.rejected":       ("Blocked or flagged", "遭阻擋或標記"),
    "filter.blocked":        ("Blocked", "遭阻擋"),
    "filter.ds_flagged":     ("Risk audit flagged (advisory)", "風控審核標記（僅建議）"),
    "filter.disagree":       ("Disagreements", "意見分歧"),
    "filter.count":          ("{shown} of {total} decision(s)", "共 {total} 筆決策，顯示 {shown} 筆"),
    "filter.records":        ("{n} record(s)", "{n} 筆紀錄"),
    "filter.records_capped": ("showing {shown} of {total} matching record(s)",
                              "符合 {total} 筆，顯示前 {shown} 筆"),

    # ── empty / hint / error ─────────────────────────────────────────────────
    "empty.no_positions":    ("No open positions.", "目前無未平倉部位。"),
    "empty.no_orders":       ("No orders yet.", "尚無任何委託紀錄。"),
    "empty.no_trades_filter":("No trades match your filter.", "查無符合篩選條件的交易紀錄。"),
    "empty.no_decisions":    ("No decisions yet.", "尚無任何決策紀錄。"),
    "empty.no_decision_data":("No decision data yet — appears after the first completed run.",
                              "尚無決策資料 — 首次執行完成後即會顯示。"),
    "empty.no_equity":       ("No equity history yet.", "尚無權益歷史資料。"),
    "empty.no_history":      ("No history yet — data appears after the first completed trading run.",
                              "尚無歷史資料 — 首次交易執行完成後即會顯示。"),
    "hint.tap_row":          ("Tap a row for the full decision chain", "點擊任一列可查看完整決策歷程"),
    "hint.select_date":      ("Select a date.", "請選擇日期。"),
    "hint.loading":          ("Loading…", "載入中…"),
    "hint.journal_empty":    ("(empty)", "（空白）"),
    "hint.journal_fail":     ("Could not load this journal entry.", "無法載入此日誌內容。"),
    "err.alpaca":            ("Alpaca API error: {msg}", "Alpaca API 錯誤：{msg}"),

    # ── decision card labels ─────────────────────────────────────────────────
    "lbl.snapshot":          ("Snapshot", "即時概況"),
    "lbl.market_data":       ("Market Data", "市場行情"),
    "lbl.position":          ("Position", "部位資訊"),
    "lbl.entry":             ("Entry", "進場價"),
    "lbl.stop":              ("Stop", "停損價"),
    "lbl.current":           ("Current", "現價"),
    "lbl.confidence":        ("Confidence", "信心度"),
    "lbl.days_held":         ("Days Held", "持有天數"),
    "lbl.model":             ("Model", "模型"),
    "lbl.day_chg":           ("Day Chg", "當日漲跌"),
    "lbl.volume":            ("Volume", "成交量"),
    "lbl.unreal_pnl":        ("Unreal P&L", "未實現損益"),
    "lbl.exit_reasoning":    ("Exit Reasoning", "出場理由"),
    "lbl.haiku_structural":  ("Haiku Structural", "Haiku 結構分析"),
    "lbl.haiku_justification":("Haiku Justification", "Haiku 審核理由"),
    "lbl.gemini_reasoning":  ("Gemini Reasoning", "Gemini 分析推論"),
    "lbl.gemini_execution":  ("Gemini Execution", "Gemini 執行策略"),
    "lbl.hf_sentiment":      ("Sentiment", "情緒指標"),
    "lbl.ds_risk_audit":     ("Risk Audit (advisory)", "風控審核（僅建議）"),
    "outcome.blocked":       ("BLOCKED", "遭阻擋"),
    "outcome.ds_flagged":    ("RISK AUDIT FLAGGED", "風控審核標記"),
    "outcome.ds_flagged_note":("advisory only — this did not stop the trade",
                              "僅供參考，並未阻擋此筆交易"),
    "lbl.final_action":      ("Final Action", "最終動作"),
    "lbl.final_outcome":     ("Final Outcome", "最終裁定"),
    "lbl.market_at_run":     ("Market data at run", "執行當下市場行情"),
    "lbl.ai_analysis":       ("AI analysis", "AI 綜合分析"),
    "lbl.risk_gate":         ("Risk gate — Claude Haiku", "風控關卡 — Claude Haiku"),
    "lbl.settlement":        ("Settlement", "結算資訊"),
    "lbl.reasoning":         ("Reasoning", "分析推論"),
    "lbl.veto_reasons":      ("Veto reasons", "否決原因"),
    "lbl.violations":        ("Violations", "風控違規"),

    # ── heartbeat ────────────────────────────────────────────────────────────
    "hb.title":              ("yfinance poller heartbeat", "yfinance 輪詢心跳狀態"),

    # ── modal field labels ───────────────────────────────────────────────────
    "modal.trade_detail":    ("Trade Detail", "交易明細"),
}

# Strings the browser needs. Sent to the page as a JSON blob so app.js and the
# chart callbacks read the same table the templates do.
JS_KEYS = (
    "hint.loading", "hint.journal_empty", "hint.journal_fail", "hint.select_date",
    "filter.count", "modal.trade_detail",
    "lbl.market_at_run", "lbl.ai_analysis", "lbl.risk_gate", "lbl.settlement",
    "lbl.reasoning", "lbl.veto_reasons", "lbl.confidence", "lbl.model",
    "col.qty", "col.price", "col.time", "col.action", "col.ticker",
)

# Extra JS-only strings with no template counterpart.
JS_EXTRA = {
    "js.value":      ("Value", "投組價值"),
    "js.equity":     ("Equity", "權益數"),
    "js.peak":       ("Peak", "歷史高點"),
    "js.drawdown":   ("Drawdown", "資金回撤"),
    "js.poller":     ("poller", "輪詢"),
    "js.never":      ("never", "從未"),
    "js.ago_s":      ("{n}s ago", "{n} 秒前"),
    "js.ago_m":      ("{n}m ago", "{n} 分前"),
    "js.ago_h":      ("{n}h ago", "{n} 小時前"),
    "js.recommendation": ("Recommendation", "建議動作"),
    "js.verdict":    ("Verdict", "審核結果"),
    "js.justification": ("Justification", "審核理由"),
    "js.sentiment":  ("Sentiment", "情緒指標"),
    "js.detail_parse_fail": ("Could not open this record.", "無法開啟此筆紀錄。"),
}


def normalize(code):
    """Any language tag → 'en' or 'zh', or None if it is neither."""
    if not code:
        return None
    code = code.strip().lower()
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    return None


def from_accept_language(header):
    """First understood language in an Accept-Language header, honouring q-values.

    'zh-TW,zh;q=0.9,en-US;q=0.8' → 'zh'. Unparseable or unknown → None, and the
    caller falls back to DEFAULT_LANG.
    """
    if not header:
        return None
    entries = []
    for i, part in enumerate(header.split(",")):
        bits = part.split(";")
        tag = bits[0].strip()
        q = 1.0
        for b in bits[1:]:
            b = b.strip()
            if b.startswith("q="):
                try:
                    q = float(b[2:])
                except ValueError:
                    q = 0.0
        # i keeps the header's own order as the tie-break, which is what
        # browsers mean when they omit q entirely.
        entries.append((-q, i, tag))
    for _, _, tag in sorted(entries):
        lang = normalize(tag)
        if lang:
            return lang
    return None


def choose(cookie_val=None, query_val=None, accept_language=None):
    """Explicit choice beats the browser. Query beats cookie so a shared link
    can pin a language; the caller persists that choice into the cookie."""
    for candidate in (query_val, cookie_val):
        lang = normalize(candidate)
        if lang:
            return lang
    return from_accept_language(accept_language) or DEFAULT_LANG


def _raw(key, lang):
    pair = STRINGS.get(key) or JS_EXTRA.get(key)
    if pair is None:
        # A missing key should be obvious in the page, not silently blank.
        return key
    en, zh = pair
    return (zh if lang == "zh" else en) or en


def make_helpers(lang):
    """The three Jinja globals, bound to one request's language."""

    def t(key, **fmt):
        s = _raw(key, lang)
        return s.format(**fmt) if fmt else s

    def _glossed(key, cls, fmt):
        text = t(key, **fmt)
        if lang != "zh":
            return Markup(escape(text))
        english = STRINGS.get(key, (key, key))[0]
        if fmt:
            english = english.format(**fmt)
        # Nothing to gloss when the zh and English forms are the same string
        # (model names, tickers), so skip the span rather than repeat it.
        if english == text:
            return Markup(escape(text))
        return Markup(f'{escape(text)}<span class="{cls}">{escape(english)}</span>')

    def L(key, **fmt):
        """zh with the English term inline after it."""
        return _glossed(key, "gloss-inline", fmt)

    def Lb(key, **fmt):
        """zh with the English term stacked underneath — for tight columns."""
        return _glossed(key, "gloss-block", fmt)

    return {"t": t, "L": L, "Lb": Lb, "lang": lang}


def js_table(lang):
    """The subset of strings the browser needs, as {key: text}."""
    keys = list(JS_KEYS) + list(JS_EXTRA)
    return {k: _raw(k, lang) for k in keys}
