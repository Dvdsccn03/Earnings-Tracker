import streamlit as st
import yfinance as yf
import pandas as pd
import calendar
from datetime import datetime, date
import requests
from openai import OpenAI
from pathlib import Path
import numpy as np
import feedparser
from urllib.parse import quote
import json
import traceback



# --- Default CIKs for top 20 companies ---
DEFAULT_TICKER_CIK_MAP = {
    'AAPL': '0000320193', 'MSFT': '0000789019', 'GOOGL': '0001652044', 'AMZN': '0001018724',
    'META': '0001326801', 'TSLA': '0001318605', 'NVDA': '0001045810', 'JPM': '0000019617',
    'UNH': '0000731766', 'V': '0001403161', 'JNJ': '0000200406', 'XOM': '0000034088',
    'PG': '0000080424', 'MA': '0001141391', 'HD': '0000354950', 'BAC': '0000070858',
    'LLY': '0000059478', 'KO': '0000021344', 'PFE': '0000078003', 'PEP': '0000077476'
}

# --- Helper: Get earnings dates ---
@st.cache_data(ttl=3600)
def get_earnings_dates(ticker_symbol):
    tk = yf.Ticker(ticker_symbol)
    df = tk.get_earnings_dates(limit=5).reset_index()
    df['Earnings Date'] = pd.to_datetime(df['Earnings Date']).dt.tz_localize(None)
    df = df[df['Event Type'] == 'Earnings']

    cal = tk.calendar
    earnings_date = cal.get('Earnings Date', [])
    next_date = earnings_date[0] if isinstance(earnings_date, list) else earnings_date

    if pd.notnull(next_date):
        next_date = pd.to_datetime(next_date).tz_localize(None)
        df = pd.concat([
            df,
            pd.DataFrame([{ 'Earnings Date': next_date, 'Event Type': 'Earnings', 'Ticker': ticker_symbol }])
        ], ignore_index=True)

    df['Ticker'] = ticker_symbol
    return df[['Earnings Date', 'Ticker']]

# --- Fetch & cache 10-K/10-Q ---
@st.cache_data(ttl=86400)
def fetch_latest_filing(cik, user_agent="MyAppName/1.0 (davidesaccone@outlook.com)"):
    cik_int = int(cik)
    cik10 = str(cik_int).zfill(10)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = requests.get(submissions_url, headers=headers, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    for form, acc in zip(recent["form"], recent["accessionNumber"]):
        if form in ("10-K", "10-Q"):
            acc_nodash = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc}.txt"
            txt_resp = requests.get(filing_url, headers=headers, timeout=60)
            txt_resp.raise_for_status()
            return txt_resp.text, filing_url

    raise ValueError("No recent 10-K or 10-Q found for this CIK.")

def compute_var_es(returns, alpha=0.95):
    cutoff = np.percentile(returns, 100 * (1 - alpha))
    es = returns[returns <= cutoff].mean()
    return cutoff, es

# --- Streamlit App ---
# Title and LinkedIn
col_i, col_t = st.columns([3,1])
with col_i: st.header('Earnings Calendar 📆')
with col_t: st.markdown("""Created by 
    <a href="https://www.linkedin.com/in/davide-saccone/" target="_blank">
        <button style="background-color: #262730; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
            Davide Saccone
        </button>
    </a>
    """, unsafe_allow_html=True)

# --- Toggle AI features ---
use_ai = st.checkbox("Enable AI features", value=False)

# --- Handle ticker clicks ---
params = st.query_params
ticker = params.get("ticker", None)

if ticker:
    st.subheader(f"{ticker} — Outlook")

    tk = yf.Ticker(ticker)
    info = tk.info

    description = info.get("longBusinessSummary", "No description available.")
    st.write(description)


    # company profile
    raw_profile = {
        "Industry": info.get("industry"),
        "Sector": info.get("sector"),
        "Employees": info.get('fullTimeEmployees'),
        "Market Cap": info.get('marketCap'),
        "Currency": info.get("currency"),
        "Shares Outstanding": info.get('sharesOutstanding'),
        "Previous Close": info.get("previousClose"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
    }
    def fmt(v):
        if v is None: return None
        if isinstance(v, (int,)) and v >= 1000:
            return f"{v:,}"
        if isinstance(v, (float,)) and v >= 1000:
            return f"{int(v):,}"
        if isinstance(v, (float,)):
            return f"{v:,.2f}"
        return v

    # Key Financial Metrics
    left_metrics = {
        "Total Revenue": info.get('totalRevenue'),
        "Gross Profits": info.get('grossProfits'),
        "EBITDA": info.get('ebitda'),
        "Gross Margin (%)": info.get("grossMargins") * 100 if info.get("grossMargins") is not None else None,
        "EBITDA Margin (%)": info.get("ebitdaMargins") * 100 if info.get("ebitdaMargins") is not None else None,
        "Profit Margin (%)": info.get("profitMargins") * 100 if info.get("profitMargins") is not None else None,
        "Total Debt": info.get('totalDebt'),
        "Debt/Equity (%)": info.get("debtToEquity"),
        "Operating Cashflow": info.get('operatingCashflow'),
        "Free Cashflow": info.get('freeCashflow')}


    right_metrics = {
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Trailing EPS": info.get("trailingEps"),
        "Forward EPS": info.get("forwardEps"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "EV/Revenue": info.get("enterpriseToRevenue"),
        "Price/Book": info.get("priceToBook"),      
        "Quick Ratio": info.get("quickRatio"),
        "ROA (%)": info.get("returnOnAssets") * 100 if info.get("returnOnAssets") is not None else None,
        "ROE (%)": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") is not None else None}

    st.markdown("#### Company Profile")
    prof_items = [(k, fmt(v)) for k,v in raw_profile.items() if v is not None]
    df_profile = pd.DataFrame(dict(prof_items), index=[0]).T.rename(columns={0: ticker})
    st.table(df_profile)

    left_items = [(k, fmt(v)) for k,v in left_metrics.items()]
    df_left = pd.DataFrame(dict(left_items), index=[0]).T.rename(columns={0: ticker})

    right_items = [(k, fmt(v)) for k,v in right_metrics.items()]
    df_right = pd.DataFrame(dict(right_items), index=[0]).T.rename(columns={0: ticker})

    st.markdown("#### Key Financial Metrics")
    coli, colw = st.columns(2)
    with coli:
        st.table(df_left)
    with colw:
        st.table(df_right)


    # Stock Summary
    # Fetch of daily close prices
    hist = tk.history(period='5y')['Close']
    rf = float(yf.download("^IRX", period='5y', interval='1d')['Close'].mean() / 100)  # Risk-free rate (IRX is 13-week T-bill)
    # Compute daily returns
    returns = hist.pct_change().dropna()
    # Annualize
    annual_return = returns.mean() * 252
    volatility = returns.std(ddof=1) * (252 ** 0.5)
    min_return = returns.min()
    max_return = returns.max()
    sharpe = (annual_return - rf) / volatility
    var95, es95 = compute_var_es(returns.values, alpha=0.95)
    # Build summary DataFrame
    summary_df = pd.DataFrame({
        " Ann. Return ": annual_return,
        "  Volatility  ": volatility,
        " Min Return ": min_return,
        " Max Return ": max_return,
        "   Sharpe   ": sharpe,
        "   95% VaR   ": var95,
        "   95% ES   ": es95
    }, index=[ticker])

    # Format and display
    st.subheader("Stock Information")
    st.table(
        summary_df.style
        .format({
            " Ann. Return ": "{:.2%}",
            "  Volatility  ": "{:.2%}",
            " Min Return ": "{:.2%}",
            " Max Return ": "{:.2%}",
            "   95% VaR   ": "{:.2%}",
            "   95% ES   ": "{:.2%}",
            "   Sharpe   ": "{:.2f}"
        })
    )




    # Determine last three past earnings dates
    df_ed = tk.get_earnings_dates(limit=5).reset_index()
    df_ed['Earnings Date'] = pd.to_datetime(df_ed['Earnings Date']).dt.tz_localize(None)
    df_ed = df_ed[df_ed['Event Type']=='Earnings']
    # Get last three past earnings, oldest first
    past = (
        df_ed[df_ed['Earnings Date'] < pd.Timestamp.now()]
        .sort_values('Earnings Date', ascending=False)
        .head(3)
    )
    table = past.copy()
    table.index = table['Earnings Date'].dt.strftime('%Y-%m-%d')
    table = table[['EPS Estimate', 'Reported EPS', 'Surprise(%)']]
    

    if past.empty:
        st.info("No past earnings dates available to plot a chart.")
    else:
        dates = pd.to_datetime(past['Earnings Date'].sort_values()).tolist()

        # Define window around most recent date
        last_ts = dates[-1]
        start = last_ts - pd.Timedelta(days=270)
        end   = datetime.today()

        price_hist  = tk.history(start=start, end=end)
        price       = price_hist['Close']
        rolling_vol = price.pct_change().rolling(window=5).std() * (252 ** 0.5)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price, name='Close Price', line=dict(color='lightblue', width=2)))
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Weekly Volatility', yaxis='y2', line=dict(color='mediumpurple', width=2)))

        # Add vertical lines using shapes instead of add_vline
        for d_ts in dates:
            fig.add_shape(
                type="line",
                x0=d_ts, x1=d_ts,
                y0=0, y1=1,
                yref="paper", xref="x",
                line=dict(color="white", dash="dash")
            )
            fig.add_annotation(
                x=d_ts, y=1.02, xref="x", yref="paper",
                text=d_ts.strftime("%Y-%m-%d"),
                showarrow=False,
                font=dict(size=10, color="white")
            )

        fig.update_layout(
            title=f"{ticker} Price & Weekly Volatility around earnings dates",
            xaxis_title="Date",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volatility", overlaying='y', side='right'),
            margin=dict(t=80)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.table(table)



    # --- News Sentiment Section ---
    # Only run when AI features are enabled
    if not use_ai:
        st.info("AI features are disabled. Please enable 'Enable AI features' to load financial summary.")
        st.stop()
    
    st.subheader(f"{ticker} — News Sentiment")
    client = st.session_state.get("client") or OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state.client = client

    # Fetch latest headlines from Google News RSS
    query = f"{ticker} stock"
    encoded_q = quote(query)
    feed_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:50]

# Collect just the titles
    headlines = [entry.title for entry in entries]
        # Build a single prompt to aggregate sentiment
    headlines_list = "".join(f"- {h}" for h in headlines)
    prompt = (
    f"Below are the most recent news for {ticker}:\n"f"{headlines_list}\n\n"
    "Based solely on these news, assign an overall sentiment score between -1.0 "
    "(very negative) and 1.0 (very positive) as an long/short equity hedge fund manager would do. Then provide a detailed paragraph explanation "
    "that summarizes the news, discusses common themes, any shifts in tone, and highlights conflicting signals or "
    "notable outliers. You should mention specifically the relevant news! Your response must be valid JSON with exactly two keys: "
    "`\"score\"` (a number) and `\"explanation\"` (a string).")

    with st.spinner("Analyzing overall sentiment..."):
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
            )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        overall_score = data.get("score")
        explanation = data.get("explanation")
    except Exception:
        overall_score = None
        explanation = content

    # Display results
    if overall_score is not None:
        st.metric(label="News Sentiment Score (-1 to 1)", value=overall_score)
    st.write(explanation)


    # --- SEC Filing Section ---
    st.subheader(f"{ticker} — Latest SEC Filing")

    # Autofill or input CIK
    if ticker in DEFAULT_TICKER_CIK_MAP:
        cik = DEFAULT_TICKER_CIK_MAP[ticker]
        st.info(f"CIK auto-filled for {ticker}: {cik}")
    else:
        cik_input = st.text_input("Enter the company's 10-digit CIK:", placeholder="0001318605")
        if not cik_input:
            st.info("Please enter a valid 10-digit CIK to proceed.")
            st.stop()
        cik = cik_input.strip().zfill(10)
        if not (cik.isdigit() and len(cik) == 10):
            st.error("CIK must be exactly 10 numeric digits.")
            st.stop()

    # Initialize OpenAI client
    client = st.session_state.get("client") or OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state.client = client

    # Initial setup: fetch filing and vector store
    if "setup_done" not in st.session_state or st.session_state.get("assistant_ticker") != ticker:
        try:
            with st.spinner("Fetching latest 10-K/10-Q..."):
                filing_text, filing_url = fetch_latest_filing(cik)
            st.success("Filing downloaded!")
            st.write("**EDGAR URL:**", f"[View on SEC.gov]({filing_url})")

            filepath = Path(f"{ticker}_filing.txt")
            filepath.write_text(filing_text, encoding="utf-8")

            # — Create the vector store (once) and upload only on first run for this ticker
            if st.session_state.get("vs_ticker") != ticker:
                # 1. create a new store
                vs = client.vector_stores.create(name=f"{ticker} Filing Store")
                st.session_state.vs_id     = vs.id
                st.session_state.vs_ticker = ticker

                # 2. **upload** your filing **only once** here
                with st.spinner("Connecting to OpenAI…"):
                    with filepath.open("rb") as file_obj:
                        client.vector_stores.file_batches.upload_and_poll(
                            vector_store_id=vs.id,
                            files=[file_obj]
                        )
            else:
                # reuse existing store (no re-upload)
                vs = client.vector_stores.retrieve(st.session_state.vs_id)


            with st.spinner("Creating AI assistant..."):
                assistant = client.beta.assistants.create(
                    name=f"{ticker} Analyst",
                    instructions="You are a long/short equity hedge fund manager.",
                    model= "gpt-4o-mini",
                    tools=[{"type": "file_search"}]
                )
                client.beta.assistants.update(
                    assistant_id=assistant.id,
                    tool_resources={"file_search": {"vector_store_ids": [vs.id]}}
                )
                msg_file = client.files.create(file=filepath.open("rb"), purpose="assistants")

            st.session_state.update({
                "setup_done": True,
                "assistant_id": assistant.id,
                "message_file_id": msg_file.id,
                "assistant_ticker": ticker
            })
        
        except Exception as e:
            st.warning("OpenAI API is not available. Please try again later.")
            st.stop()

    # Generate summary once
    if "summary" not in st.session_state:
        summary_prompt = (
    "You are a long/short equity hedge-fund manager. Produce a plain-ASCII executive summary of the attached SEC filing, obeying these exact rules:\n"
    "Write 4 detailed paragraphs, each with a single bullet point:\n"
    "1. Key Financial Metrics: write only in plain-ASCII a paragraph highlighting at least 5 key financial metrics with comparizon to previous quarter or year (with % changes).\n"
    "2. Trends: one concise paragraph on key trends.\n"
    "3. Risks: one concise paragraph on major risks.\n"
    "4. Opportunities: one concise paragraph on key opportunities.\n"
    "No markdown, no HTML, no italics, no merged words, no citations, no extra spaces, plain ASCII only."
)
        with st.spinner("Generating executive summary..."):
            thread = client.beta.threads.create(messages=[{
                "role": "user",
                "content": summary_prompt,
                "attachments": [{"file_id": st.session_state.message_file_id, "tools": [{"type": "file_search"}]}]
            }])
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=st.session_state.assistant_id)
            while run.status in ("queued", "in_progress"):
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            msgs = client.beta.threads.messages.list(thread_id=thread.id)
        
        raw = msgs.data[0].content[0].text.value
        st.session_state.summary = [line.strip() for line in raw.splitlines() if line.strip()]

    # Display summary
    st.markdown("### Executive Summary")
    for b in st.session_state.summary:
        st.markdown(f"- {b}")

    # Follow-up Q&A
    question = st.text_input("Ask any follow-up about this filing:")
    if question:
        followup_prompt = (
    "You are a forward-looking financial analyst. Reply in plain ASCII only, using numbered bullets for multiple points and plain paragraphs for explanations. "
    "No markdown, no HTML, no italics, no merged words, no citations, no extra spaces.\n\n"
    f"Question: {question}")

        with st.spinner("Generating answer..."):
            q_thread = client.beta.threads.create(messages=[{
                "role": "user",
                "content": followup_prompt,
                "attachments": [{"file_id": st.session_state.message_file_id, "tools": [{"type": "file_search"}]}]
            }])
            q_run = client.beta.threads.runs.create(thread_id=q_thread.id, assistant_id=st.session_state.assistant_id)
            while q_run.status in ("queued", "in_progress"):
                q_run = client.beta.threads.runs.retrieve(thread_id=q_thread.id, run_id=q_run.id)
            q_msgs = client.beta.threads.messages.list(thread_id=q_thread.id)
        
        answer = q_msgs.data[0].content[0].text.value
        st.markdown(f"**Answer:**\n\n{answer}", unsafe_allow_html=True)

    st.stop()




# --- Earnings Calendar UI ---
def main_calendar():
    default_top_20 = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'UNH', 'V',
        'JNJ', 'XOM', 'PG', 'MA', 'HD',
        'BAC', 'LLY', 'KO', 'PFE', 'PEP'
    ]
    filter_input = st.text_input("Filter only few companies (e.g., AAPL, NVDA, ...)", "")
    filtered_input = [t.strip().upper() for t in filter_input.replace(",", " ").split() if t.strip()]
    filtered_tickers = [t for t in filtered_input if t in default_top_20] if filtered_input else default_top_20
    custom_input = st.text_input("Add new yfinance tickers (e.g., NFLX, ABNB, ...)", "")
    custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()] if custom_input else []
    tickers = list(set(filtered_tickers + custom_tickers))
    if not tickers:
        st.warning("Please enter at least one valid ticker.")
        return

    data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = pd.concat([data, get_earnings_dates(ticker)])
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")

    today = datetime.today()
    st.session_state.year = st.session_state.get('year', today.year)
    st.session_state.month = st.session_state.get('month', today.month)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.year = st.selectbox(
            "Year", list(range(today.year - 1, today.year + 2)),
            index=list(range(today.year - 1, today.year + 2)).index(st.session_state.year),
            key="year_select"
        )
    with col2:
        months = list(calendar.month_name)[1:]
        sel = st.selectbox("Month", months, index=st.session_state.month - 1, key="month_select")
        st.session_state.month = months.index(sel) + 1

    year, month = st.session_state.year, st.session_state.month
    cal_matrix = calendar.monthcalendar(year, month)
    today_date = today.date()
    weeks_html = []
    for week in cal_matrix:
        days_html = []
        for day in week:
            if day == 0:
                days_html.append("<td class='empty'></td>")
                continue
            d = date(year, month, day)
            earned = data[data['Earnings Date'].dt.date == d]['Ticker'].tolist()
            classes = ['cell']
            if d == today_date:
                classes.append('today')
            if earned:
                classes.append('earnings')
            day_html = f"<strong>{day}</strong>"
            if earned:
                links = [
                    f'<a href="?ticker={t}&year={year}&month={month}" target="_blank" rel="noopener noreferrer">{t}</a>'
                    for t in earned
                ]
                day_html += "<br>" + "<br>".join(links)
            days_html.append(f"<td class='{' '.join(classes)}'>{day_html}</td>")
        weeks_html.append("<tr>" + "".join(days_html) + "</tr>")

    calendar_html = f"""
<style>
  table.calendar {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  table.calendar th {{ background: #222; color: #fff; padding: 8px; }}
  table.calendar td {{ border: 1px solid #444; height: 100px; padding: 8px; vertical-align: top; font-size: 0.9em; }}
  .cell.today {{ background: #2a5298; color: #fff; }}
  .cell.earnings {{ background: #1e1e1e; color: #fff; }}
  .cell.today.earnings {{ background: #c62828; }}
  td.empty {{ background: #111; }}
  table.calendar td a {{ color: #64b5f6; text-decoration: none; font-weight: 600; }}
  table.calendar td a:hover {{ text-decoration: underline; }}
</style>
<table class='calendar'>
  <thead><tr>{''.join(f'<th>{d}</th>' for d in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])}</tr></thead>
  <tbody>{''.join(weeks_html)}</tbody>
</table>
"""
    st.markdown(f"### {calendar.month_name[month]} {year}", unsafe_allow_html=True)
    st.markdown(calendar_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main_calendar()






