import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import feedparser
import google.generativeai as genai
import re
import time
import plotly.graph_objects as go
from urllib.parse import quote
from datetime import datetime

# =========================================================
# 0. 기본 설정 & API 키
# =========================================================
# 실제 배포 시에는 st.secrets를 사용하는 것이 좋습니다.
FOOTBALL_API_KEY = "d69efb6a4cc6101411098c976504bed46179c0b1655af0c378b4df7d8f9af5b7"
GEMINI_API_KEY = "AIzaSyBLF5RDLC0WRtMTA3bpK4teb1tKN4J2yqI"

st.set_page_config(
    page_title="FIELD GRID – 구단 운영 AI 리포트",
    page_icon="⚽",
    layout="wide"
)

# =========================================================
# 1. 팀·리그·티커 정보 (기존 데이터 유지)
# =========================================================
SEASON_IDS_BY_LEAGUE = {
    "England Premier League": [161, 1625, 2012, 4759, 6135, 7704, 9660, 12325, 15050],
    "France Ligue 1":          [180, 1508, 2392, 4505, 6019, 7500, 9674, 12337, 14932],
    "Germany Bundesliga":      [177, 1636, 4392, 4673, 6192, 7664, 9655, 12529, 14968],
    "Italy Serie A":           [182, 1680, 2588, 4889, 6198, 7608, 9697, 12530, 15068],
    "Netherlands Eredivisie":  [178, 1585, 2272, 4746, 5951, 7482, 9653, 12322, 14936],
    "Scotland Premiership":    [164, 1600, 2361, 4478, 5992, 7494, 9636, 12455, 15000]
}

TEAM_IDS = {
    "England Premier League": 149,
    "France Ligue 1": 57,
    "Germany Bundesliga": 33,
    "Italy Serie A": 90,
    "Netherlands Eredivisie": 120,
    "Scotland Premiership": 23
}

TEAM_NAME_ALIASES = {
    "맨유": "Manchester United",
    "맨체스터유나이티드": "Manchester United",
    "맨체스터 유나이티드": "Manchester United",
    "도르트문트": "Borussia Dortmund",
    "보루시아 도르트문트": "Borussia Dortmund",
    "BVB": "Borussia Dortmund",
    "리옹": "Lyon",
    "유벤투스": "Juventus",
    "아약스": "Ajax",
    "셀틱": "Celtic"
}

TEAM_TO_LEAGUE = {
    "Manchester United": "England Premier League",
    "Lyon": "France Ligue 1",
    "Borussia Dortmund": "Germany Bundesliga",
    "Juventus": "Italy Serie A",
    "Ajax": "Netherlands Eredivisie",
    "Celtic": "Scotland Premiership"
}

STOCK_TICKERS = {
    "Manchester United": "MANU",
    "Borussia Dortmund": "BVB.DE",
    "Juventus": "JUVE.MI",
    "Ajax": "AJAX.AS",
    "Celtic": "CCP.L",
    "Lyon": "OLG.PA",
}

MARKET_INDEX_TICKERS = {
    "England Premier League": "^FTSE",
    "Germany Bundesliga": "^GDAXI",
    "Netherlands Eredivisie": "^AEX",
    "France Ligue 1": "^FCHI",
    "Scotland Premiership": "^FTSE",
    "Italy Serie A": "FTSEMIB.MI"
}
SEASON_DATE_RANGE = {
    "England Premier League": ("08-01", "06-01"),
    "France Ligue 1": ("08-01", "06-01"),
    "Germany Bundesliga": ("08-01", "06-01"),
    "Italy Serie A": ("08-01", "06-01"),
    "Netherlands Eredivisie": ("08-01", "06-01"),
    "Scotland Premiership": ("08-01", "06-01"),
}


# =========================================================
# 2. 유틸 함수 (팀명 정규화, 기간 파싱)
# =========================================================
def normalize_team_name(query: str):
    if not query:
        return None
    q = query.lower()
    for alias, canonical in TEAM_NAME_ALIASES.items():
        if alias.lower() in q:
            return canonical
    for official in TEAM_TO_LEAGUE.keys():
        if official.lower() in q:
            return official
    return None

def parse_period(query: str):
    season_word = re.search(r"(20\d{2})\s*시즌", query)
    if season_word:
        return ("season", int(season_word.group(1)))

    full = re.search(r"(\d{4})[./-](\d{2})[./-](\d{2})", query)
    if full:
        val = datetime.strptime(full.group(0).replace(".", "-"), "%Y-%m-%d")
        return ("date", val)

    ym = re.search(r"(\d{4})[./-](\d{2})", query)
    if ym:
        val = datetime.strptime(ym.group(0).replace(".", "-"), "%Y-%m")
        return ("month", val)

    year = re.search(r"(20\d{2})", query)
    if year:
        return ("season", int(year.group(1)))

    return (None, None)
    
def extract_season_year(query):
    match = re.search(r"(\d{4})\s*시즌", query)
    if match:
        return int(match.group(1))
    return None

# =========================================================
# [NEW] 뉴스 이벤트 타입 분류 + 카테고리화 + 주가 영향 분석
# =========================================================
NEWS_EVENT_RULES = [
    # (event_type, category, keywords)
    ("Transfer", "Squad & Transfers", ["transfer", "sign", "signed", "loan", "contract", "release", "deal", "이적", "영입", "임대", "계약", "방출"]),
    ("Injury", "Injuries & Fitness", ["injury", "injured", "out", "ruled out", "hamstring", "ankle", "knee", "surgery", "부상", "결장", "수술", "회복", "복귀"]),
    ("Manager/Coach", "Management & Governance", ["manager", "head coach", "coach", "sacked", "appointed", "resign", "director", "ceo", "감독", "경질", "선임", "사임", "단장", "ceo"]),
    ("Discipline/Legal", "Discipline/Legal", ["ban", "suspended", "fine", "appeal", "lawsuit", "court", "charge", "investigation", "징계", "출전정지", "벌금", "소송", "법원", "수사", "조사"]),
    ("Finance/Sponsor", "Finance & Commercial", ["sponsor", "sponsorship", "revenue", "profit", "loss", "earnings", "deal", "brand", "commercial", "파트너", "스폰서", "후원", "매출", "손실", "실적", "계약"]),
    ("Competition/Match", "Match/Competition", ["vs", "match", "fixture", "derby", "cup", "champions league", "europa", "kickoff", "경기", "대진", "컵", "챔피언스리그", "유로파", "킥오프"]),
    ("Fan/Community", "Fan/Community", ["fan", "supporter", "protest", "boycott", "ticket", "stadium", "community", "팬", "서포터", "시위", "보이콧", "티켓", "구장", "커뮤니티"]),
]

# =========================================================
# [NEW] 뉴스 카테고리 / 이벤트 타입 한글 라벨 매핑
# =========================================================
CATEGORY_KR_MAP = {
    "Squad & Transfers": "선수단 · 이적",
    "Injuries & Fitness": "부상 · 컨디션",
    "Match/Competition": "경기 · 대회",
    "Management & Governance": "경영 · 거버넌스",
    "Finance & Commercial": "재무 · 상업",
    "Fan/Community": "팬 · 커뮤니티",
    "Discipline/Legal": "징계 · 법적 이슈",
    "Other": "기타"
}

EVENT_TYPE_KR_MAP = {
    "Transfer": "이적",
    "Injury": "부상",
    "Competition/Match": "경기/대회",
    "Manager/Coach": "감독/코칭스태프",
    "Finance/Sponsor": "재무/스폰서",
    "Fan/Community": "팬 이슈",
    "Discipline/Legal": "징계/법적",
    "Other": "기타"
}

def classify_news_title(title: str):
    if not isinstance(title, str) or not title.strip():
        return ("Other", "Other")
    t = title.lower()
    for etype, cat, kws in NEWS_EVENT_RULES:
        for kw in kws:
            if kw.lower() in t:
                return (etype, cat)
    return ("Other", "Other")

def analyze_news_stock_impact(news_df: pd.DataFrame, stock: pd.DataFrame, market: pd.DataFrame):
    """
    뉴스 게시일(published_dt) 기준으로 5거래일 뒤 초과수익률(Abnormal Return)을 계산하고,
    카테고리별로 평균 영향도를 요약합니다.
    """
    if news_df is None or news_df.empty:
        return news_df, pd.DataFrame()

    if stock is None or stock.empty or market is None or market.empty:
        # 주가 데이터 없으면 분류만 제공
        out = news_df.copy()
        out[["event_type", "category"]] = out["title"].apply(lambda x: pd.Series(classify_news_title(x)))
        return out, pd.DataFrame()

    out = news_df.copy()
    out[["event_type", "category"]] = out["title"].apply(lambda x: pd.Series(classify_news_title(x)))

    # 날짜 정리
    out["event_date"] = pd.to_datetime(out.get("published_dt", out.get("published", "")), errors="coerce")
    out = out.dropna(subset=["event_date"])

    s = stock.copy()
    m = market.copy()
    s["date"] = pd.to_datetime(s["date"])
    m["date"] = pd.to_datetime(m["date"])
    s = s.sort_values("date")
    m = m.sort_values("date")
    out = out.sort_values("event_date")

    # 뉴스 직전 종가 (backward)
    s_prev = pd.merge_asof(out[["event_date"]], s[["date", "close"]], left_on="event_date", right_on="date", direction="backward")
    # 5거래일 후 종가 근사 (forward + 7D tolerance)
    s_post = pd.merge_asof(out[["event_date"]], s[["date", "close"]], left_on="event_date", right_on="date",
                           direction="forward", tolerance=pd.Timedelta("7D"))

    m_prev = pd.merge_asof(out[["event_date"]], m[["date", "close"]], left_on="event_date", right_on="date", direction="backward")
    m_post = pd.merge_asof(out[["event_date"]], m[["date", "close"]], left_on="event_date", right_on="date",
                           direction="forward", tolerance=pd.Timedelta("7D"))

    out["prev_close"] = s_prev["close"]
    out["post_close"] = s_post["close"]
    out["m_prev"] = m_prev["close"]
    out["m_post"] = m_post["close"]

    # 수익률(%) 및 초과수익률(%)
    out["return_approx"] = (out["post_close"] - out["prev_close"]) / out["prev_close"] * 100
    out["market_return_approx"] = (out["m_post"] - out["m_prev"]) / out["m_prev"] * 100
    out["abnormal_return"] = out["return_approx"] - out["market_return_approx"]

    # 요약 통계
    summary = (
        out.dropna(subset=["abnormal_return"])
           .groupby(["category", "event_type"])["abnormal_return"]
           .agg(count="count", mean="mean", median="median")
           .reset_index()
           .sort_values(["count", "mean"], ascending=[False, False])
    )
    return out.sort_values("event_date", ascending=False).reset_index(drop=True), summary

# =========================================================
# 3. 데이터 수집 함수 (경기, 뉴스, 주가, 시장지수)
# =========================================================
@st.cache_data(ttl=3600)
def fetch_all_matches(team, key, limit_recent=None):
    league = TEAM_TO_LEAGUE[team]
    team_id = TEAM_IDS[league]
    season_ids = SEASON_IDS_BY_LEAGUE[league]

    rows = []
    for sid in reversed(season_ids):
        url = f"https://api.football-data-api.com/league-matches?key={key}&season_id={sid}"
        try:
            r = requests.get(url, timeout=10).json()
        except:
            continue
        if "data" not in r:
            continue
        for m in r["data"]:
            if m.get("homeID") == team_id or m.get("awayID") == team_id:
                m["season_id"] = sid
                rows.append(m)
        if limit_recent and len(rows) >= limit_recent:
            break
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date_unix" in df:
        df["date"] = pd.to_datetime(df["date_unix"], unit="s", utc=True).dt.tz_localize(None)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    if limit_recent is not None:
        return df.head(limit_recent)
    return df

@st.cache_data(ttl=3600)
def fetch_season_news(team, start_date, end_date, limit=200):
    keywords = [
        f"{team}", f"{team} news", f"{team} transfer", f"{team} injury",
        f"{team} manager", f"{team} 시즌 뉴스", f"{team} 이적", f"{team} 부상", f"{team} 감독",
    ]
    all_rows = []
    for kw in keywords:
        encoded = quote(kw, safe="")
        query = f"{encoded}+after:{start_date.strftime('%Y-%m-%d')}+before:{end_date.strftime('%Y-%m-%d')}"
        url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                all_rows.append({"title": e.title, "link": e.link, "published": e.get("published", "")})
        except Exception:
            continue
    if not all_rows:
        return pd.DataFrame(columns=["title", "link", "published"])
    news_df = pd.DataFrame(all_rows).drop_duplicates(subset="title")
    news_df["published_dt"] = pd.to_datetime(news_df["published"], errors="coerce")
    news_df = news_df.dropna(subset=["published_dt"])
    news_df = news_df[(news_df["published_dt"] >= pd.to_datetime(start_date)) & (news_df["published_dt"] <= pd.to_datetime(end_date))]
    return news_df.sort_values("published_dt", ascending=False).head(limit).reset_index(drop=True)

@st.cache_data(ttl=86400)
def fetch_stock(team):
    ticker = STOCK_TICKERS[team]
    df = yf.download(ticker, start="2017-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False, threads=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "Open", "Close"]].rename(columns={"Open": "open", "Close": "close"})

@st.cache_data(ttl=86400)
def fetch_market_index(team):
    ticker = MARKET_INDEX_TICKERS[TEAM_TO_LEAGUE[team]]
    df = yf.download(ticker, start="2017-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False, threads=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "Close"]].rename(columns={"Close": "close"})

# =========================================================
# 4. 데이터 가공 (경기결과, 상관분석)
# =========================================================
def enrich_results(df, team):
    if df.empty: return df
    league = TEAM_TO_LEAGUE[team]
    team_id = TEAM_IDS[league]
    results, scores, gf, ga = [], [], [], []
    for _, r in df.iterrows():
        hg, ag = r.get("homeGoalCount", 0), r.get("awayGoalCount", 0)
        home, away = r.get("homeID"), r.get("awayID")
        goals_for = hg if home == team_id else ag
        goals_against = ag if home == team_id else hg
        winner = r.get("winningTeam")
        if winner == team_id: label, score = "승", 1
        elif winner == -1: label, score = "무", 0
        else: label, score = "패", -1
        results.append(label); scores.append(score); gf.append(goals_for); ga.append(goals_against)
    df = df.copy()
    df["결과"] = results; df["결과점수"] = scores; df["득점"] = gf; df["실점"] = ga
    return df

def correlate(df, stock, market):
    if df.empty or stock.empty or market.empty: return None, None, None
    df = df.sort_values("date"); stock = stock.sort_values("date"); market = market.sort_values("date")
    
    pre = pd.merge_asof(df[["date", "결과점수"]], stock[["date", "close"]], on="date", direction="backward").rename(columns={"close": "prev_close"})
    post = pd.merge_asof(df[["date"]], stock[["date", "close"]], on="date", direction="forward", tolerance=pd.Timedelta("7D")).rename(columns={"close": "post_close_5d"})
    merged = pre.merge(post, on="date")
    merged["return_5d"] = (merged["post_close_5d"] - merged["prev_close"]) / merged["prev_close"] * 100
    
    m_pre = pd.merge_asof(df[["date"]], market[["date", "close"]], on="date", direction="backward").rename(columns={"close": "m_prev"})
    m_post = pd.merge_asof(df[["date"]], market[["date", "close"]], on="date", direction="forward", tolerance=pd.Timedelta("7D")).rename(columns={"close": "m_post"})
    merged["market_return_5d"] = (m_post["m_post"] - m_pre["m_prev"]) / m_pre["m_prev"] * 100
    merged["abnormal_return"] = merged["return_5d"] - merged["market_return_5d"]
    merged["결과라벨"] = merged["결과점수"].map({1: "승", 0: "무", -1: "패"})
    merged = merged.dropna()
    
    if merged.empty or len(merged) < 2: return merged, None, pd.DataFrame()
    corr_val = merged["결과점수"].corr(merged["abnormal_return"])
    stats = merged.groupby("결과라벨")["abnormal_return"].agg(count="count", mean="mean").reset_index()
    return merged, corr_val, stats

# =========================================================
# 5. [NEW] 감성 분석 데이터 처리 함수 (CSV 연동)
# =========================================================
def process_sentiment_data(csv_file, season_year, stock_df):
    """
    CSV에서 감성 데이터를 로드하고 특정 시즌으로 필터링한 후 주가 데이터와 결합
    """
    if csv_file is None:
        return None, None

    try:
        sent_df = pd.read_csv(csv_file)
        # 컬럼명 소문자 통일
        sent_df.columns = [c.lower() for c in sent_df.columns]
        
        if 'season' not in sent_df.columns or 'date' not in sent_df.columns:
            st.warning("⚠️ CSV 파일에 'season' 또는 'date' 컬럼이 필요합니다.")
            return None, None

        # 날짜 변환
        sent_df['date'] = pd.to_datetime(sent_df['date'], errors='coerce')
        sent_df = sent_df.dropna(subset=['date'])

        # 시즌 필터링 (예: 2024 -> "2024-2025" 문자열 포함 여부 확인)
        target_season_str = f"{season_year}-{season_year+1}" 
        filtered_sent = sent_df[sent_df['season'].astype(str).str.contains(target_season_str, na=False)].copy()

        if filtered_sent.empty:
            st.warning(f"⚠️ CSV에서 시즌 '{target_season_str}'에 해당하는 데이터를 찾을 수 없습니다.")
            return None, None
        
        # 감성 점수 컬럼 찾기 (sentiment, sentiment_score, score 등)
        score_col = next((c for c in filtered_sent.columns if 'score' in c or 'sentiment' in c), None)
        if not score_col:
             st.warning("⚠️ 감성 점수 컬럼(예: sentiment_score)을 찾을 수 없습니다.")
             return None, None

        # 날짜별 평균 감성 점수 집계
        daily_sent = filtered_sent.groupby('date')[score_col].mean().reset_index()
        daily_sent.rename(columns={score_col: 'avg_sentiment'}, inplace=True)

        # 주가 데이터와 병합 (Sentiment Date 기준으로 주가 매핑)
        stock_df = stock_df.sort_values('date')
        merged_sent = pd.merge_asof(
            daily_sent.sort_values('date'),
            stock_df[['date', 'close']],
            on='date',
            direction='nearest', 
            tolerance=pd.Timedelta('3D') # 주말 고려
        ).dropna()

        # 상관계수 계산 (감성점수 vs 주가)
        corr_val = merged_sent['avg_sentiment'].corr(merged_sent['close'])
        
        return merged_sent, corr_val

    except Exception as e:
        st.error(f"감성 분석 데이터 처리 중 오류: {e}")
        return None, None


# =========================================================
# 6. Gemini 리포트 생성 (감성 분석 포함 + 상세 전략 요청)
# =========================================================
def generate_gemini_report(team, merged, corr_val, stats, news_df, sent_merged, sent_corr, api_key):
    genai.configure(api_key=api_key)

    # 1) 경기 요약
    match_summary = (
        f"{len(merged)}경기 분석: "
        f"승 {sum(merged['결과라벨']=='승')} / "
        f"무 {sum(merged['결과라벨']=='무')} / "
        f"패 {sum(merged['결과라벨']=='패')}"
    )
    corr_text = f"{corr_val:.2f}" if corr_val is not None else "N/A"

    # 2) 뉴스 요약
    if news_df is not None and not news_df.empty:
        top_news = news_df.head(30)
        news_text = "\n".join(
            f"- {str(row['published_dt'].date())} : {row['title']}"
            for _, row in top_news.iterrows()
        )
    else:
        news_text = "해당 시즌 내 뉴스 데이터가 충분하지 않습니다."
        
    # 3) 감성 데이터 요약
    if sent_merged is not None and not sent_merged.empty:
        sent_info = f"""
        - 팬 감성 점수(Sentiment)와 주가 간의 상관계수: {sent_corr:.2f}
        - 감성 데이터 표본 수: {len(sent_merged)}일
        - (상관계수가 양수면 여론이 좋을 때 주가 상승 경향, 음수면 반대)
        """
    else:
        sent_info = "감성 분석 데이터(CSV)가 제공되지 않았습니다."

    # 4) 프롬프트 작성
    prompt = f"""
당신은 {team} 구단의 최고 데이터 분석가입니다.
경기 결과(정형), 뉴스(비정형), 그리고 팬들의 감성 분석(Sentiment) 데이터를 결합한 심층 리포트를 작성하세요.

### 📌 1. 경기 성과 요약
{match_summary}

### 📌 2. 경기결과 vs 주가 상관분석
- 경기결과 점수 ↔ 5거래일 시장 초과수익률 상관계수: {corr_text}
- 승·무·패별 평균 초과수익률:
{stats.to_string(index=False) if stats is not None else "N/A"}

### 📌 3. 팬 심리(감성분석) vs 주가 연관성
{sent_info}
- 팬들의 커뮤니티 여론/댓글 감성 점수가 주가에 어떤 영향을 미치는지 분석하세요.
- 경기 결과와 별개로 팬들의 분노(부정 감성)나 기대감(긍정 감성)이 주가 변동을 설명하는 부분이 있는지 확인하세요.

### 📌 4. 비정형 데이터(뉴스) 기반 영향 분석
주요 뉴스:
{news_text}
- 경기 결과와 무관하게 주가가 움직인 날짜의 설명 (이적, 부상, 스캔들 등)
- 비정형 데이터가 왜 정량적 모델을 보완하는지

### 📌 5. 종합 운영 인사이트 및 전략 제안 (구체적으로 작성)
아래 3가지 관점에서 구체적인 행동 전략을 제안해주세요:
1. **구단 이해관계자(프런트/경영진) 행동 전략**: 단기/중기/장기 과제
2. **스폰서 브랜드 행동 전략**: 마케팅 타이밍 및 리스크 관리
3. **소비자/팬/투자자 의사결정 권고**: 팬심에 휩쓸리지 않는 합리적 판단 가이드

전문 애널리스트 스타일로 자세하고 논리적으로 작성하세요.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")  
    try:
        res = model.generate_content(prompt)
        return res.text
    except:
        return "리포트 생성 중 오류가 발생했습니다."


# =========================================================
# 7. Streamlit UI 메인
# =========================================================
st.title("⚽ FIELD GRID – 구단 운영 AI 분석 시스템")
st.caption("지원 팀: 맨유, 도르트문트, 리옹, 유벤투스, 아약스, 셀틱")

# --- [NEW] 사이드바: CSV 업로드 ---
with st.sidebar:
    st.header("📂 데이터 업로드")
    sentiment_file = st.file_uploader("감성 분석 CSV 업로드 (컬럼: date, sentiment_score, season)", type=["csv"])
    st.caption("ℹ️ 'season' 컬럼에 '2024-2025' 같은 형식이 있어야 합니다.")

query = st.text_input("분석할 내용을 입력하세요.", placeholder="예: 맨유 2024시즌")

if not query:
    st.stop()

team = normalize_team_name(query)
if not team:
    st.error("⚠️ 팀명을 찾을 수 없습니다. (예: 맨유, 도르트문트 등)")
    st.stop()

st.success(f"분석 대상 팀: {team}")

# --- 기간 파싱 ---
period_type, period_value = parse_period(query)
season_year = extract_season_year(query) # 예: 2024
limit = 10 if "최근" in query else None

# --- 데이터 수집 ---
with st.spinner("데이터 수집 중..."):
    matches = fetch_all_matches(team, FOOTBALL_API_KEY, limit_recent=limit)
    stock = fetch_stock(team)
    market = fetch_market_index(team)
    enriched = enrich_results(matches, team)

    if enriched.empty:
        st.error("선택한 팀의 경기 데이터가 없습니다.")
        st.stop()

    enriched["date"] = pd.to_datetime(enriched["date"])

    # 필터링 로직
    if season_year:
        league = TEAM_TO_LEAGUE[team]
        start_mmdd, end_mmdd = SEASON_DATE_RANGE[league]
        season_start = pd.to_datetime(f"{season_year}-{start_mmdd}")
        season_end = pd.to_datetime(f"{season_year + 1}-{end_mmdd}")
        filtered = enriched[(enriched["date"] >= season_start) & (enriched["date"] <= season_end)].copy()
    elif period_type == "date":
        target_date = period_value.normalize()
        filtered = enriched[enriched["date"].dt.normalize() == target_date].copy()
        season_start = target_date - pd.Timedelta(days=7)
        season_end = target_date + pd.Timedelta(days=7)
    elif period_type == "month":
        ym = period_value
        filtered = enriched[(enriched["date"].dt.year == ym.year) & (enriched["date"].dt.month == ym.month)].copy()
        season_start = ym
        season_end = (ym + pd.offsets.MonthEnd(0))
    else:
        filtered = enriched.copy()
        if not filtered.empty:
            season_start = filtered["date"].min() - pd.Timedelta(days=7)
            season_end = filtered["date"].max() + pd.Timedelta(days=7)
        else:
            season_end = pd.Timestamp.today()
            season_start = season_end - pd.Timedelta(days=30)

    if filtered.empty:
        st.error("요청하신 기간(시즌)에 해당하는 경기 데이터가 없습니다.")
        st.stop()

    # 뉴스 및 상관분석
    news_df = fetch_season_news(team, season_start, season_end, limit=200)
    merged, corr_val, stats = correlate(filtered, stock, market)

# --- [NEW] 감성 데이터 처리 ---
sent_merged = None
sent_corr = 0.0
if sentiment_file is not None and season_year is not None:
    with st.spinner(f"팬 감성 데이터 분석 중... ({season_year}-{season_year+1} 시즌)"):
        sent_merged, sent_corr = process_sentiment_data(sentiment_file, season_year, stock)
        if sent_merged is not None:
            st.success(f"✅ 감성 데이터 로드 완료! (상관계수: {sent_corr:.2f})")


if merged is None or merged.empty:
    st.error("⚠️ 분석 가능한 데이터가 부족합니다.")
    st.stop()


# =========================================================
# 8. 시각화 (탭 구조)
# =========================================================
st.subheader("📊 데이터 시각화")
tab1, tab2, tab3 = st.tabs(["📉 경기vs주가", "❤️ 팬심(감성)vs주가", "📰 뉴스"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=merged["date"], y=merged["결과점수"], name="경기결과", marker_color="royalblue", opacity=0.6))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["abnormal_return"], yaxis="y2", name="초과수익률(%)", line=dict(color='firebrick')))
    fig.update_layout(
        title=f"{team} 경기성과 vs 주가 반응",
        yaxis=dict(title="경기결과", range=[-1.5, 1.5]),
        yaxis2=dict(title="초과수익률(%)", overlaying="y", side="right"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if corr_val is not None:
        st.info(f"**경기결과 점수 vs 5거래일 초과수익률 상관계수:** `{corr_val:.2f}`")
        if stats is not None and not stats.empty:
            st.dataframe(stats.rename(columns={"결과라벨": "결과", "count": "경기수", "mean": "평균 초과수익률(%)"}), use_container_width=True)

with tab2:
    if sent_merged is not None and not sent_merged.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sent_merged['date'], y=sent_merged['close'], name='주가 (Close)', line=dict(color='gray', width=2)))
        fig2.add_trace(go.Scatter(x=sent_merged['date'], y=sent_merged['avg_sentiment'], name='평균 감성 점수', yaxis='y2', line=dict(color='purple', width=2), mode='lines+markers'))
        fig2.update_layout(
            title=f"{team} {season_year}시즌 팬 심리 vs 주가",
            yaxis=dict(title="주가"),
            yaxis2=dict(title="감성 점수", overlaying="y", side="right"),
            template="plotly_white",
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"**팬 감성 vs 주가 상관계수:** `{sent_corr:.2f}`")
    else:
        st.info("좌측 사이드바에서 CSV 파일을 업로드하고, '2024시즌'과 같이 시즌을 명시해야 분석이 가능합니다.")

with tab3:
    if news_df is None or news_df.empty:
        st.info("뉴스를 찾지 못했습니다.")
    else:
        labeled_news, news_summary = analyze_news_stock_impact(news_df, stock, market)

        st.markdown("#### 🧩 뉴스 이벤트 타입 분류 & 카테고리별 주가 영향(초과수익률) 요약")
        if news_summary is None or news_summary.empty:
            st.info("카테고리 분류는 수행했지만, 주가/시장 데이터 매칭이 부족해 영향도 요약이 제한됩니다.")
        else:
            display_summary = news_summary.copy()

            display_summary["category"] = display_summary["category"].map(
                CATEGORY_KR_MAP
            ).fillna(display_summary["category"])

            display_summary["event_type"] = display_summary["event_type"].map(
                EVENT_TYPE_KR_MAP
            ).fillna(display_summary["event_type"])
            st.dataframe(
                display_summary.rename(columns={
                    "category": "카테고리",
                    "event_type": "이벤트 타입",
                    "count": "표본수",
                    "mean": "평균 초과수익률(%)",
                    "median": "중앙값 초과수익률(%)",
                }),
                use_container_width=True
            )

            # 카테고리 평균(가중) 간단 시각화
            cat_mean = (
                labeled_news.dropna(subset=["abnormal_return"])
                            .groupby("category")["abnormal_return"]
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index()
            )
            fig_news = go.Figure()
            fig_news.add_trace(go.Bar(x=cat_mean["category"], y=cat_mean["abnormal_return"], name="카테고리 평균 초과수익률(%)"))
            fig_news.update_layout(title="뉴스 카테고리별 평균 초과수익률(%)", template="plotly_white")
            st.plotly_chart(fig_news, use_container_width=True)

        for _, n in labeled_news.head(10).iterrows():
            ar = n.get("abnormal_return", None)
            ar_txt = f"{ar:.2f}%" if pd.notna(ar) else "NA"
            st.markdown(
                f"- **[{n['event_date'].date()}]** ({n.get('category','Other')} / {n.get('event_type','Other')} / AR:{ar_txt}) "
                f"[{n['title']}]({n['link']})"
            )

        with st.expander(f"전체 뉴스 보기 ({len(news_df)}개)"):
            for _, n in news_df.iterrows():
                st.markdown(f"- **[{n['published_dt'].date()}]** [{n['title']}]({n['link']})")


# =========================================================
# 9. AI 리포트 및 행동 권고 (기존 기능 복원 + 감성 분석 반영)
# =========================================================
st.divider()
st.subheader("🤖 AI 구단 운영 리포트 생성")

# session_state 초기화
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None
if "show_stakeholders" not in st.session_state:
    st.session_state.show_stakeholders = False
if "show_sponsor" not in st.session_state:
    st.session_state.show_sponsor = False
if "show_consumer" not in st.session_state:
    st.session_state.show_consumer = False

if st.button("AI 리포트 생성하기", type="primary"):
    with st.spinner("Gemini가 경기 결과, 뉴스, 그리고 팬들의 감성(Sentiment)까지 종합 분석 중입니다..."):
        report = generate_gemini_report(team, merged, corr_val, stats, news_df, sent_merged, sent_corr, GEMINI_API_KEY)
        st.session_state.ai_report = report
        # 버튼 상태 초기화
        st.session_state.show_stakeholders = False
        st.session_state.show_sponsor = False
        st.session_state.show_consumer = False

# 리포트 출력 및 하단 버튼 액션
if st.session_state.ai_report:
    st.markdown(st.session_state.ai_report)
    st.markdown("---")

    st.markdown("### 📌 분석 보고서를 기반으로 한 ‘행동 권고 요약’")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("① 구단 이해관계자 행동 권고"):
            st.session_state.show_stakeholders = not st.session_state.show_stakeholders
    with colB:
        if st.button("② 스폰서 행동 전략"):
            st.session_state.show_sponsor = not st.session_state.show_sponsor
    with colC:
        if st.button("③ 소비자·팬 의사결정 권고"):
            st.session_state.show_consumer = not st.session_state.show_consumer

    # ---------------------------------------------------------
    # ① 구단 이해관계자 행동 권고 (AI 리포트 기반 내용으로 구성하도록 유도)
    # ---------------------------------------------------------
    if st.session_state.show_stakeholders:
        st.markdown(f"### 🏟 ① 구단 이해관계자 행동 전략 권고 ({team})")
        # 동적 데이터를 반영한 권고 문구 (예시)
        advice_text = f"""
**이번 시즌 {team}의 경기결과(상관관계 {corr_val:.2f})와 팬 감성(상관관계 {sent_corr:.2f}) 데이터를 기반으로 한 전략입니다.**

### 🔥 1) 단기(1~4주) 우선조치
- **팬 감성 관리**: 현재 팬 감성 데이터와 주가의 상관관계를 볼 때, 경기 직후 공식 SNS 소통을 강화하여 부정적 여론 확산을 조기에 차단해야 합니다.
- **리스크 대응**: 패배 직후 주가가 민감하게 반응한다면, 경기 결과에 대한 해명보다 '다음 경기 전술 수정 계획'을 빠르게 발표하십시오.

### 🔥 2) 중기(1~2개월) 전략
- **선수단 운용**: 뉴스 데이터에서 지적된 특정 포지션 약점을 보완하거나, 해당 포지션 유망주 기용을 통해 팬들의 기대감을 조성하십시오.
- **재정 안정화**: 주가 하락 방어를 위해 성적 부진 시기에는 재정적 호재(새로운 파트너십 등)를 전략적으로 배치하십시오.
"""
        st.info(advice_text)

    # ---------------------------------------------------------
    # ② 스폰서 행동 전략
    # ---------------------------------------------------------
    if st.session_state.show_sponsor:
        st.markdown("### 💼 ② 스폰서 브랜드 행동 권고")
        advice_text = f"""
**분석 결과, {team}의 경기력과 팬들의 감정 기복은 브랜드 노출 효율에 직접적인 영향을 줍니다.**

### 🔵 1) 캠페인 집행 타이밍
- **승리/호재 뉴스 직후**: 경기 승리뿐만 아니라, 팬 감성 점수가 높은 구간(긍정 여론)에 광고 예산을 집중하십시오. ROI가 극대화됩니다.
- **감성 지표 활용**: 팬 감성 점수가 급락하는 시기(부정적 여론)에는 브랜드 노출을 최소화하여 부정적 이미지 전이를 막으십시오.

### 🔵 2) 위기 관리
- 악재 뉴스(부상/불화) 발생 시 24시간 모니터링을 강화하고, 구단을 응원하는 '동반자' 컨셉의 메시지로 톤앤매너를 조정하십시오.
"""
        st.info(advice_text)

    # ---------------------------------------------------------
    # ③ 소비자·팬 의사결정 권고
    # ---------------------------------------------------------
    if st.session_state.show_consumer:
        st.markdown("### 🎟 ③ 소비자·팬 의사결정 권고")
        advice_text = f"""
**일반 팬 및 투자자 관점에서 AI 데이터가 제안하는 합리적 의사결정입니다.**

### ✔ 1) 감정적 동요 방지
- **팬심 vs 주가**: 감성 점수와 주가의 상관계수가 `{sent_corr:.2f}`입니다. 이는 팬들의 분위기가 주가에 {( '상당 부분 반영됨' if abs(sent_corr)>0.5 else '큰 영향을 주지 않음' )}을 의미합니다. 
- 패배 직후의 분노나 연승 직후의 환희에 휩쓸려 굿즈를 과소비하거나 주식을 매매하는 것을 경계하십시오.

### ✔ 2) 뉴스 팩트 체크
- AI 분석 결과, 경기 성적보다 뉴스(이적설, CEO 발언 등)에 주가가 더 크게 반응하는 날이 있습니다. 단순 경기 결과만 보고 판단하지 마십시오.
"""
        st.info(advice_text)

# =========================================================
# 10. 법적 고지
# =========================================================
st.markdown("---")
st.caption("""
💡 **본 서비스에서 제공되는 모든 분석 및 권고는 정보 제공 목적이며,  
투자·재정적 의사결정에 대한 책임은 전적으로 사용자 본인에게 있습니다.**
""")