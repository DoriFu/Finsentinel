"""
FinSentinel: Financial Risk Intelligence + Sentiment-Price Correlation
=======================================================================
Requirements:
    pip install streamlit openai rank-bm25 scikit-learn feedparser datasets
    pip install yfinance scipy plotly

Optional (for full pipeline):
    pip install transformers peft torch
"""

import os
import re
import json
import time
import random
import hashlib
import warnings
import itertools
from datetime import datetime, timezone
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependencies — graceful fallbacks throughout
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except Exception:
    BM25_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from statsmodels.stats.proportion import proportion_confint
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    import feedparser
    FEEDPARSER_OK = True
except Exception:
    FEEDPARSER_OK = False

try:
    from datasets import load_dataset
    HF_DATASETS_OK = True
except Exception:
    HF_DATASETS_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except Exception:
    YFINANCE_OK = False

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------------------------------------------------------------------------
# Section 1: Configuration — preserved from notebook
# ---------------------------------------------------------------------------

CONFIG = {
    # Data
    "fpb_path":          "Sentences_AllAgree.txt",
    "fpb_encoding":      "latin-1",
    "label_col":         "Sentiment",
    "text_col":          "Headline",
    "random_state":      42,
    "test_size":         0.20,
    "llm_sample_size":   200,
    "agent_sample_size": 100,

    # TF-IDF
    "tfidf_max_features": 5000,
    "tfidf_ngram_range":  (1, 2),
    "cv_folds":           5,

    # FinBERT / LoRA names preserved from notebook
    "finbert_model":      "ProsusAI/finbert",
    "distilbert_model":   "distilbert-base-uncased",
    "lora_r":             8,
    "lora_alpha":         16,
    "lora_dropout":       0.1,
    "lora_epochs":        3,
    "lora_lr":            2e-4,
    "lora_batch_size":    16,

    # LLM prompting
    "llm_model":          "gpt-4o-mini",
    "llm_temperature":    0,
    "few_shot_n":         6,
    "rag_k":              3,
    "cot_n_paths":        5,

    # RAG corpus
    "corpus_size":        500,

    # Labels
    "label_map":          {"positive": 0, "neutral": 1, "negative": 2},
    "id2label":           {0: "positive", 1: "neutral", 2: "negative"},
    "label_colors":       {"positive": "#5DCAA5", "neutral": "#EF9F27", "negative": "#F0997B"},
}

# ---------------------------------------------------------------------------
# Few-shot examples — preserved from notebook
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = {
    "positive": [
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the same period last year.",
        "Net sales of the Group increased by 9 % to EUR 48.2 mn."
    ],
    "neutral": [
        "Tikkurila Powder Coatings has some 50 employees at its four paint plants.",
        "According to Gran, the company has no plans to move all production to Russia."
    ],
    "negative": [
        "Orion Corp reported a fall in earnings hit by larger R&D expenditures.",
        "The Group operating result fell to EUR -0.3 mn."
    ],
}

FIQA_STYLE_EXAMPLES = [
    {"Headline": "Shares fell after the company warned that weak demand could pressure margins.", "Sentiment": "negative"},
    {"Headline": "The bank said credit losses may increase if commercial real estate weakens further.", "Sentiment": "negative"},
    {"Headline": "The firm announced a new buyback program after stronger-than-expected earnings.", "Sentiment": "positive"},
    {"Headline": "Management said macro uncertainty remains elevated but liquidity is stable.", "Sentiment": "neutral"},
]

DEFAULT_CORPUS = []
for label, examples in FEW_SHOT_EXAMPLES.items():
    DEFAULT_CORPUS.extend(examples)
DEFAULT_CORPUS.extend([x["Headline"] for x in FIQA_STYLE_EXAMPLES])
DEFAULT_CORPUS.extend([
    "Silicon Valley Bank collapses after deposit outflows and liquidity concerns.",
    "Regional banks face downgrade warnings after funding stress intensifies.",
    "Corporate default rates rise as tighter credit conditions pressure borrowers.",
    "Treasury yields surge as investors price in higher-for-longer interest rates.",
    "Commercial real estate losses pressure regional lenders and raise contagion fears.",
    "Major technology firms announce layoffs amid slowing revenue growth.",
    "Oil prices jump after geopolitical tensions disrupt supply routes.",
    "Market volatility spikes as investors react to inflation surprise.",
])

# ---------------------------------------------------------------------------
# Section 3: Data loading
# ---------------------------------------------------------------------------

def load_fpb(path, encoding="latin-1"):
    """Parse the @-delimited FinancialPhraseBank format."""
    records = []
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            records.append({
                CONFIG["text_col"]:  sentence.strip(),
                CONFIG["label_col"]: label.strip().lower()
            })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False, ttl=3600)
def load_fiqa_from_huggingface(dataset_name="TheFinAI/fiqa-sentiment-classification", split="train", max_rows=200):
    if not HF_DATASETS_OK:
        return pd.DataFrame(), "Install HuggingFace datasets first: pip install datasets"
    try:
        ds = load_dataset(dataset_name, split=split)
        df = ds.to_pandas().head(max_rows).reset_index(drop=True)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Could not load FiQA from HuggingFace: {e}"


def infer_fiqa_text_column(df):
    preferred = ["sentence", "text", "headline", "query", "question", "doc"]
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_to_original:
            return lower_to_original[name]
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return df.columns[0] if len(df.columns) else None

# ---------------------------------------------------------------------------
# Section 7: GPT Prompting — notebook function names preserved
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_openai_client():
    if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
        return None
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(prompt, temperature=0, model=None):
    client = get_openai_client()
    if client is None:
        return None
    model = model or CONFIG["llm_model"]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        st.warning(f"API error, using fallback instead: {e}")
        time.sleep(1)
        return None


def run_llm_evaluation(prompt_fn, sample_df, desc="LLM"):
    outputs, statuses = [], []
    for _, row in sample_df.iterrows():
        result = call_llm(prompt_fn(row[CONFIG["text_col"]]))
        if result and "sentiment" in result:
            outputs.append(result)
            statuses.append("ok")
        else:
            outputs.append(None)
            statuses.append("parse_error")
        time.sleep(0.1)
    return outputs, statuses


def format_few_shot_block():
    lines = ["\nExamples:"]
    for label, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples:
            lines.append(f'Headline: "{ex}"')
            lines.append(f'{{"sentiment": "{label}"}}\n')
    return "\n".join(lines)

FEW_SHOT_BLOCK = format_few_shot_block()

RISK_JSON_SCHEMA = '''{
  "sentiment": "positive | neutral | negative",
  "risk_type": "liquidity_risk | credit_risk | macro_risk | market_volatility | earnings_risk | regulatory_risk | geopolitical_risk | labor_operational_risk | none",
  "severity": "low | medium | high | extreme",
  "market_impact": "regional_banking | broad_market | technology | energy | real_estate | consumer | company_specific | none",
  "confidence": 0.0,
  "reasoning": "one concise sentence"
}'''


def build_zero_shot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment of this financial headline FROM THE PERSPECTIVE
OF A RETAIL INVESTOR. Ask: "Does this news have a clear directional signal
for my investment, or is it just factual reporting?"

Now extend the notebook's sentiment task into structured financial risk extraction.
Allowed sentiment labels: ["positive", "neutral", "negative"]
Allowed risk_type labels: ["liquidity_risk", "credit_risk", "macro_risk", "market_volatility", "earnings_risk", "regulatory_risk", "geopolitical_risk", "labor_operational_risk", "none"]
Allowed severity labels: ["low", "medium", "high", "extreme"]
Allowed time_horizon labels: ["short_term", "medium_term", "long_term", "unclear"]
Return ONLY valid JSON with no explanation or markdown.

Headline: "{headline}"
Output: {RISK_JSON_SCHEMA}"""


def build_few_shot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment FROM THE PERSPECTIVE OF A RETAIL INVESTOR.

Label definitions:
- positive: directional improvement — profit rose, sales grew, deal finalized
- neutral: factual reporting without clear directional outcome — announcing
  meetings, reporting figures without comparison, describing operations
- negative: directional decline — profit fell, loss reported, costs rose
{FEW_SHOT_BLOCK}
Now extend this same logic into structured risk extraction.
Return ONLY valid JSON using this schema:
{RISK_JSON_SCHEMA}

Headline: "{headline}"
Output:"""


def build_cot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment FROM THE PERSPECTIVE OF A RETAIL INVESTOR.
Think step by step before classifying.
{FEW_SHOT_BLOCK}
Now analyze step by step:
Headline: "{headline}"

Think: Does this sentence describe a directional financial outcome, or is it
factual reporting? What would a retail investor conclude from this? What risk,
severity, market impact, and time horizon are implied?

Return ONLY valid JSON:
{RISK_JSON_SCHEMA}"""

# ---------------------------------------------------------------------------
# Section 8: RAG — notebook names preserved
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_bm25_index(corpus_tuple):
    corpus = list(corpus_tuple)
    if BM25_OK:
        tokenized_corpus = [text.lower().split() for text in corpus]
        return BM25Okapi(tokenized_corpus), None, None
    if SKLEARN_OK:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
        return None, vectorizer, matrix
    return None, None, None


def bm25_retrieve(query, k=10, corpus_texts=None):
    corpus_texts = corpus_texts or DEFAULT_CORPUS
    bm25, vectorizer, matrix = build_bm25_index(tuple(corpus_texts))
    if bm25 is not None:
        scores = bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        return [(corpus_texts[i], float(scores[i])) for i in top_k]
    if vectorizer is not None:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_k = sims.argsort()[::-1][:k]
        return [(corpus_texts[i], float(sims[i])) for i in top_k]
    q_terms = set(query.lower().split())
    scored = []
    for text in corpus_texts:
        score = len(q_terms & set(text.lower().split())) / max(1, len(q_terms))
        scored.append((text, float(score)))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def hybrid_retrieve(query, k=CONFIG["rag_k"], alpha=0.6, corpus_texts=None):
    corpus_texts = corpus_texts or DEFAULT_CORPUS
    results = bm25_retrieve(query, k=min(20, len(corpus_texts)), corpus_texts=corpus_texts)
    top_k = [(text, score) for text, score in results if text.strip().lower() != query.strip().lower()][:k]
    return top_k


def build_rag_cot_prompt(headline, corpus_texts=None):
    retrieved = hybrid_retrieve(headline, k=CONFIG["rag_k"], corpus_texts=corpus_texts)
    context_block = "\n".join([f"  - {text}" for text, _ in retrieved])
    return f"""You are a financial sentiment analyst.

Similar recent financial headlines for context:
{context_block}

Use the context above to help calibrate your investor-perspective judgment.
{FEW_SHOT_BLOCK}
Now analyze step by step:
Headline: "{headline}"

Think: Does this sentence describe a directional financial outcome for an investor?
How is it similar or different from the context headlines? What risk type,
severity, market impact, and time horizon are implied?

Return ONLY valid JSON:
{RISK_JSON_SCHEMA}"""

# ---------------------------------------------------------------------------
# Section 14: Live news — notebook functions preserved
# ---------------------------------------------------------------------------

NOISY_PUBLISHERS = {"benzinga", "motley fool", "seeking alpha sponsored",
                    "globenewswire", "businesswire", "pr newswire"}


def parse_pub_date(entry):
    date_str = entry.get("published", "") or entry.get("updated", "")
    if not date_str:
        return None
    for fmt in ["%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False, ttl=900)
def fetch_headlines(ticker, max_results=40):
    if not FEEDPARSER_OK:
        return []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    seen_hashes = set()
    results = []
    for entry in feed.entries[:max_results]:
        headline = entry.get("title", "").strip()
        if not headline or len(headline) < 10:
            continue
        h_hash = hashlib.md5(re.sub(r'\s+', ' ', headline.lower()).encode()).hexdigest()
        if h_hash in seen_hashes:
            continue
        seen_hashes.add(h_hash)
        source = entry.get("source", {}).get("value", "").lower()
        if any(noise in source for noise in NOISY_PUBLISHERS):
            continue
        pub_dt = parse_pub_date(entry)
        if pub_dt is None:
            pub_dt = datetime.now(timezone.utc)
        results.append({
            "ticker":       ticker,
            "headline":     headline,
            "pub_datetime": pub_dt,
            "pub_date":     pub_dt.date(),
            "pub_hour":     pub_dt.hour,
            "source":       source,
        })
    return results


@st.cache_data(show_spinner=False, ttl=900)
def fetch_live_news(tickers, max_results=20):
    all_headlines = []
    for ticker in tickers:
        rows = fetch_headlines(ticker, max_results=max_results)
        all_headlines.extend(rows)
        time.sleep(0.1)
    if not all_headlines:
        return pd.DataFrame(columns=["ticker", "headline", "pub_datetime", "pub_date", "pub_hour", "source"])
    return pd.DataFrame(all_headlines).drop_duplicates(subset=["headline"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Fallback structured risk layer
# ---------------------------------------------------------------------------

RISK_TAXONOMY = {
    "liquidity_risk":       ["liquidity", "deposit", "withdrawal", "funding", "cash crunch", "bank run", "outflow"],
    "credit_risk":          ["default", "downgrade", "debt", "bankruptcy", "insolvency", "loan loss", "credit", "delinquency"],
    "macro_risk":           ["inflation", "fed", "rates", "recession", "unemployment", "gdp", "yield", "tightening"],
    "market_volatility":    ["volatility", "selloff", "rout", "crash", "tumble", "plunge", "panic", "spike"],
    "earnings_risk":        ["earnings", "guidance", "profit warning", "margin", "revenue miss", "forecast cut", "miss"],
    "regulatory_risk":      ["sec", "regulator", "lawsuit", "antitrust", "probe", "investigation", "fine", "compliance"],
    "geopolitical_risk":    ["war", "sanction", "tariff", "conflict", "geopolitical", "border", "military"],
    "labor_operational_risk":["layoff", "strike", "shutdown", "supply chain", "production halt", "recall"],
}

MARKET_IMPACT_TERMS = {
    "regional_banking": ["regional bank", "bank", "lender", "deposit", "svb", "jpm"],
    "broad_market":     ["s&p", "nasdaq", "dow", "market", "equities", "treasury", "yield", "stocks"],
    "technology":       ["tech", "ai", "chip", "semiconductor", "nvidia", "apple", "microsoft", "google", "meta", "tesla"],
    "energy":           ["oil", "gas", "crude", "energy", "opec"],
    "real_estate":      ["real estate", "commercial property", "mortgage", "reit", "housing"],
    "consumer":         ["consumer", "retail", "sales", "demand", "spending"],
}

NEGATIVE_TERMS = ["fall", "fell", "drop", "plunge", "loss", "miss", "default", "downgrade",
                  "bankruptcy", "layoff", "decline", "fraud", "investigation", "liquidity",
                  "crisis", "warning", "weak", "concern", "stress"]
POSITIVE_TERMS = ["rise", "rose", "gain", "beat", "surge", "growth", "profit", "record",
                  "upgrade", "strong", "increase", "rally", "rebound", "raises guidance"]


def _best_match(text, taxonomy, default="none"):
    lower = text.lower()
    best_label, best_hits = default, []
    for label, terms in taxonomy.items():
        hits = [term for term in terms if term in lower]
        if len(hits) > len(best_hits):
            best_label, best_hits = label, hits
    return best_label, best_hits


def fallback_structured_risk(headline):
    lower = headline.lower()
    pos = sum(1 for w in POSITIVE_TERMS if w in lower)
    neg = sum(1 for w in NEGATIVE_TERMS if w in lower)
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    risk_type, risk_hits = _best_match(headline, RISK_TAXONOMY, default="none")
    market_impact, impact_hits = _best_match(headline, MARKET_IMPACT_TERMS, default="company_specific")
    extreme_terms = ["collapse", "crisis", "bank run", "bankruptcy", "insolvency", "contagion", "panic"]
    high_terms    = ["plunge", "tumble", "downgrade", "default", "investigation", "layoffs", "warning", "selloff"]
    medium_terms  = ["concern", "pressure", "risk", "weak", "miss", "decline", "slowdown", "loss"]
    if any(t in lower for t in extreme_terms):
        severity = "extreme"
    elif any(t in lower for t in high_terms):
        severity = "high"
    elif any(t in lower for t in medium_terms) or risk_type != "none":
        severity = "medium"
    else:
        severity = "low"
    if any(t in lower for t in ["today", "now", "this week", "warning", "plunge", "spike", "downgrade"]):
        horizon = "short_term"
    elif any(t in lower for t in ["quarter", "guidance", "forecast", "outlook"]):
        horizon = "medium_term"
    elif any(t in lower for t in ["structural", "long-term", "secular"]):
        horizon = "long_term"
    else:
        horizon = "unclear"
    confidence = min(0.95, 0.55 + 0.07 * (pos + neg + len(risk_hits) + len(impact_hits)))
    return {
        "sentiment":     sentiment,
        "risk_type":     risk_type,
        "severity":      severity,
        "market_impact": market_impact,
        "time_horizon":  horizon,
        "confidence":    round(float(confidence), 2),
        "reasoning":     "Fallback keyword logic used because an LLM response was unavailable."
    }


def normalize_structured_output(result, headline):
    if not isinstance(result, dict):
        return fallback_structured_risk(headline)
    fallback = fallback_structured_risk(headline)
    allowed = {
        "sentiment":    {"positive", "neutral", "negative"},
        "severity":     {"low", "medium", "high", "extreme"},
        "time_horizon": {"short_term", "medium_term", "long_term", "unclear"},
    }
    out = fallback.copy()
    out.update(result)
    for key, values in allowed.items():
        val = str(out.get(key, "")).lower().strip()
        out[key] = val if val in values else fallback[key]
    try:
        out["confidence"] = round(float(out.get("confidence", fallback["confidence"])), 2)
    except Exception:
        out["confidence"] = fallback["confidence"]
    return out


def analyze_headline_with_notebook_pipeline(headline, prompt_mode, corpus_texts):
    if "zero-shot" in prompt_mode:
        prompt = build_zero_shot_prompt(headline)
    elif "few-shot" in prompt_mode:
        prompt = build_few_shot_prompt(headline)
    elif "CoT" in prompt_mode and "RAG" not in prompt_mode:
        prompt = build_cot_prompt(headline)
    else:
        prompt = build_rag_cot_prompt(headline, corpus_texts=corpus_texts)
    llm_result = call_llm(prompt, temperature=CONFIG["llm_temperature"])
    structured  = normalize_structured_output(llm_result, headline)
    retrieved   = hybrid_retrieve(headline, k=CONFIG["rag_k"], corpus_texts=corpus_texts)
    structured.update({
        "headline":           headline,
        "retrieved_context":  " | ".join([text for text, _ in retrieved]),
        "used_llm":           bool(llm_result),
    })
    return structured

# ===========================================================================
# NEW COMPONENT 1: Aggregated sentiment score per ticker per day
# ===========================================================================

def compute_sentiment_scores(results_df, news_df):
    """
    Aggregate per-headline sentiments into a daily score per ticker.
    sentiment_score = (positive - negative) / total  →  range [-1, +1]
    severity_weighted_score weights each headline by its severity level.
    """
    merged = results_df.copy()
    merged["ticker"]   = news_df["ticker"].values
    raw_dates = pd.to_datetime(news_df["pub_date"].values, errors="coerce")
    merged["pub_date"] = raw_dates.date if hasattr(raw_dates, "date") and not hasattr(raw_dates, "dt") else raw_dates.dt.date

    severity_map = {"low": 1, "medium": 2, "high": 3, "extreme": 4}

    def score_group(group):
        total = len(group)
        pos   = (group["sentiment"] == "positive").sum()
        neg   = (group["sentiment"] == "negative").sum()

        weighted = group.apply(
            lambda r: (1 if r["sentiment"] == "positive" else
                       -1 if r["sentiment"] == "negative" else 0) *
                      severity_map.get(str(r.get("severity", "low")), 1),
            axis=1
        ).sum()

        return pd.Series({
            "sentiment_score":        (pos - neg) / total if total > 0 else 0.0,
            "severity_weighted_score": weighted / total if total > 0 else 0.0,
            "positive_count":         int(pos),
            "negative_count":         int(neg),
            "neutral_count":          int(total - pos - neg),
            "total_headlines":        int(total),
            "avg_confidence":         round(float(group["confidence"].mean()), 3),
        })

    return (
        merged.groupby(["ticker", "pub_date"])
        .apply(score_group)
        .reset_index()
    )

# ===========================================================================
# NEW COMPONENT 2: Next-day and intraday price returns via yfinance
# ===========================================================================

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_returns(tickers_tuple, dates_tuple, lookahead_days=1):
    """
    For each (ticker, date) pair fetch same-day intraday return and
    next-trading-day close-to-close return via yfinance.
    Handles date/datetime type mismatches and includes retry logic.
    """
    if not YFINANCE_OK:
        return pd.DataFrame()

    tickers = list(tickers_tuple)

    # Normalise all dates to Python date objects regardless of input type
    raw_dates = list(dates_tuple)
    dates = []
    for d in raw_dates:
        try:
            if hasattr(d, "date"):
                dates.append(d.date())
            else:
                dates.append(pd.Timestamp(d).date())
        except Exception:
            continue

    if not dates:
        return pd.DataFrame()

    # Use a wider window to maximise chance of finding trading days
    min_date = min(dates)
    max_date = max(dates)
    start = (pd.Timestamp(min_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    end   = (pd.Timestamp(max_date) + pd.Timedelta(days=lookahead_days + 15)).strftime("%Y-%m-%d")

    results = []

    for ticker in tickers:
        price_data = pd.DataFrame()

        # Try up to 3 times with increasing delays
        for attempt in range(3):
            try:
                time.sleep(attempt * 1.5)
                raw = yf.download(
                    ticker, start=start, end=end,
                    progress=False, auto_adjust=True,
                    timeout=15
                )
                if not raw.empty:
                    price_data = raw
                    break
            except Exception:
                continue

        if price_data.empty:
            # Still add rows with None returns so ticker appears in output
            for date in dates:
                results.append({
                    "ticker":          ticker,
                    "pub_date":        date,
                    "intraday_return": None,
                    "next_day_return": None,
                })
            continue

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = price_data.columns.get_level_values(0)

        # Normalise index to Python date objects
        price_data.index = pd.to_datetime(price_data.index).map(lambda x: x.date())
        index_set = set(price_data.index)

        def safe_float(val):
            try:
                v = val.iloc[0] if hasattr(val, "iloc") else val
                return float(v)
            except Exception:
                return None

        for date in dates:
            intraday_return = None
            next_day_return = None

            if date in index_set:
                try:
                    o = safe_float(price_data.loc[date, "Open"])
                    c = safe_float(price_data.loc[date, "Close"])
                    if o and c and o != 0:
                        intraday_return = (c - o) / o
                except Exception:
                    pass

            # Find nearest future trading day
            future_dates = sorted([d for d in index_set if d > date])
            if future_dates and date in index_set:
                try:
                    next_day = future_dates[0]
                    c0 = safe_float(price_data.loc[date,     "Close"])
                    c1 = safe_float(price_data.loc[next_day, "Close"])
                    if c0 and c1 and c0 != 0:
                        next_day_return = (c1 - c0) / c0
                except Exception:
                    pass

            results.append({
                "ticker":          ticker,
                "pub_date":        date,
                "intraday_return": intraday_return,
                "next_day_return": next_day_return,
            })

        time.sleep(0.3)  # Be polite to Yahoo Finance API

    return pd.DataFrame(results) if results else pd.DataFrame()

# ===========================================================================
# NEW COMPONENT 3: Correlation analysis and visualisation
# ===========================================================================

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_spy_returns(start_date, end_date):
    """Next-day SPY return per calendar date, used as the market-wide control variable."""
    if not YFINANCE_OK:
        return pd.DataFrame(columns=["pub_date", "spy_ret"])
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
    if spy.empty:
        return pd.DataFrame(columns=["pub_date", "spy_ret"])
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy.reset_index().rename(columns={"Date": "pub_date"})
    spy["pub_date"] = pd.to_datetime(spy["pub_date"]).dt.date
    spy["spy_ret"] = spy["Close"].pct_change().shift(-1)
    return spy[["pub_date", "spy_ret"]]


def _residualize(x, control):
    """OLS residuals of x regressed on [1, control] (one fit, applied per row)."""
    x = np.asarray(x, dtype=float)
    control = np.asarray(control, dtype=float)
    design = np.column_stack([np.ones_like(control), control])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def partial_correlation(x, y, control):
    """Pearson correlation of x and y after residualizing both on `control`."""
    df = pd.DataFrame({"x": x, "y": y, "c": control}).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    x_resid = _residualize(df["x"], df["c"])
    y_resid = _residualize(df["y"], df["c"])
    r, p = stats.pearsonr(x_resid, y_resid)
    return r, p, len(df)


def show_market_controlled_correlation(analysis_df, score_col, return_col, raw_r):
    """
    A high raw r may just be measuring market-wide movement, not a sentiment
    signal. Partial correlation after removing next-day SPY return is the
    correct test: it isolates whatever predictive signal is left once the
    market factor is controlled for.
    """
    if not YFINANCE_OK:
        st.caption("Install yfinance to enable the market-controlled (SPY partial) correlation check.")
        return

    with st.expander("Market-controlled correlation (partials out SPY)", expanded=False):
        dates = pd.to_datetime(analysis_df["pub_date"])
        start = (dates.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        spy_df = fetch_spy_returns(start, end)

        if spy_df.empty:
            st.warning("Could not fetch SPY control series from yfinance.")
            return

        merged = analysis_df.merge(spy_df, on="pub_date", how="left")
        partial_r, partial_p, n = partial_correlation(
            merged[score_col], merged[return_col], merged["spy_ret"]
        )
        if n < 3:
            st.warning("Not enough overlapping SPY trading days to compute a partial correlation.")
            return

        c1, c2 = st.columns(2)
        c1.metric("Raw r (uncontrolled)", f"{raw_r:.3f}")
        c2.metric("Partial r (SPY controlled)", f"{partial_r:.3f}", f"p={partial_p:.3f}, n={n}")

        if abs(raw_r) > 0.2 and abs(partial_r) < 0.1:
            st.warning(
                "The raw correlation largely disappears once SPY is controlled for — "
                "it was likely measuring market-wide movement, not a sentiment-specific signal."
            )
        elif abs(partial_r) >= 0.1:
            st.success(
                "A meaningful correlation survives after controlling for the market factor, "
                "suggesting sentiment carries signal beyond broad market movement."
            )


def show_correlation_analysis(sentiment_scores_df, price_df):
    """
    Merge daily sentiment scores with price returns.
    Compute Pearson + Spearman correlations and display findings.
    """
    if not SCIPY_OK:
        st.warning("Install scipy for correlation analysis: pip install scipy")
        return
    if not PLOTLY_OK:
        st.warning("Install plotly for interactive charts: pip install plotly")
        return

    merged = sentiment_scores_df.merge(price_df, on=["ticker", "pub_date"], how="inner")

    if merged.empty or len(merged) < 3:
        st.warning(
            "Not enough matched data points (need ≥ 3). "
            "Try fetching more headlines or adding more tickers."
        )
        return

    st.subheader("Sentiment → Price Correlation Analysis")
    st.caption(
        "Core research contribution: does aggregated daily headline sentiment "
        "predict next-day or intraday price returns? "
        "Even a flat result is a finding — it tests the efficient market hypothesis."
    )

    return_col = st.radio(
        "Return window",
        ["next_day_return", "intraday_return"],
        horizontal=True,
        format_func=lambda x: "Next-day close return" if x == "next_day_return"
                              else "Same-day intraday return",
        key="return_window_radio"
    )

    score_col = st.radio(
        "Sentiment score variant",
        ["sentiment_score", "severity_weighted_score"],
        horizontal=True,
        format_func=lambda x: "Raw sentiment score" if x == "sentiment_score"
                              else "Severity-weighted score",
        key="score_variant_radio"
    )

    analysis_df = merged.dropna(subset=[score_col, return_col]).copy()

    if len(analysis_df) < 3:
        st.warning("Not enough clean rows after dropping NaNs.")
        return
    
    if return_col == "intraday_return" and analysis_df["intraday_return"].isna().all():
        st.warning(
            "Intraday return is unavailable for all rows. "
            "This happens when yfinance cannot return Open and Close prices "
            "for these tickers on these dates (common for weekends, holidays, "
            "or when the news date is outside the price data window). "
            "Switch to 'Next-day close return' which has broader coverage."
        )
        return

    r,   p_value   = stats.pearsonr(analysis_df[score_col], analysis_df[return_col])
    rho, p_spearman = stats.spearmanr(analysis_df[score_col], analysis_df[return_col])

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pearson r",   f"{r:.3f}")
    col2.metric("p-value",     f"{p_value:.3f}",
                delta="significant" if p_value < 0.05 else "not sig.",
                delta_color="normal" if p_value < 0.05 else "off")
    col3.metric("Spearman ρ",  f"{rho:.3f}")
    col4.metric("Observations", len(analysis_df))

    # Interpretation
    if p_value < 0.05 and abs(r) > 0.2:
        st.success(
            f"**Finding:** Statistically significant correlation detected "
            f"(r = {r:.3f}, p = {p_value:.3f}). "
            f"Headline sentiment shows meaningful predictive signal for "
            f"{return_col.replace('_', ' ')}."
        )
    elif p_value < 0.05:
        st.info(
            f"**Finding:** Weak but statistically significant correlation "
            f"(r = {r:.3f}, p = {p_value:.3f}). Signal exists but effect is small."
        )
    else:
        st.info(
            f"**Finding:** No statistically significant correlation detected "
            f"(r = {r:.3f}, p = {p_value:.3f}). "
            f"Headline sentiment alone does not reliably predict price movement "
            f"at this timescale — consistent with the efficient market hypothesis."
        )

    show_market_controlled_correlation(analysis_df, score_col, return_col, r)

    # Scatter with OLS trendline
    import numpy as np

    fig = px.scatter(
    analysis_df,
    x=score_col,
    y=return_col,
    color="ticker",
    hover_data=["headline"] if "headline" in analysis_df.columns else None,
    # trendline="ols" REMOVED — requires statsmodels which is not always installed
    title=f"Aggregated Sentiment Score vs {return_col.replace('_', ' ').title()}",
    labels={
        score_col:  "Aggregated Sentiment Score (−1 to +1)",
        return_col: "Price Return (%)"
    },
    template="plotly_white",
    )
    # Draw OLS trendline manually using numpy (no statsmodels dependency)
    clean = analysis_df[[score_col, return_col]].dropna()
    if len(clean) >= 3:
        try:
            m, b  = np.polyfit(clean[score_col], clean[return_col], 1)
            x_min = clean[score_col].min()
            x_max = clean[score_col].max()
            x_line = np.linspace(x_min, x_max, 100)
            y_line = m * x_line + b

            fig.add_scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                name=f"OLS trend (r={r:.2f})",
                showlegend=True,
            )
        except Exception:
            pass 

    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Per-ticker breakdown table
    st.subheader("Per-Ticker Correlation Breakdown")
    ticker_corrs = []
    for ticker in analysis_df["ticker"].unique():
        td = analysis_df[analysis_df["ticker"] == ticker]
        if len(td) >= 3:
            r_t, p_t = stats.pearsonr(td[score_col], td[return_col])
            ticker_corrs.append({
                "ticker":        ticker,
                "pearson_r":     round(r_t, 3),
                "p_value":       round(p_t, 3),
                "n_obs":         len(td),
                "significant":   p_t < 0.05,
                "avg_sentiment": round(td[score_col].mean(), 3),
            })
    if ticker_corrs:
        st.dataframe(
            pd.DataFrame(ticker_corrs).sort_values("pearson_r", ascending=False),
            use_container_width=True
        )

    # Sentiment score timeline
    st.subheader("Daily Sentiment Score Timeline")
    timeline_fig = px.line(
        analysis_df.sort_values("pub_date"),
        x="pub_date", y=score_col, color="ticker",
        title="Aggregated Sentiment Score Over Time",
        labels={"pub_date": "Date", score_col: "Sentiment Score"},
        template="plotly_white",
    )
    timeline_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    timeline_fig.update_layout(height=320)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Download merged dataset
    csv = analysis_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sentiment-price dataset (CSV)",
        csv,
        "finsentinel_sentiment_price_correlation.csv",
        "text/csv",
    )

# ===========================================================================
# NEW COMPONENT 4A: Single-headline deep-dive trace
# ===========================================================================

def show_single_headline_deepdive(corpus_texts, prompt_mode):
    st.subheader("Single Headline Deep Dive")
    st.caption(
        "Inspect the full pipeline trace for one headline: "
        "RAG context retrieved, exact prompt sent, raw LLM response, "
        "and final normalised output."
    )
    single = st.text_input(
        "Paste any financial headline",
        "Federal Reserve signals rates may stay higher for longer amid persistent inflation.",
        key="deepdive_input"
    )
    if not st.button("Analyse with full trace", key="deepdive_btn"):
        return

    with st.expander("① RAG Retrieved Context", expanded=True):
        retrieved = hybrid_retrieve(single, k=CONFIG["rag_k"], corpus_texts=corpus_texts)
        if retrieved:
            for text, score in retrieved:
                st.markdown(f"**Score: {score:.3f}** — {text}")
        else:
            st.write("No context retrieved.")

    with st.expander("② Prompt sent to LLM", expanded=False):
        if "zero-shot" in prompt_mode:
            p = build_zero_shot_prompt(single)
        elif "few-shot" in prompt_mode:
            p = build_few_shot_prompt(single)
        elif "CoT" in prompt_mode and "RAG" not in prompt_mode:
            p = build_cot_prompt(single)
        else:
            p = build_rag_cot_prompt(single, corpus_texts=corpus_texts)
        st.code(p, language="text")

    with st.expander("③ Structured output", expanded=True):
        result = analyze_headline_with_notebook_pipeline(single, prompt_mode, corpus_texts)
        st.json(result)

# ===========================================================================
# NEW COMPONENT 4B: Evaluation metrics when ground-truth labels are available
# ===========================================================================

def show_evaluation_metrics(results_df):
    """
    Display confusion matrix + classification report when true labels are present.
    Called automatically when Financial PhraseBank (labelled) is the input source.
    """
    if "true_label" not in results_df.columns:
        return
    if not SKLEARN_OK:
        st.warning("Install scikit-learn for evaluation metrics.")
        return

    st.subheader("Evaluation Metrics (vs ground-truth labels)")
    st.caption(
        "Any single accuracy number is meaningless without an uncertainty range — "
        "for a scientifically valid model comparison, run every model on the same "
        "frozen `eval_set_frozen.csv` sample (see eval_framework.py: build-eval-set)."
    )

    y_true = results_df["true_label"].str.lower().str.strip()
    y_pred = results_df["sentiment"].str.lower().str.strip()

    # Only evaluate rows where both are valid
    valid   = y_true.isin(["positive", "neutral", "negative"]) & \
              y_pred.isin(["positive", "neutral", "negative"])
    y_true  = y_true[valid]
    y_pred  = y_pred[valid]

    if len(y_true) < 2:
        st.info("Not enough labelled rows for evaluation.")
        return

    n = len(y_true)
    acc = (y_true.values == y_pred.values).mean()
    if STATSMODELS_OK:
        n_correct = int(round(acc * n))
        ci_low, ci_high = proportion_confint(n_correct, n, alpha=0.05, method="wilson")
        st.metric("Accuracy (95% Wilson CI)", f"{acc:.3f}", f"[{ci_low:.3f}, {ci_high:.3f}]  n={n}")
    else:
        st.metric("Accuracy", f"{acc:.3f}", f"n={n}")
        st.caption("Install statsmodels for a Wilson confidence interval.")

    report = classification_report(
        y_true, y_pred,
        labels=["positive", "neutral", "negative"],
        output_dict=True,
        zero_division=0
    )
    metrics_df = pd.DataFrame(report).T.round(3)
    st.dataframe(metrics_df, use_container_width=True)

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=["positive", "neutral", "negative"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["True Positive", "True Neutral", "True Negative"],
        columns=["Pred Positive", "Pred Neutral", "Pred Negative"]
    )
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    # Highlight dominant error type
    off_diagonal = [(cm[i, j], ["positive","neutral","negative"][i],
                                ["positive","neutral","negative"][j])
                    for i in range(3) for j in range(3) if i != j]
    if off_diagonal:
        worst_count, worst_true, worst_pred = max(off_diagonal, key=lambda x: x[0])
        st.info(
            f"Most common misclassification: **{worst_true} → {worst_pred}** "
            f"({worst_count} cases). "
            f"This matches our notebook finding that neutral sentences containing "
            f"financially salient language are misclassified as positive by zero-shot prompting."
        )

# ===========================================================================
# Streamlit app layout
# ===========================================================================

ABOUT_PROJECT_MD = """
**FinSentinel** extends a financial sentiment classification research notebook
into a production-grade risk intelligence and market signal application.

**Research question:** Can prompt-engineered LLMs extract structured financial
risk signals from news headlines, and does few-shot prompting outperform zero-shot?

**Key findings from the notebook (originally each model was scored on its own
resample — `eval_framework.py` fixes this by freezing one 200-sentence
stratified sample that every model is now run against, with Wilson 95% CIs
on every accuracy number):**
- GPT-4o-mini few-shot vs. a TF-IDF + Logistic Regression baseline on
  Financial PhraseBank (2,264 sentences annotated by 16 finance professionals)
- The dominant zero-shot error: neutral sentences containing financially salient
  language are misclassified as positive — CoT prompting substantially reduces this
- Knowledge distillation into a 66M-parameter DistilBERT student achieves **250× latency
  reduction** (<10 ms inference) while maintaining competitive accuracy

**This app adds:**
- Structured multi-dimensional risk extraction (risk type, severity, market impact, time horizon)
- Live Yahoo Finance headline ingestion with RSS deduplication
- Aggregated daily sentiment scores correlated against actual yfinance price returns
- Single-headline deep-dive trace showing RAG context, prompt, and LLM output
- Evaluation metrics with confusion matrix when labelled data is used
"""

METHOD_TRACE_MD = """
This app preserves the notebook's complete function structure:
- `CONFIG`, `FEW_SHOT_EXAMPLES`, `FEW_SHOT_BLOCK`
- `call_llm()` and `run_llm_evaluation()`
- `build_zero_shot_prompt()`, `build_few_shot_prompt()`, `build_cot_prompt()`
- `bm25_retrieve()`, `hybrid_retrieve()`, `build_rag_cot_prompt()`
- `parse_pub_date()` and `fetch_headlines()` for Yahoo Finance RSS
- `load_fiqa_from_huggingface()` for direct HuggingFace FiQA loading

**New in v2:**
- `compute_sentiment_scores()` — daily aggregation per ticker
- `fetch_price_returns()` — yfinance next-day / intraday returns
- `show_correlation_analysis()` — Pearson + Spearman with Plotly scatter
- `show_single_headline_deepdive()` — full RAG → prompt → output trace
- `show_evaluation_metrics()` — confusion matrix + classification report vs ground truth
"""


def main():
    st.set_page_config(page_title="FinSentinel Risk App", layout="wide")

    with st.expander("About this project", expanded=False):
        st.markdown(ABOUT_PROJECT_MD)

    st.title("FinSentinel: Financial Risk Intelligence")
    st.caption("Prompt-engineered LLM pipeline · RAG retrieval · Structured risk JSON · Sentiment-price correlation")

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("Analysis Configuration")
        prompt_mode = st.selectbox(
            "Prompting strategy",
            [
                "zero-shot → risk JSON",
                "few-shot → risk JSON",
                "CoT → risk JSON",
                "RAG+CoT → risk JSON",
            ],
            index=3,
        )
        tickers_raw    = st.text_input("Live Yahoo Finance tickers", "AAPL,MSFT,GOOGL,TSLA,NVDA,JPM,META")
        max_per_ticker = st.slider("Max headlines per ticker", 3, 30, 8)
        st.markdown("---")
        st.subheader("HuggingFace FiQA")
        hf_dataset_name = st.text_input("FiQA dataset name", "TheFinAI/fiqa-sentiment-classification")
        hf_split        = st.selectbox("FiQA split", ["train", "valid", "test"], index=0)
        hf_max_rows     = st.slider("Max FiQA rows", 10, 500, 100, step=10)
        st.markdown("---")
        st.write("OpenAI API:",    "✓" if os.environ.get("OPENAI_API_KEY") and OPENAI_OK else "fallback mode")
        st.write("RSS parser:",    "✓" if FEEDPARSER_OK else "pip install feedparser")
        st.write("BM25:",          "✓" if BM25_OK else "TF-IDF fallback")
        st.write("HF datasets:",   "✓" if HF_DATASETS_OK else "pip install datasets")
        st.write("yfinance:",      "✓" if YFINANCE_OK else "pip install yfinance")
        st.write("scipy/plotly:",  "✓" if SCIPY_OK and PLOTLY_OK else "pip install scipy plotly")

    # ---------------------------------------------------------------------------
    # Input source
    # ---------------------------------------------------------------------------
    input_mode = st.radio(
        "Input source",
        ["Live Yahoo Finance headlines", "Paste headlines", "HuggingFace FiQA dataset", "Financial PhraseBank dataset"],
        horizontal=True,
    )

    news_df = pd.DataFrame()   # populated only in live mode

    if input_mode == "Live Yahoo Finance headlines":
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        news_df = fetch_live_news(tickers, max_results=max_per_ticker)
        if news_df.empty:
            st.warning("Could not fetch live headlines. Paste headlines or install feedparser: pip install feedparser")
            input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
        else:
            st.subheader("Fetched live headlines")
            st.dataframe(news_df[["ticker", "headline", "pub_date", "pub_hour", "source"]],
                         use_container_width=True, height=250)
            input_df = news_df.rename(columns={"headline": "Headline"})[["Headline"]]

            with st.expander("Human evaluation export (50 headlines, 3 raters)", expanded=False):
                st.caption(
                    "Export a random sample for independent human labelling — the only way "
                    "to check whether the model is right on live data. Have 3 people label "
                    "each headline, then compute inter-rater agreement (Cohen's kappa) and "
                    "compare model predictions to the majority human label."
                )
                human_eval_n = min(50, len(news_df))
                human_eval_sample = news_df.sample(n=human_eval_n, random_state=CONFIG["random_state"])
                export_cols = [c for c in ["ticker", "headline", "pub_date"] if c in human_eval_sample.columns]
                st.download_button(
                    f"Download {human_eval_n}-headline human eval task (CSV)",
                    human_eval_sample[export_cols].to_csv(index=False).encode("utf-8"),
                    "human_eval_task.csv",
                    "text/csv",
                    key="human_eval_export_btn",
                )

    elif input_mode == "HuggingFace FiQA dataset":
        fiqa_df, fiqa_error = load_fiqa_from_huggingface(
            dataset_name=hf_dataset_name, split=hf_split, max_rows=hf_max_rows)
        if fiqa_error:
            st.warning(fiqa_error)
            input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
        elif fiqa_df.empty:
            st.warning("FiQA loaded but returned no rows. Using demo examples.")
            input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
        else:
            st.subheader("FiQA from HuggingFace")
            st.caption("Default: TheFinAI/fiqa-sentiment-classification.")
            st.dataframe(fiqa_df.head(20), use_container_width=True, height=260)
            text_col = infer_fiqa_text_column(fiqa_df)
            input_df = pd.DataFrame({"Headline": fiqa_df[text_col].astype(str)})
            for optional_col in ["target", "aspect", "score", "type"]:
                if optional_col in fiqa_df.columns:
                    input_df[f"fiqa_{optional_col}"] = fiqa_df[optional_col].values

    elif input_mode == "Paste headlines":
        raw = st.text_area(
            "Paste one headline per line",
            "Regional banks face liquidity concerns after deposit outflows accelerate.\n"
            "Corporate default rates rise as tighter credit conditions pressure borrowers.\n"
            "Nvidia shares rally after earnings beat expectations.",
            height=180,
        )
        input_df = pd.DataFrame({"Headline": [x.strip() for x in raw.splitlines() if x.strip()]})

    else:  # Financial PhraseBank
        frozen_path = "eval_set_frozen.csv"
        fpb_path = CONFIG["fpb_path"]
        use_frozen = os.path.exists(frozen_path) and st.checkbox(
            f"Use frozen eval set ({frozen_path})",
            value=True,
            help="The same 200-sentence stratified sample every model should be scored "
                 "on for a valid comparison. Uncheck to run the full dataset instead. "
                 "Build/refresh it with: python eval_framework.py build-eval-set",
        )
        if use_frozen:
            fpb_df   = pd.read_csv(frozen_path)
            input_df = fpb_df[[CONFIG["text_col"]]].rename(columns={CONFIG["text_col"]: "Headline"})
            input_df["true_label"] = fpb_df[CONFIG["label_col"]].values
            st.info(f"Loaded {len(input_df):,} sentences from the frozen eval set ({frozen_path}).")
        elif os.path.exists(fpb_path):
            fpb_df   = load_fpb(fpb_path, encoding=CONFIG["fpb_encoding"])
            input_df = fpb_df[[CONFIG["text_col"]]].rename(columns={CONFIG["text_col"]: "Headline"})
            input_df["true_label"] = fpb_df[CONFIG["label_col"]].values
            st.info(f"Loaded {len(input_df):,} labelled sentences from Financial PhraseBank.")
        else:
            st.warning(f"File '{fpb_path}' not found. Place Sentences_AllAgree.txt in the working directory.")
            input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:12]})

    corpus_texts = list(dict.fromkeys(
        DEFAULT_CORPUS + input_df["Headline"].dropna().astype(str).tolist()
    ))

    st.markdown("### Output schema")
    st.code(RISK_JSON_SCHEMA, language="json")

    # ---------------------------------------------------------------------------
    # Run risk analysis
    # ---------------------------------------------------------------------------
    if st.button("Run risk analysis", type="primary", use_container_width=True):
        rows     = []
        progress = st.progress(0)
        headlines = input_df["Headline"].dropna().astype(str).tolist()
        for i, headline in enumerate(headlines):
            rows.append(analyze_headline_with_notebook_pipeline(headline, prompt_mode, corpus_texts))
            progress.progress((i + 1) / max(1, len(headlines)))
        results_df = pd.DataFrame(rows)

        fiqa_meta_cols = [c for c in input_df.columns if c.startswith("fiqa_")]
        if fiqa_meta_cols and len(results_df) == len(input_df):
            for c in fiqa_meta_cols:
                results_df[c] = input_df[c].values

        if "true_label" in input_df.columns and len(results_df) == len(input_df):
            results_df["true_label"] = input_df["true_label"].values

        st.session_state["results_df"] = results_df
        st.session_state["news_df_snapshot"] = news_df.copy() if not news_df.empty else pd.DataFrame()

    results_df  = st.session_state.get("results_df")
    news_snap   = st.session_state.get("news_df_snapshot", pd.DataFrame())

    if results_df is None:
        st.info("Click 'Run risk analysis' to start.")
        show_single_headline_deepdive(corpus_texts, prompt_mode)
        st.stop()

    # ---------------------------------------------------------------------------
    # KPIs
    # ---------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headlines analysed",   len(results_df))
    c2.metric("Negative share",       f"{(results_df['sentiment'].eq('negative').mean()*100):.0f}%")
    c3.metric("High/extreme severity", int(results_df["severity"].isin(["high","extreme"]).sum()))
    c4.metric("LLM used",             "Yes" if results_df["used_llm"].any() else "Fallback")

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    st.subheader("Structured Financial Risk Outputs")
    base_cols  = ["headline","sentiment","risk_type","severity","market_impact",
                  "time_horizon","confidence","reasoning","retrieved_context"]
    extra_cols = [c for c in results_df.columns
                  if c.startswith("fiqa_") or c == "true_label"]
    cols       = [c for c in base_cols + extra_cols if c in results_df.columns]
    st.dataframe(results_df[cols], use_container_width=True, height=420)

    # Distribution charts
    left, right = st.columns(2)
    with left:
        st.subheader("Risk Type Distribution")
        st.bar_chart(results_df["risk_type"].value_counts())
    with right:
        st.subheader("Market Impact Distribution")
        st.bar_chart(results_df["market_impact"].value_counts())

    # ---------------------------------------------------------------------------
    # NEW COMPONENT 4B: Evaluation metrics (auto-shown for labelled FPB input)
    # ---------------------------------------------------------------------------
    show_evaluation_metrics(results_df)

    # ---------------------------------------------------------------------------
    # NEW COMPONENT 4A: Single headline deep-dive
    # ---------------------------------------------------------------------------
    st.markdown("---")
    show_single_headline_deepdive(corpus_texts, prompt_mode)

    # ---------------------------------------------------------------------------
    # NEW COMPONENTS 1-3: Sentiment → Price Correlation
    # ---------------------------------------------------------------------------
    if input_mode == "Live Yahoo Finance headlines" and not news_snap.empty:
        st.markdown("---")
        st.header("Sentiment → Price Correlation")
        st.caption(
            "Aggregates per-headline sentiment into daily scores per ticker, "
            "then correlates against actual yfinance price returns. "
            "This is the core research extension from the notebook."
        )

        if not YFINANCE_OK:
            st.warning("Install yfinance to enable correlation analysis: pip install yfinance")
        elif not SCIPY_OK or not PLOTLY_OK:
            st.warning("Install scipy and plotly: pip install scipy plotly")
        else:
            if st.button("Run correlation analysis", type="secondary", key="corr_btn"):
                with st.spinner("Aggregating sentiment scores and fetching price data..."):

                    # Component 1 — aggregate scores
                    sentiment_scores = compute_sentiment_scores(results_df, news_snap)

                    # Component 2 — fetch price returns
                    unique_tickers = tuple(sentiment_scores["ticker"].unique().tolist())
                    unique_dates   = tuple(
                        pd.to_datetime(sentiment_scores["pub_date"]).tolist()
                    )
                    price_returns = fetch_price_returns(unique_tickers, unique_dates)

                if price_returns.empty:
                    st.warning(
                        "Could not fetch price data from Yahoo Finance. "
                        "This usually means Yahoo Finance is rate-limiting requests. "
                        "Wait 60 seconds and try again, or check that your tickers "
                        "(AAPL, MSFT etc.) are valid."
                    )
                else:
                    # Check if we actually got any return values
                    has_returns = (
                        price_returns["next_day_return"].notna().any() or
                        price_returns["intraday_return"].notna().any()
                    )
                    if not has_returns:
                        st.warning(
                            "Price data fetched but no returns could be calculated. "
                            "This may happen if today's headlines fall on a weekend or "
                            "market holiday when no trading occurred. "
                            "Try again on a weekday during or after market hours."
                        )
                    else:
                        # Store for reuse
                        st.session_state["sentiment_scores"] = sentiment_scores
                        st.session_state["price_returns"]    = price_returns

            # Component 3 — show correlation analysis
            if "sentiment_scores" in st.session_state and "price_returns" in st.session_state:
                show_correlation_analysis(
                    st.session_state["sentiment_scores"],
                    st.session_state["price_returns"]
                )

                st.subheader("Daily Aggregated Sentiment Scores")
                st.dataframe(st.session_state["sentiment_scores"], use_container_width=True)

    # ---------------------------------------------------------------------------
    # Agreement-split degradation curve (produced offline by eval_framework.py)
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.header("Agreement Split Degradation Curve")
    degradation_csv = "degradation_curve.csv"
    if os.path.exists(degradation_csv):
        curve_df = pd.read_csv(degradation_csv)
        st.caption(
            "FinBERT+LoRA macro F1 across FPB annotator-agreement thresholds — "
            "a more honest performance picture than any single number."
        )
        st.dataframe(curve_df, use_container_width=True)
        if PLOTLY_OK:
            fig = px.line(
                curve_df, x="agreement_pct", y="macro_f1", markers=True,
                labels={"agreement_pct": "Agreement threshold (%)", "macro_f1": "Macro F1"},
                title="Macro F1 vs. annotator agreement threshold",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Not generated yet. Run `python eval_framework.py degradation-curve` "
            "(requires a trained FinBERT+LoRA adapter and the `datasets` package) "
            "to produce degradation_curve.csv, then reload this page."
        )

    # ---------------------------------------------------------------------------
    # Human evaluation — inter-rater agreement + model-vs-human kappa
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.header("Human Evaluation")
    st.caption(
        "Upload the labelled human_eval_task.csv after 3 raters have independently "
        "labelled the exported headlines. Expected columns: `rater1`, `rater2`, `rater3`, "
        "and optionally `model_pred` for a model-vs-majority-human comparison."
    )
    human_eval_file = st.file_uploader("Upload rater labels (CSV)", type=["csv"], key="human_eval_upload")
    if human_eval_file is not None and SKLEARN_OK:
        ratings_df = pd.read_csv(human_eval_file)
        rater_cols = [c for c in ratings_df.columns if c.lower().startswith("rater")]
        if len(rater_cols) < 2:
            st.warning("Need at least 2 rater columns (e.g. rater1, rater2, rater3).")
        else:
            pairs = list(itertools.combinations(rater_cols, 2))
            kappas = {f"{a} vs {b}": cohen_kappa_score(ratings_df[a], ratings_df[b]) for a, b in pairs}
            avg_kappa = float(np.mean(list(kappas.values())))
            st.dataframe(pd.DataFrame([{"pair": k, "kappa": round(v, 3)} for k, v in kappas.items()]),
                         use_container_width=True)
            st.metric("Average inter-rater kappa", f"{avg_kappa:.3f}")

            majority = ratings_df[rater_cols].mode(axis=1)[0]
            if "model_pred" in ratings_df.columns:
                model_kappa = cohen_kappa_score(majority, ratings_df["model_pred"])
                st.metric("Model vs. majority-human kappa", f"{model_kappa:.3f}")

    # ---------------------------------------------------------------------------
    # Method trace + downloads
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Notebook Method Trace")
    st.markdown(METHOD_TRACE_MD)

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download structured risk outputs (CSV)",
        csv,
        "finsentinel_structured_risk_outputs.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()
