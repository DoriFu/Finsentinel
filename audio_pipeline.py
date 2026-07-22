"""
FinSentinel Phase 3: audio pipeline
====================================
Text headlines are lagging indicators — by the time a headline is published,
the market has often already moved. Audio from earnings calls, financial
podcasts, and market commentary contains forward-looking language that often
moves markets before headlines are written: CEO tone, analyst confidence,
specific guidance phrases.

Pipeline: YouTube / podcast URL or local file -> Whisper transcript ->
pyannote speaker diarization -> align transcript segments to speakers ->
FinBERT sentiment + forward-looking-language detection per speaker turn ->
fuse with the existing headline sentiment signal into one unified score per
ticker/day (audio weighted higher than text — it leads, headlines lag).

CLI:
    python audio_pipeline.py download <youtube_url> [--out audio.mp3]
    python audio_pipeline.py transcribe <audio.mp3> [--model medium]
    python audio_pipeline.py diarize <audio.mp3> [--hf-token TOKEN]
    python audio_pipeline.py analyze <audio.mp3> --ticker AAPL [--hf-token TOKEN]
    python audio_pipeline.py edgar-transcript --ticker AAPL --year 2025

All library functions can also be imported directly:
    from audio_pipeline import transcribe_audio, diarize_audio, align_speakers
"""

import os
import re
import subprocess
import itertools
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from finsentinel_v3 import CONFIG, logger, fallback_structured_risk

warnings.filterwarnings("ignore")

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import whisper
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_OK = True
except Exception:
    PYANNOTE_OK = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False


# ---------------------------------------------------------------------------
# 1-2. Download audio from YouTube / podcast
# ---------------------------------------------------------------------------

def download_audio(url, output_path="audio.mp3"):
    """
    Download the audio track from a YouTube (or other yt-dlp-supported) URL.
    Works for earnings call recordings, financial podcast episodes, CNBC /
    Bloomberg interview clips, conference talks.
    """
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_path,
        "--no-playlist",
        url,
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    latency_ms = (time.time() - start) * 1000
    if result.returncode != 0:
        logger.error(f"download_audio FAILED url={url} latency={latency_ms:.0f}ms error={result.stderr[:300]}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    logger.info(f"download_audio OK url={url} out={output_path} latency={latency_ms:.0f}ms")
    return output_path


# ---------------------------------------------------------------------------
# 3. Transcribe with Whisper
# ---------------------------------------------------------------------------

_WHISPER_MODEL_CACHE = {}


def _get_whisper_model(model_size="medium"):
    if model_size not in _WHISPER_MODEL_CACHE:
        _WHISPER_MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _WHISPER_MODEL_CACHE[model_size]


def transcribe_audio(audio_path, model_size="medium"):
    """
    Transcribe audio to timestamped segments. model_size: tiny/base/small/
    medium/large — medium is the best speed/accuracy balance for financial
    terms ("EPS", "EBITDA", "guidance"); small models tend to miss these.
    """
    if not WHISPER_OK:
        raise RuntimeError("openai-whisper not installed — cannot transcribe audio")

    start = time.time()
    model = _get_whisper_model(model_size)
    result = model.transcribe(audio_path, language="en", word_timestamps=True, verbose=False)
    latency_ms = (time.time() - start) * 1000

    segments = [{
        "start": round(seg["start"], 2),
        "end":   round(seg["end"], 2),
        "text":  seg["text"].strip(),
    } for seg in result["segments"]]

    logger.info(f"transcribe_audio OK audio={audio_path} model={model_size} "
                f"segments={len(segments)} latency={latency_ms:.0f}ms")
    return segments


# ---------------------------------------------------------------------------
# 4. Speaker diarization — who said what
# ---------------------------------------------------------------------------

def diarize_audio(audio_path, hf_token=None):
    """
    Identify and separate speakers. Returns a list of {speaker, start, end}
    turns. For earnings calls: typically CEO, CFO, moderator, analysts.
    Requires a HuggingFace token with access to pyannote/speaker-diarization-3.1.
    """
    if not PYANNOTE_OK:
        raise RuntimeError("pyannote.audio not installed — cannot diarize speakers")

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("A HuggingFace token is required for pyannote speaker diarization")

    start = time.time()
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    if TORCH_OK:
        pipeline = pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    diarization = pipeline(audio_path)
    speaker_turns = [{
        "speaker": speaker,
        "start":   round(turn.start, 2),
        "end":     round(turn.end, 2),
    } for turn, _, speaker in diarization.itertracks(yield_label=True)]

    latency_ms = (time.time() - start) * 1000
    logger.info(f"diarize_audio OK audio={audio_path} turns={len(speaker_turns)} latency={latency_ms:.0f}ms")
    return speaker_turns


# ---------------------------------------------------------------------------
# 5. Align transcription with speaker turns
# ---------------------------------------------------------------------------

def align_speakers(segments, speaker_turns):
    """Match each Whisper segment to a speaker by timestamp-midpoint overlap."""
    enriched = []
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        speaker = "UNKNOWN"
        for turn in speaker_turns:
            if turn["start"] <= seg_mid <= turn["end"]:
                speaker = turn["speaker"]
                break
        enriched.append({**seg, "speaker": speaker})
    return enriched


def group_speaker_blocks(aligned_segments):
    """Group consecutive same-speaker segments into contiguous speaking blocks."""
    speaker_blocks = []
    for speaker, group in itertools.groupby(aligned_segments, key=lambda x: x["speaker"]):
        segs = list(group)
        full_text = " ".join(s["text"] for s in segs)
        speaker_blocks.append({
            "speaker":  speaker,
            "start":    segs[0]["start"],
            "end":      segs[-1]["end"],
            "text":     full_text,
            "duration": segs[-1]["end"] - segs[0]["start"],
        })
    return speaker_blocks


# ---------------------------------------------------------------------------
# 6. Financial signal extraction per speaker turn
# ---------------------------------------------------------------------------

FORWARD_LOOKING_PATTERNS = {
    "positive_guidance": r"\b(expect|anticipate|project|forecast|guide|outlook)\b.*"
                          r"\b(growth|increase|improve|strong|above)\b",
    "negative_guidance": r"\b(expect|anticipate|project|forecast)\b.*"
                          r"\b(decline|decrease|below|challenging|headwind)\b",
    "uncertainty":       r"\b(uncertain|volatile|unpredictable|unclear|risk)\b",
    "beat":              r"\b(beat|exceeded|above expectations|record)\b",
    "miss":              r"\b(missed|below expectations|disappointed|shortfall)\b",
}

FIN_TERMS_PATTERN = re.compile(
    r'\b(revenue|earnings|EPS|guidance|margin|growth|'
    r'forecast|outlook|quarter|full.year|EBITDA)\b',
    re.IGNORECASE,
)

_FINBERT_CACHE = {}


def _load_finbert(model_name=None, adapter_dir=None):
    key = (model_name or CONFIG["finbert_model"], adapter_dir)
    if key in _FINBERT_CACHE:
        return _FINBERT_CACHE[key]
    if not TRANSFORMERS_OK:
        raise RuntimeError("transformers/torch not installed — cannot run FinBERT")

    model_name = model_name or CONFIG["finbert_model"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if adapter_dir:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
        except Exception as e:
            raise RuntimeError("peft not installed or adapter load failed — cannot apply LoRA adapter") from e
    model.eval()
    _FINBERT_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def score_sentiment(text, model_name=None, adapter_dir=None):
    """
    Sentiment probabilities for one piece of text. Uses FinBERT (optionally
    with a LoRA adapter) when transformers/torch are available; otherwise
    falls back to the same keyword-based scorer the rest of the app uses so
    the audio pipeline still runs end-to-end without heavy ML dependencies.
    """
    if TRANSFORMERS_OK:
        try:
            tokenizer, model = _load_finbert(model_name, adapter_dir)
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
            label2prob = {id2label[i]: p for i, p in enumerate(probs)}
            label = max(label2prob, key=label2prob.get)
            return {
                "label":       label,
                "p_positive":  label2prob.get("positive", 0.0),
                "p_negative":  label2prob.get("negative", 0.0),
                "p_neutral":   label2prob.get("neutral", 0.0),
                "confidence":  max(probs),
            }
        except Exception as e:
            logger.warning(f"score_sentiment FINBERT_FAILED error={e}, using fallback")

    fallback = fallback_structured_risk(text)
    label = fallback["sentiment"]
    conf = fallback["confidence"]
    return {
        "label":      label,
        "p_positive": conf if label == "positive" else (1 - conf) / 2,
        "p_negative": conf if label == "negative" else (1 - conf) / 2,
        "p_neutral":  conf if label == "neutral" else (1 - conf) / 2,
        "confidence": conf,
    }


def extract_audio_signals(speaker_block, ticker, model_name=None, adapter_dir=None):
    text = speaker_block["text"]
    speaker = speaker_block["speaker"]

    sentiment = score_sentiment(text, model_name=model_name, adapter_dir=adapter_dir)

    lower = text.lower()
    fwd_signals = [name for name, pattern in FORWARD_LOOKING_PATTERNS.items() if re.search(pattern, lower)]
    fin_terms = set(m.lower() for m in FIN_TERMS_PATTERN.findall(text))

    return {
        "ticker":       ticker,
        "speaker":      speaker,
        "start":        speaker_block["start"],
        "text_snippet": text[:200],
        "sentiment":    sentiment["label"],
        "p_positive":   sentiment["p_positive"],
        "p_negative":   sentiment["p_negative"],
        "confidence":   sentiment["confidence"],
        "fwd_signals":  len(fwd_signals),
        "fwd_signal_types": ",".join(fwd_signals),
        "fin_terms":    len(fin_terms),
    }


def analyze_speaker_blocks(speaker_blocks, ticker, model_name=None, adapter_dir=None):
    rows = [extract_audio_signals(b, ticker, model_name=model_name, adapter_dir=adapter_dir)
            for b in speaker_blocks]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Fuse audio + text signals -> unified market predictor
# ---------------------------------------------------------------------------

DEFAULT_SPEAKER_WEIGHTS = {
    "SPEAKER_00": 1.5,  # CEO — typically the largest speaking share
    "SPEAKER_01": 1.2,  # CFO
    "SPEAKER_02": 0.8,  # Analyst
    "UNKNOWN":    0.5,
}


def compute_unified_signal(audio_df, text_df, ticker, date,
                            audio_weight=0.6, text_weight=0.4,
                            speaker_weights=None, text_score_col="sentiment_score"):
    """
    Fuse the audio (earnings-call) signal and the text (headline) signal into
    one weighted score. Audio is weighted higher because earnings-call
    guidance is forward-looking (leading), while headlines report what
    already happened (lagging) — CEO/CFO tone has documented predictive
    power in the literature.
    """
    speaker_weights = speaker_weights or DEFAULT_SPEAKER_WEIGHTS

    audio_scores = audio_df[audio_df["ticker"] == ticker].copy()
    if len(audio_scores):
        audio_scores["weight"] = audio_scores["speaker"].map(speaker_weights).fillna(0.7)
        audio_scores["weighted_sent"] = (
            (audio_scores["p_positive"] - audio_scores["p_negative"])
            * audio_scores["weight"] * audio_scores["confidence"]
        )
        audio_signal = audio_scores["weighted_sent"].sum() / audio_scores["weight"].sum()
    else:
        audio_signal = 0.0

    text_rows = text_df[(text_df["ticker"] == ticker) & (text_df["pub_date"] == date)]
    text_signal = text_rows[text_score_col].mean() if len(text_rows) else 0.0
    text_signal = 0.0 if pd.isna(text_signal) else text_signal

    unified = audio_weight * audio_signal + text_weight * text_signal
    return {
        "ticker":        ticker,
        "date":          date,
        "audio_signal":  round(float(audio_signal), 4),
        "text_signal":   round(float(text_signal), 4),
        "unified_score": round(float(unified), 4),
    }


# ---------------------------------------------------------------------------
# 8. Earnings call transcripts via SEC EDGAR (free, no YouTube needed)
# ---------------------------------------------------------------------------

SEC_EDGAR_USER_AGENT = os.environ.get("SEC_EDGAR_USER_AGENT", "FinSentinel research finsentinel@nyu.edu")


def fetch_earnings_transcript_filings(ticker, year, forms="8-K"):
    """
    SEC EDGAR full-text search for a ticker's earnings-related 8-K filings in
    a given year — more reliable than scraping YouTube and covers far more
    companies. Returns the raw list of filing hits (most recent first).
    """
    if not REQUESTS_OK:
        raise RuntimeError("requests not installed — cannot query SEC EDGAR")

    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": '"earnings call"',
        "dateRange": "custom",
        "startdt": f"{year}-01-01",
        "enddt": f"{year}-12-31",
        "entityName": ticker,
        "forms": forms,
    }
    start = time.time()
    resp = requests.get(url, params=params, headers={"User-Agent": SEC_EDGAR_USER_AGENT}, timeout=15)
    latency_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    filings = resp.json().get("hits", {}).get("hits", [])
    logger.info(f"fetch_earnings_transcript_filings OK ticker={ticker} year={year} "
                f"filings={len(filings)} latency={latency_ms:.0f}ms")
    return filings


def fetch_earnings_transcript(ticker, year):
    """Convenience wrapper: (file_date, period_of_report) of the most recent matching filing, or None."""
    filings = fetch_earnings_transcript_filings(ticker, year)
    if not filings:
        return None
    source = filings[0].get("_source", {})
    return source.get("file_date"), source.get("period_of_report")


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------

def run_audio_pipeline(audio_path, ticker, hf_token=None, whisper_model="medium",
                        finbert_model=None, adapter_dir=None):
    """Full pipeline for one audio file: transcribe -> diarize -> align -> extract signals."""
    segments = transcribe_audio(audio_path, model_size=whisper_model)
    speaker_turns = diarize_audio(audio_path, hf_token=hf_token)
    aligned = align_speakers(segments, speaker_turns)
    speaker_blocks = group_speaker_blocks(aligned)
    audio_df = analyze_speaker_blocks(speaker_blocks, ticker, model_name=finbert_model, adapter_dir=adapter_dir)
    return audio_df, speaker_blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="FinSentinel Phase 3 audio pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download audio from a YouTube/podcast URL")
    p_dl.add_argument("url")
    p_dl.add_argument("--out", default="audio.mp3")

    p_tr = sub.add_parser("transcribe", help="Transcribe an audio file with Whisper")
    p_tr.add_argument("audio_path")
    p_tr.add_argument("--model", default="medium")

    p_dz = sub.add_parser("diarize", help="Speaker-diarize an audio file")
    p_dz.add_argument("audio_path")
    p_dz.add_argument("--hf-token", default=None)

    p_an = sub.add_parser("analyze", help="Full pipeline: transcribe + diarize + extract signals")
    p_an.add_argument("audio_path")
    p_an.add_argument("--ticker", required=True)
    p_an.add_argument("--hf-token", default=None)
    p_an.add_argument("--whisper-model", default="medium")
    p_an.add_argument("--adapter-dir", default=None)
    p_an.add_argument("--out-csv", default="audio_signals.csv")

    p_edgar = sub.add_parser("edgar-transcript", help="Look up earnings-call 8-K filings via SEC EDGAR")
    p_edgar.add_argument("--ticker", required=True)
    p_edgar.add_argument("--year", type=int, required=True)

    args = parser.parse_args()

    if args.command == "download":
        path = download_audio(args.url, output_path=args.out)
        print(f"Downloaded to {path}")

    elif args.command == "transcribe":
        segments = transcribe_audio(args.audio_path, model_size=args.model)
        print(f"Transcribed {len(segments)} segments")
        for seg in segments[:5]:
            print(f"  [{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")

    elif args.command == "diarize":
        turns = diarize_audio(args.audio_path, hf_token=args.hf_token)
        print(f"Found {len(turns)} speaker turns")
        for t in turns[:10]:
            print(f"  {t['speaker']}: {t['start']:.1f}-{t['end']:.1f}")

    elif args.command == "analyze":
        audio_df, speaker_blocks = run_audio_pipeline(
            args.audio_path, args.ticker, hf_token=args.hf_token,
            whisper_model=args.whisper_model, adapter_dir=args.adapter_dir,
        )
        print(audio_df.to_string(index=False))
        audio_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")

    elif args.command == "edgar-transcript":
        result = fetch_earnings_transcript(args.ticker, args.year)
        print(result if result else "No matching filings found")


if __name__ == "__main__":
    _cli()
