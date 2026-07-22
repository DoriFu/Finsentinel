"""
FinSentinel evaluation framework (Phase 1)
===========================================
Fixes the "every model gets a different test set" problem: freezes one
stratified 200-sentence sample from the FPB held-out split and reuses it
for every model. Adds Wilson 95% confidence intervals to every accuracy
number, an agreement-split degradation curve, a market-controlled partial
correlation test, and a human-evaluation (inter-rater kappa) pipeline.

CLI:
    python eval_framework.py build-eval-set
    python eval_framework.py run-baselines [--with-finbert]
    python eval_framework.py degradation-curve
    python eval_framework.py human-eval-export --live-csv news.csv

All library functions can also be imported directly:
    from eval_framework import build_frozen_eval_set, evaluate_with_ci
"""

import os
import itertools
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportion_confint

from finsentinel_v3 import (
    CONFIG,
    DEFAULT_CORPUS,
    load_fpb,
    analyze_headline_with_notebook_pipeline,
)

warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
    HF_DATASETS_OK = True
except Exception:
    HF_DATASETS_OK = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except Exception:
    YFINANCE_OK = False

try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------------------------------------------------------------------------
# Frozen evaluation set — the single change that makes the comparison table
# scientifically valid. Every model, from the majority baseline through every
# GPT prompting condition, is scored on this exact 200-row sample.
# ---------------------------------------------------------------------------

EVAL_SEED = 42
EVAL_SAMPLE_SIZE = 200
FROZEN_EVAL_PATH = "eval_set_frozen.csv"


def build_train_test_split(fpb_path=None, test_size=None, random_state=None):
    """Stratified train/test split of the full FinancialPhraseBank file."""
    fpb_path = fpb_path or CONFIG["fpb_path"]
    test_size = CONFIG["test_size"] if test_size is None else test_size
    random_state = CONFIG["random_state"] if random_state is None else random_state

    df = load_fpb(fpb_path, encoding=CONFIG["fpb_encoding"])
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[CONFIG["label_col"]],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_frozen_eval_set(test_df=None, n=EVAL_SAMPLE_SIZE, seed=EVAL_SEED,
                           out_path=FROZEN_EVAL_PATH):
    """
    Draw one stratified n-sentence sample from the held-out test split and
    freeze it to disk. Call this ONCE. Every model from then on loads
    `out_path` via load_frozen_eval_set() — never resample.
    """
    if test_df is None:
        _, test_df = build_train_test_split()

    if len(test_df) <= n:
        eval_sample = test_df.copy()
    else:
        eval_sample, _ = train_test_split(
            test_df,
            test_size=len(test_df) - n,
            stratify=test_df[CONFIG["label_col"]],
            random_state=seed,
        )

    eval_sample = eval_sample.reset_index(drop=True)
    eval_sample.to_csv(out_path, index=False)
    return eval_sample


def load_frozen_eval_set(path=FROZEN_EVAL_PATH):
    """Load the frozen eval set. Raises if it hasn't been built yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_frozen_eval_set() once to create it, "
            "then reuse that same file for every model comparison."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Wilson 95% CI on every metric
# ---------------------------------------------------------------------------

def evaluate_with_ci(y_true, y_pred, model_name="model", verbose=True):
    """Accuracy + macro F1 with a Wilson 95% CI on accuracy."""
    y_true = pd.Series(y_true).astype(str).str.lower().str.strip().reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).str.lower().str.strip().reset_index(drop=True)

    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    n_correct = int(round(acc * n))
    lo, hi = proportion_confint(n_correct, n, alpha=0.05, method="wilson")

    result = {
        "model": model_name,
        "n": n,
        "accuracy": acc,
        "ci_low": lo,
        "ci_high": hi,
        "macro_f1": macro,
    }
    if verbose:
        print(model_name)
        print(f"  Accuracy:  {acc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  Macro F1:  {macro:.3f}")
        print(f"  n = {n}")
    return result


def build_comparison_table(results):
    """results: list of dicts returned by evaluate_with_ci()."""
    rows = [{
        "model":    r["model"],
        "n":        r["n"],
        "accuracy": round(r["accuracy"], 3),
        "ci_low":   round(r["ci_low"], 3),
        "ci_high":  round(r["ci_high"], 3),
        "macro_f1": round(r["macro_f1"], 3),
    } for r in results]
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline model runners — all consume the same frozen eval_df
# ---------------------------------------------------------------------------

def predict_majority(train_df, eval_df):
    majority_label = train_df[CONFIG["label_col"]].mode().iloc[0]
    return [majority_label] * len(eval_df)


def predict_tfidf_logreg(train_df, eval_df, max_features=None, ngram_range=None):
    max_features = max_features or CONFIG["tfidf_max_features"]
    ngram_range = ngram_range or CONFIG["tfidf_ngram_range"]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_df[CONFIG["text_col"]])
    X_eval = vectorizer.transform(eval_df[CONFIG["text_col"]])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_df[CONFIG["label_col"]])
    return clf.predict(X_eval).tolist()


def predict_finbert_zero_shot(eval_df, model_name=None, batch_size=16):
    """Zero-shot inference with the pretrained FinBERT checkpoint (no fine-tuning)."""
    if not TRANSFORMERS_OK:
        raise RuntimeError("transformers/torch not installed — cannot run FinBERT zero-shot")

    model_name = model_name or CONFIG["finbert_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    texts = eval_df[CONFIG["text_col"]].tolist()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            logits = model(**enc).logits
            batch_preds = logits.argmax(dim=-1).tolist()
            preds.extend(id2label[p] for p in batch_preds)
    return preds


def train_finbert_lora(train_df, model_name=None, output_dir="finbert_lora_adapter"):
    """
    Fine-tune FinBERT with a LoRA adapter on train_df using the hyperparameters
    frozen in CONFIG (lora_r / lora_alpha / lora_dropout / lora_epochs / lora_lr).
    Requires `peft` + a GPU for reasonable runtime; not exercised in CI.
    """
    if not TRANSFORMERS_OK:
        raise RuntimeError("transformers/torch not installed — cannot train FinBERT+LoRA")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise RuntimeError("peft not installed — cannot train FinBERT+LoRA") from e

    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    model_name = model_name or CONFIG["finbert_model"]
    label2id = CONFIG["label_map"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id)
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config)

    class FPBDataset(Dataset):
        def __init__(self, df):
            self.texts = df[CONFIG["text_col"]].tolist()
            self.labels = [label2id[l] for l in df[CONFIG["label_col"]]]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, max_length=128, padding="max_length")
            enc = {k: torch.tensor(v) for k, v in enc.items()}
            enc["labels"] = torch.tensor(self.labels[idx])
            return enc

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["lora_epochs"],
        per_device_train_batch_size=CONFIG["lora_batch_size"],
        learning_rate=CONFIG["lora_lr"],
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=FPBDataset(train_df))
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def predict_finbert_lora(eval_df, adapter_dir="finbert_lora_adapter", base_model_name=None,
                          batch_size=16):
    if not TRANSFORMERS_OK:
        raise RuntimeError("transformers/torch not installed — cannot run FinBERT+LoRA")
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft not installed — cannot run FinBERT+LoRA") from e

    base_model_name = base_model_name or CONFIG["finbert_model"]
    id2label = {v: k for k, v in CONFIG["label_map"].items()}

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(CONFIG["label_map"])
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    texts = eval_df[CONFIG["text_col"]].tolist()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            logits = model(**enc).logits
            batch_preds = logits.argmax(dim=-1).tolist()
            preds.extend(id2label[p] for p in batch_preds)
    return preds


def predict_gpt_condition(eval_df, prompt_mode, corpus_texts=None):
    """
    Runs the existing notebook GPT pipeline (zero-shot / few-shot / CoT / RAG+CoT)
    from finsentinel_v3 on every row of the frozen eval set.
    """
    corpus_texts = corpus_texts or DEFAULT_CORPUS
    preds = []
    for headline in eval_df[CONFIG["text_col"]]:
        result = analyze_headline_with_notebook_pipeline(headline, prompt_mode, corpus_texts)
        preds.append(result.get("sentiment", "neutral"))
    return preds


GPT_CONDITIONS = [
    "zero-shot → risk JSON",
    "few-shot → risk JSON",
    "CoT → risk JSON",
    "RAG+CoT → risk JSON",
]


def run_full_model_comparison(train_df=None, eval_df=None, with_finbert_zero_shot=False,
                               with_gpt=False, verbose=True):
    """
    Runs majority, TF-IDF, (optionally) FinBERT zero-shot, and (optionally) every
    GPT prompting condition on the same frozen eval_df. Returns the comparison table.
    """
    if train_df is None:
        train_df, _ = build_train_test_split()
    if eval_df is None:
        eval_df = load_frozen_eval_set()

    y_true = eval_df[CONFIG["label_col"]]
    results = []

    results.append(evaluate_with_ci(y_true, predict_majority(train_df, eval_df),
                                     "Majority baseline", verbose))
    results.append(evaluate_with_ci(y_true, predict_tfidf_logreg(train_df, eval_df),
                                     "TF-IDF + LogisticRegression", verbose))

    if with_finbert_zero_shot:
        results.append(evaluate_with_ci(y_true, predict_finbert_zero_shot(eval_df),
                                         "FinBERT (zero-shot)", verbose))

    if with_gpt:
        for mode in GPT_CONDITIONS:
            results.append(evaluate_with_ci(y_true, predict_gpt_condition(eval_df, mode),
                                             f"GPT-4o-mini ({mode})", verbose))

    return build_comparison_table(results)


# ---------------------------------------------------------------------------
# Agreement-split degradation curve
# ---------------------------------------------------------------------------

AGREEMENT_CONFIGS = ["sentences_50agree", "sentences_66agree",
                     "sentences_75agree", "sentences_allagree"]
AGREEMENT_THRESHOLDS = {"sentences_50agree": 50, "sentences_66agree": 66,
                        "sentences_75agree": 75, "sentences_allagree": 100}


def load_agreement_split(config_name):
    if not HF_DATASETS_OK:
        raise RuntimeError("`datasets` not installed — cannot load financial_phrasebank splits")
    ds = load_dataset("financial_phrasebank", config_name, trust_remote_code=True)
    df = ds["train"].to_pandas().rename(columns={"sentence": CONFIG["text_col"]})
    label_names = ds["train"].features["label"].names
    df[CONFIG["label_col"]] = df["label"].map(lambda i: label_names[i].lower())
    return df[[CONFIG["text_col"], CONFIG["label_col"]]]


def run_agreement_degradation_curve(predict_fn, configs=AGREEMENT_CONFIGS, verbose=True):
    """
    predict_fn(df) -> list of predicted labels, same length/order as df.
    Typically the FinBERT+LoRA model — pass predict_finbert_lora bound to an
    eval_df built per split, e.g. `lambda df: predict_finbert_lora(df, adapter_dir)`.
    """
    results = []
    for cfg in configs:
        df = load_agreement_split(cfg)
        y_pred = predict_fn(df)
        macro = f1_score(df[CONFIG["label_col"]], y_pred, average="macro", zero_division=0)
        row = {"config": cfg, "agreement_pct": AGREEMENT_THRESHOLDS[cfg],
               "macro_f1": macro, "n": len(df)}
        results.append(row)
        if verbose:
            print(f"{cfg}: macro F1 = {macro:.3f}  (n={len(df)})")
    return pd.DataFrame(results)


def plot_degradation_curve(curve_df, out_path=None):
    """Returns a Plotly figure; writes HTML to out_path if given (no kaleido needed)."""
    if not PLOTLY_OK:
        raise RuntimeError("plotly not installed — cannot plot degradation curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["agreement_pct"], y=curve_df["macro_f1"],
        mode="lines+markers+text",
        text=[f"n={n}" for n in curve_df["n"]],
        textposition="top center",
    ))
    fig.update_layout(
        title="FinBERT+LoRA: macro F1 vs. annotator agreement threshold",
        xaxis_title="Agreement threshold (%)",
        yaxis_title="Macro F1",
        template="plotly_white",
        height=420,
    )
    if out_path:
        fig.write_html(out_path)
    return fig


# ---------------------------------------------------------------------------
# Market-controlled (partial) correlation
# ---------------------------------------------------------------------------

def fetch_spy_returns(start_date, end_date):
    if not YFINANCE_OK:
        raise RuntimeError("yfinance not installed — cannot fetch SPY control series")
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy.reset_index().rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"]).dt.date
    spy["spy_ret"] = spy["Close"].pct_change().shift(-1)
    return spy[["date", "spy_ret"]]


def _residualize(x, control):
    """OLS residuals of x regressed on [1, control] — one coefficient pair, applied per row."""
    x = np.asarray(x, dtype=float)
    control = np.asarray(control, dtype=float)
    design = np.column_stack([np.ones_like(control), control])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def partial_correlation(x, y, control):
    """
    Partial correlation between x and y controlling for `control`, computed by
    residualizing both x and y on control (with its own intercept) and taking
    the Pearson correlation of the residuals.
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy not installed — cannot compute partial correlation")
    df = pd.DataFrame({"x": x, "y": y, "c": control}).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    x_resid = _residualize(df["x"], df["c"])
    y_resid = _residualize(df["y"], df["c"])
    r, p = scipy_stats.pearsonr(x_resid, y_resid)
    return r, p, len(df)


def market_controlled_correlation(merged_df, sentiment_col="sentiment_score",
                                   return_col="next_day_return", control_col="spy_ret"):
    """
    Compares the raw sentiment→return correlation against the same correlation
    after partialling out the market-wide (SPY) return, so a high raw r that's
    really just beta to the market gets exposed.
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy not installed")
    clean_raw = merged_df[[sentiment_col, return_col]].dropna()
    if len(clean_raw) >= 2:
        raw_r, raw_p = scipy_stats.pearsonr(clean_raw[sentiment_col], clean_raw[return_col])
    else:
        raw_r, raw_p = float("nan"), float("nan")

    partial_r, partial_p, n = partial_correlation(
        merged_df[sentiment_col], merged_df[return_col], merged_df[control_col]
    )
    result = {
        "raw_r": raw_r, "raw_p": raw_p,
        "partial_r": partial_r, "partial_p": partial_p,
        "n": n,
    }
    print(f"Raw r (uncontrolled):          {raw_r:.3f}  p={raw_p:.4f}")
    print(f"Partial r (controlling SPY):  {partial_r:.3f}  p={partial_p:.4f}  n={n}")
    return result


# ---------------------------------------------------------------------------
# Human evaluation — 50 headlines, 3 raters, Cohen's kappa
# ---------------------------------------------------------------------------

def export_human_eval_task(live_headlines_df, n=50, seed=EVAL_SEED, out_path="human_eval_task.csv"):
    cols = [c for c in ["ticker", "headline", "pub_date"] if c in live_headlines_df.columns]
    sample = live_headlines_df.sample(n=min(n, len(live_headlines_df)), random_state=seed)
    sample[cols].to_csv(out_path, index=False)
    return sample[cols]


def compute_inter_rater_agreement(rater_labels):
    """rater_labels: {rater_name: [labels...]} — same order/length for every rater."""
    names = list(rater_labels.keys())
    kappas = {
        f"{a}_vs_{b}": cohen_kappa_score(rater_labels[a], rater_labels[b])
        for a, b in itertools.combinations(names, 2)
    }
    avg_kappa = float(np.mean(list(kappas.values()))) if kappas else float("nan")
    print(f"Inter-rater agreement (avg kappa): {avg_kappa:.3f}")
    return {"pairwise": kappas, "avg_kappa": avg_kappa}


def majority_vote_labels(rater_labels):
    """Per-row majority vote across raters; ties broken by first-seen order."""
    names = list(rater_labels.keys())
    n = len(rater_labels[names[0]])
    votes = []
    for i in range(n):
        row = [rater_labels[name][i] for name in names]
        votes.append(Counter(row).most_common(1)[0][0])
    return votes


def compute_model_vs_human_kappa(majority_labels, model_preds):
    return cohen_kappa_score(majority_labels, model_preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="FinSentinel Phase 1 evaluation framework")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-eval-set", help="Freeze the 200-row stratified eval sample")

    p_run = sub.add_parser("run-baselines", help="Run majority/TF-IDF/(FinBERT/GPT) on the frozen eval set")
    p_run.add_argument("--with-finbert", action="store_true")
    p_run.add_argument("--with-gpt", action="store_true")

    p_curve = sub.add_parser("degradation-curve", help="FinBERT+LoRA macro F1 across agreement splits")
    p_curve.add_argument("--adapter-dir", default="finbert_lora_adapter")
    p_curve.add_argument("--out-html", default="degradation_curve.html")

    args = parser.parse_args()

    if args.command == "build-eval-set":
        eval_df = build_frozen_eval_set()
        print(f"Wrote {len(eval_df)} rows to {FROZEN_EVAL_PATH}")

    elif args.command == "run-baselines":
        table = run_full_model_comparison(with_finbert_zero_shot=args.with_finbert,
                                           with_gpt=args.with_gpt)
        print(table.to_string(index=False))
        table.to_csv("eval_comparison_table.csv", index=False)

    elif args.command == "degradation-curve":
        def predict_fn(df):
            return predict_finbert_lora(df, adapter_dir=args.adapter_dir)
        curve_df = run_agreement_degradation_curve(predict_fn)
        plot_degradation_curve(curve_df, out_path=args.out_html)
        curve_df.to_csv("degradation_curve.csv", index=False)
        print(f"Wrote degradation_curve.csv and {args.out_html}")


if __name__ == "__main__":
    _cli()
