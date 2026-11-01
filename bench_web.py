"""
Web answers through Javid's real pipeline, timed as the page hears them: the before/after check for changes to web
search (first used 2026-10-25 for the search router, search.py). Each question runs in its own process from a scratch
folder (no real chats, tasks or memory), with the Groq and Tavily keys only and gpt-oss-20b alone. Its real tokens are
counted in llm_usage.jsonl and its real credits in data/providers.db, where real use is counted.

    python bench_web.py                                    # this code, all six questions
    git archive b7c156b | tar -x -C /tmp/old && python bench_web.py --code /tmp/old --label before
    python bench_web.py --report                           # medians per label, from bench_results.jsonl

Per question: first audio (the "Searching..." phrase, cached on the server, so it plays as its event arrives), first
answer audio (the reply text, plus edge-tts for its first sentence), Tavily credits (from the ledger), tokens and model
requests (the turn's usage), the tools run. It costs ~6k Groq tokens (gpt-oss-20b) and 1-3 Tavily credits a question,
real ones: Javid's own cap is 40 Tavily credits a day, and the model's answers vary run to run (2-5 requests), so
compare medians of several questions, never one.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = ["Look up how tall Mount Everest is.", "Look up the population of Bangladesh.",
             "Look up the latest news about the Dhaka metro rail extension.",
             "Search for companies like Pathao in Bangladesh.", "Look up how to renew a Bangladeshi passport online.",
             "Look up when the Padma Bridge opened."]
KEEP_KEYS = ("GROQ_API_KEY", "TAVILY_API_KEY")       # one model provider and one search engine: what Javid has today
BLANK_KEYS = ("GEMINI_API_KEY", "IFM_API_KEY", "DEEPSEEK_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_IDS",
              "EXA_API_KEY", "BRAVE_SEARCH_API_KEY", "FIRECRAWL_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY",
              "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AI_TOKEN", "ELEVENLABS_API_KEY")


def one(i: int, label: str, code: str) -> dict:
    """One question, in this process (the scratch folder is the working directory)."""
    sys.path.insert(0, code)              # the code being measured, ahead of this script's own folder
    if os.path.abspath(code) != REPO and REPO in sys.path:
        sys.path.remove(REPO)
    from dotenv import dotenv_values
    real = dotenv_values(os.path.join(REPO, ".env"))
    for k in KEEP_KEYS:
        os.environ[k] = real.get(k) or ""
    for k in BLANK_KEYS:
        os.environ[k] = ""
    with open("llm_chain.json", "w") as f:
        json.dump({"chain": ["openai/gpt-oss-20b"]}, f)
    import providers
    import javid
    import web_tools
    providers.DB_FILE = os.path.join(REPO, "data", "providers.db")    # real credits are counted where real use is
    providers.IMPORT_FILES = {}
    javid.LLM_USAGE_FILE = os.path.join(REPO, "llm_usage.jsonl")      # and real tokens
    javid.synthesize(web_tools.SEARCHING)
    javid.new_session()
    t0 = time.time()
    first_say = reply_at = None
    reply, tools = "", []
    for kind, value in javid.think_stream(QUESTIONS[i]):
        now = time.time() - t0
        if kind == "say" and first_say is None:
            first_say = now
        elif kind == "tool":
            tools.append(value)
        elif kind == "delta":
            reply += value
            reply_at = now
    end = time.time()
    t1 = time.time()
    try:
        javid.synthesize((web_tools.sentences(reply) or [reply])[0])
    except Exception:
        pass
    tts = time.time() - t1
    rows = [c for c in providers.calls("tavily", t0, exclude=()) if c["ts"] <= end + 1]
    usage = javid.last_turn_usage or {}
    return {"label": label, "i": i, "q": QUESTIONS[i], "first_audio": round(first_say if first_say is not None
                                                                            else reply_at + tts, 2),
            "first_answer_audio": round(reply_at + tts, 2), "reply_out": round(reply_at, 2), "tts": round(tts, 2),
            "credits": sum(c["units"] for c in rows), "tokens": usage.get("total"), "requests": usage.get("requests"),
            "tools": tools, "engines": sorted({e for s in web_tools.sources() for e in s.get("engines", [])}),
            "reply": reply[:300]}


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for label in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == label]
        med = lambda k, rs=rs: statistics.median(r[k] for r in rs)
        print(f"{label}: n={len(rs)} · first audio median {med('first_audio'):.2f} s · first answer audio median "
              f"{med('first_answer_audio'):.1f} s (range {min(r['first_answer_audio'] for r in rs):.1f}-"
              f"{max(r['first_answer_audio'] for r in rs):.1f}) · credits an answer {statistics.mean(r['credits'] for r in rs):.2f}"
              f" · tokens an answer median {med('tokens'):,.0f} · model requests median {med('requests')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=REPO, help="the Javid code to measure (default: this one)")
    ap.add_argument("--label", default="after")
    ap.add_argument("--questions", default="1-6", help="e.g. 1-6 or 2,4")
    ap.add_argument("--out", default="bench_results.jsonl")
    ap.add_argument("--report", action="store_true", help="medians per label from --out, no requests")
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one is not None:
        print(json.dumps(one(args.one, args.label, os.path.abspath(args.code))))
        return 0
    if args.report:
        report(args.out)
        return 0
    picks = sorted({i for part in args.questions.split(",") for i in
                    (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])})
    for n in picks:
        folder = tempfile.mkdtemp(prefix="javid-bench-")
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", str(n - 1), "--label", args.label,
                               "--code", os.path.abspath(args.code)], cwd=folder, capture_output=True, text=True)
        line = (done.stdout.strip().splitlines() or [""])[-1]
        try:
            row = json.loads(line)
        except ValueError:
            print(f"Q{n} failed: {(done.stderr or line).strip()[-300:]}")
            continue
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Q{n} {row['first_audio']:.2f} s / {row['first_answer_audio']:.1f} s · {row['credits']} credits · "
              f"{row['tokens']} tokens · {row['tools']} · {row['reply'][:80]}")
    report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
