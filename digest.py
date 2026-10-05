"""
Analytics & AI Weekly Digest agent (plain-English edition).
Pipeline: fetch feeds -> LLM selects/explains (grounded in fetched items only)
-> readability check (retry once if too hard) -> verify startup name -> HTML email -> Gmail SMTP.
"""
import os
import re
import sys
import json
import html
import smtplib
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import feedparser
from bs4 import BeautifulSoup
import anthropic
import textstat

# ----------------------------------------------------------------------------
# CONFIG - edit this section only
# ----------------------------------------------------------------------------
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
LOOKBACK_DAYS = 8
MAX_ARTICLES = 60
MAX_GRADE = 10.5  # retry once if the email reads harder than this (Flesch-Kincaid grade)
UA = "Mozilla/5.0 (compatible; analytics-ai-digest-agent/1.0)"

# (display name, feed url, kind). kind: news | startup | vendor | research
# vendor = company blogs (their own marketing). URLs are unverified: broken feeds are
# skipped and listed in the email footer.
FEEDS = [
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", "news"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "news"),
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "startup"),
    ("Crunchbase News", "https://news.crunchbase.com/feed/", "startup"),
    ("KDnuggets", "https://www.kdnuggets.com/feed", "news"),
    ("Towards Data Science", "https://towardsdatascience.com/feed", "research"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "research"),
    ("Import AI", "https://importai.substack.com/feed", "research"),
    ("Databricks Blog", "https://www.databricks.com/feed", "vendor"),
    ("Snowflake Blog", "https://www.snowflake.com/en/blog/feed/", "vendor"),
]

PROFILE = """
Product marketing manager at an enterprise analytics and AI software company, focused on
decision intelligence (software that helps businesses make better automated decisions).
Strong in marketing, messaging, launches and sales enablement. NEW to analytics and AI technology:
needs concepts explained from scratch. Wants material usable for networking conversations,
blog posts and job interviews.
"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
NOW = dt.datetime.now(dt.timezone.utc)
FAILURES = []


def clean(raw):
    text = BeautifulSoup(raw or "", "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')).strip().lower()


def esc(s):
    return html.escape(str(s) if s is not None else "")


# ----------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------
def fetch_articles():
    cutoff = NOW - dt.timedelta(days=LOOKBACK_DAYS)
    items = []
    for name, url, kind in FEEDS:
        try:
            feed = feedparser.parse(url, agent=UA)
            if not feed.entries:
                raise RuntimeError("no entries returned")
            for e in feed.entries[:12]:
                ts = e.get("published_parsed") or e.get("updated_parsed")
                if ts and dt.datetime(*ts[:6], tzinfo=dt.timezone.utc) < cutoff:
                    continue
                content = e.get("content")
                body = content[0].get("value", "") if content else e.get("summary", "")
                items.append({
                    "source": name,
                    "kind": kind,
                    "title": clean(e.get("title", "")),
                    "url": e.get("link", ""),
                    "text": clean(body)[:1200],
                })
        except Exception as ex:
            FAILURES.append(f"Feed failed: {name} ({ex})")
    seen, unique = set(), []
    for it in items:
        if it["url"] and it["url"] not in seen:
            seen.add(it["url"])
            unique.append(it)
    unique = unique[:MAX_ARTICLES]
    for i, it in enumerate(unique):
        it["id"] = i
    return unique


# ----------------------------------------------------------------------------
# LLM (grounded: may only reference article ids it was given)
# ----------------------------------------------------------------------------
SYSTEM = """You write a weekly briefing for a smart adult who is new to analytics and AI.
Explain everything like you are talking to a 10th grader. You are given numbered ARTICLES fetched this week.
Rules:
- Plain English. Short sentences (aim for under 18 words). Everyday words. Define any technical term in simple words the first time it appears. Give one concrete everyday analogy for the main innovation. No hype words like 'revolutionary' or 'game-changing'.
- Reference articles ONLY by numeric id. Never invent articles, URLs, companies, people, numbers, funding amounts or dates. Include a name, number or date only if it appears in the cited article text.
- Articles marked 'vendor' are company marketing. When you rely on one, say it is the company's own claim.
- The startup must be a real company named in an article's text. If no article names a suitable startup, set startup to null.
- 'wild_idea' is clearly labeled speculation, not a forecast. It must be built from 2-4 cited articles, say what would make it wrong, and be honest about low confidence.
- 'insight' items must be things the reader can actually say or write, tied to a cited article.
Return ONLY valid JSON, no markdown fences, matching this schema:
{
 "innovation": {"id": int, "headline": str, "plain_explanation": str, "analogy": str, "why_it_matters": str,
                "learn_next": {"term": str, "why": str}, "glossary": [{"term": str, "meaning": str}]},
 "startup": {"id": int, "name": str, "what_it_does": str, "why_watch": str, "reality_check": str} | null,
 "insight": {"networking": {"id": int, "line": str}, "blogging": {"id": int, "angle": str}, "interviewing": {"id": int, "line": str}},
 "wild_idea": {"title": str, "story": str, "based_on_ids": [int], "what_would_make_it_wrong": str, "confidence": "low"|"medium"}
}
'glossary' has at most 4 entries. 'learn_next.term' is ONE concept to look up this week."""


def build_user(articles):
    block = "\n\n".join(
        f"[ARTICLE {a['id']}] ({a['kind']}) {a['source']} | {a['title']}\n{a['text']}" for a in articles
    )
    return f"READER PROFILE:\n{PROFILE}\n\nARTICLES:\n{block}"


def parse_json(raw):
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"LLM returned no JSON: {raw[:300]}")
    return json.loads(raw[start:end + 1])


def text_of(result):
    """All reader-facing prose, for the readability score."""
    parts = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and key not in ("name", "term", "confidence", "headline", "title"):
            parts.append(x if x.endswith((".", "!", "?")) else x + ".")

    walk(result)
    return " ".join(parts)


def call_llm(articles):
    client = anthropic.Anthropic()
    user = build_user(articles)
    msgs = [{"role": "user", "content": user}]
    msg = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM, messages=msgs)
    raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    result = parse_json(raw)
    grade = textstat.flesch_kincaid_grade(text_of(result))
    if grade > MAX_GRADE:
        msgs += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That reads at about grade {grade:.1f}. Rewrite the SAME JSON at a 9th-10th grade level: "
                "shorter sentences, simpler words, jargon explained. Keep ids and facts identical. Return only JSON."
            )},
        ]
        msg = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM, messages=msgs)
        raw2 = "".join(b.text for b in msg.content if getattr(b, "text", None))
        try:
            result2 = parse_json(raw2)
            grade2 = textstat.flesch_kincaid_grade(text_of(result2))
            if grade2 < grade:
                result, grade = result2, grade2
        except Exception as ex:
            FAILURES.append(f"Readability retry failed, kept first draft ({ex})")
    return result, grade


# ----------------------------------------------------------------------------
# Validate + render
# ----------------------------------------------------------------------------
def by_id(items, i):
    try:
        return next(x for x in items if x["id"] == int(i))
    except (StopIteration, TypeError, ValueError):
        return None


def link(a):
    return f'<a href="{esc(a["url"])}">{esc(a["source"])}: {esc(a["title"])}</a>'


P = 'style="font-family:Arial,sans-serif;font-size:15px;line-height:1.55;color:#222;margin:6px 0"'
SMALL = 'style="font-family:Arial,sans-serif;font-size:13px;line-height:1.5;color:#555;margin:4px 0"'


def section(title, body_html):
    return (
        f'<h2 style="font-family:Arial,sans-serif;font-size:18px;color:#111;margin:28px 0 8px;'
        f'border-bottom:2px solid #111;padding-bottom:4px">{esc(title)}</h2>{body_html}'
    )


def render(result, articles, grade):
    parts = []

    # Innovation
    inn = result.get("innovation") or {}
    a = by_id(articles, inn.get("id"))
    if a:
        gl = "".join(
            f'<li {SMALL}><b>{esc(g.get("term"))}</b>: {esc(g.get("meaning"))}</li>'
            for g in (inn.get("glossary") or [])[:4]
        )
        ln = inn.get("learn_next") or {}
        body = (
            f'<p {P}><b>{esc(inn.get("headline", a["title"]))}</b></p>'
            f'<p {P}>{esc(inn.get("plain_explanation"))}</p>'
            f'<p {P}><b>Think of it like this:</b> {esc(inn.get("analogy"))}</p>'
            f'<p {P}><b>Why it matters:</b> {esc(inn.get("why_it_matters"))}</p>'
            + (f'<p {P}><b>Look up this week:</b> {esc(ln.get("term"))}. {esc(ln.get("why"))}</p>' if ln.get("term") else "")
            + (f'<p {SMALL}><b>Words to know</b></p><ul style="padding-left:20px">{gl}</ul>' if gl else "")
            + f'<p {SMALL}>Source: {link(a)}'
            + (" (company blog: the claims are the company's own)" if a["kind"] == "vendor" else "")
            + "</p>"
        )
    else:
        body = f"<p {P}>No grounded innovation this week (model returned an invalid source).</p>"
    parts.append(section("New Innovation of the Week", body))

    # Startup (fail closed: name must appear in the cited article)
    st = result.get("startup")
    sa = by_id(articles, (st or {}).get("id")) if st else None
    if st and sa and norm(st.get("name")) and norm(st.get("name")) in norm(sa["title"] + " " + sa["text"]):
        body = (
            f'<p {P}><b>{esc(st.get("name"))}</b></p>'
            f'<p {P}>{esc(st.get("what_it_does"))}</p>'
            f'<p {P}><b>Why watch it:</b> {esc(st.get("why_watch"))}</p>'
            f'<p {P}><b>Reality check:</b> {esc(st.get("reality_check"))}</p>'
            f'<p {SMALL}>Source: {link(sa)}</p>'
        )
    else:
        body = f"<p {P}>No verifiable startup this week. Skipped rather than guessing.</p>"
    parts.append(section("Startup to Watch", body))

    # Insight
    ins = result.get("insight") or {}
    rows = []
    for label, key, field in (("Networking: ask this", "networking", "line"),
                              ("Blogging: write this", "blogging", "angle"),
                              ("Interviewing: say this", "interviewing", "line")):
        item = ins.get(key) or {}
        ia = by_id(articles, item.get("id"))
        if ia and item.get(field):
            rows.append(
                f'<p {P}><b>{label}</b><br>{esc(item.get(field))}<br><span {SMALL}>Based on: {link(ia)}</span></p>'
            )
    parts.append(section("Insight to Take Away", "".join(rows) or f"<p {P}>No grounded insights this week.</p>"))

    # Wild idea
    w = result.get("wild_idea") or {}
    based = [by_id(articles, i) for i in (w.get("based_on_ids") or [])]
    based = [b for b in based if b]
    if w.get("story") and based:
        body = (
            f'<p {P}><b>{esc(w.get("title"))}</b></p>'
            f'<p {P}>{esc(w.get("story"))}</p>'
            f'<p {P}><b>What would make this wrong:</b> {esc(w.get("what_would_make_it_wrong"))}</p>'
            f'<p {SMALL}>Speculation, not a forecast. Confidence: {esc(w.get("confidence", "low"))}. '
            f'Built from: {" | ".join(link(b) for b in based)}</p>'
        )
    else:
        body = f"<p {P}>No grounded 5-year idea this week.</p>"
    parts.append(section("Wild 5-Year Idea", body))

    footer = [f"Reading level: about grade {grade:.1f}"]
    if grade > MAX_GRADE:
        footer.append("(above target; the retry did not fix it)")
    foot_html = f'<p style="font-family:Arial,sans-serif;font-size:12px;color:#888;margin-top:30px">{esc(" ".join(footer))}</p>'
    if FAILURES:
        fails = "".join(f"<li>{esc(f)}</li>" for f in FAILURES)
        foot_html += (
            f'<p style="font-family:Arial,sans-serif;font-size:12px;color:#a00">Agent health warnings:</p>'
            f'<ul style="font-size:12px;color:#a00">{fails}</ul>'
        )

    date_str = NOW.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%b %d, %Y")
    return (
        f'<div style="max-width:640px;margin:0 auto;padding:16px">'
        f'<h1 style="font-family:Arial,sans-serif;font-size:22px;margin:0">Analytics &amp; AI Weekly Digest</h1>'
        f'<p style="font-family:Arial,sans-serif;font-size:13px;color:#666;margin:2px 0 0">{date_str} | '
        f"{len(articles)} articles scanned</p>"
        + "".join(parts) + foot_html + "</div>"
    ), date_str


# ----------------------------------------------------------------------------
# Send
# ----------------------------------------------------------------------------
def send_email(subject, html_body):
    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ.get("DIGEST_TO") or sender
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(re.sub(r"<[^>]+>", "", html_body), "plain"))
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, password)
        s.sendmail(sender, [to], msg.as_string())


def main():
    # Two cron entries cover PDT and PST; only the one landing at 7am Pacific proceeds.
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        pt_hour = dt.datetime.now(ZoneInfo("America/Los_Angeles")).hour
        if pt_hour != 7:
            print(f"Pacific hour is {pt_hour}, not 7. Skipping.")
            return

    articles = fetch_articles()
    if not articles:
        print("No articles fetched. Failing so the Action shows red.", file=sys.stderr)
        sys.exit(1)
    result, grade = call_llm(articles)
    body, date_str = render(result, articles, grade)
    send_email(f"Analytics & AI Weekly Digest - {date_str}", body)
    print("Sent.")


if __name__ == "__main__":
    main()
