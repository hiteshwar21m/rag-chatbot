import re, html, chardet, pathlib, os, sys
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

# --------------- compiled regex -----------------
RE_JS = re.compile(r'javascript:[^)]*\)', re.I)
RE_EMAIL = re.compile(
    r'(\[?mailto:)?([a-zA-Z0-9_.+-]+)\[?[\(\[]*(?:at|@)[\)\]]*([a-zA-Z0-9-]+)\[?[\(\[]*(?:dot|\.)[\)\]]*([a-zA-Z0-9-.]+)',
    re.I)
RE_PHONE = re.compile(r'(\+?91[\-\s]?)?\d{2,5}[\-\s]?\(?\d{3,5}\)?[\-\s]?\d{4,5}')
RE_MULTI_NL = re.compile(r'\n{3,}')
# ------------------------------------------------

def clean_manit_v2(raw_html: str) -> str:
    """Return sanitised markdown-ready text."""
    # 1. slice body
    start = raw_html.find("You are here")
    end = raw_html.find("![Notice]", start)
    text = raw_html[start:end] if start != -1 and end != -1 else raw_html

    # 2. quick strips
    text = RE_JS.sub('', text)
    text = html.unescape(text)

    # 3. email / phone
    text = RE_EMAIL.sub(lambda m: f"{m[2]}@{m[3]}.{m[4]}", text)
    text = RE_PHONE.sub(lambda m: re.sub(r'\D', '', m[0]), text)

    # 4. remove markdown artefacts but KEEP table separators
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'#{2,}', '', text)

    # 5. de-duplicate lines (normalised key)
    seen, out = set(), []
    for line in text.splitlines():
        key = re.sub(r'\W', '', line.strip().lower())
        if key and len(line) > 20 and key not in seen:
            seen.add(key)
            out.append(line.rstrip())
    text = '\n'.join(out)

    return RE_MULTI_NL.sub('\n\n', text).strip()

def process_all(indir=None, outdir=None):
    # Get config and setup paths
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    
    if indir is None:
        indir = project_root / "data" / "raw" / "webpages"
    if outdir is None:
        outdir = project_root / "data" / "extracted" / "webpage_text"
    
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    for file in pathlib.Path(indir).glob("*.txt"):
        raw = file.read_bytes()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        cleaned = clean_manit_v2(raw.decode(enc, errors='replace'))
        pathlib.Path(outdir, f"clean_{file.name}").write_text(cleaned, encoding='utf-8')
        print("✅", file.name)

if __name__ == "__main__":
    process_all()