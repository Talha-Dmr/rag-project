import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.bankofengland.co.uk/_api/sitemap/getsitemap"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "finreg" / "uk_boe"
RAW_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR = RAW_DIR / "_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = RAW_DIR / "fetch_manifest.json"

HEADERS = {
    "User-Agent": "AcademicRAGBot/1.0 (research use)"
}


def fetch(url: str, timeout: int = 30):
    req = Request(url, headers=HEADERS)
    return urlopen(req, timeout=timeout)


def fetch_xml(url: str) -> bytes:
    with fetch(url) as response:
        return response.read()


def fetch_text(url: str) -> str:
    with fetch(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def download_file(url: str, out_path: Path):
    with fetch(url, timeout=60) as response:
        out_path.write_bytes(response.read())


def parse_xml_locs(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]


def collect_urls_from_sitemap(sitemap_url: str):
    locs = parse_xml_locs(fetch_xml(sitemap_url))

    nested_sitemaps = [u for u in locs if "sitemap" in u.lower() and u.lower().endswith(".xml")]
    normal_urls = [u for u in locs if u not in nested_sitemaps]

    for nested in nested_sitemaps:
        try:
            normal_urls.extend(parse_xml_locs(fetch_xml(nested)))
            time.sleep(0.5)
        except Exception as e:
            print(f"[WARN] nested sitemap failed: {nested} -> {e}")

    return list(dict.fromkeys(normal_urls))


def looks_relevant(url: str) -> bool:
    u = url.lower()

    if "prudential-regulation" not in u:
        return False

    good = any(k in u for k in [
        "policy-statement",
        "supervisory-statement",
        "consultation-paper",
        "rulebook"
    ])

    bad = any(k in u for k in [
        "speech",
        "working-paper",
        "research",
        "statistics",
        "event",
        "blog"
    ])

    return good and not bad



def _is_pdf_href(href: str) -> bool:
    if not href:
        return False
    href = str(href).strip()
    if not href:
        return False
    return urlparse(href).path.lower().endswith('.pdf')


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s/]+", "_", value)
    return value


def extract_pdf_link(html: str, page_url: str):
    soup = BeautifulSoup(html, "html.parser")

    priority_links = []
    fallback_links = []

    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        link_text = a.get_text(" ", strip=True).lower()

        if not _is_pdf_href(href):
            continue

        full_url = urljoin(page_url, href)

        if any(k in link_text for k in ["pdf", "download", "full paper", "full report"]):
            priority_links.append(full_url)
        else:
            fallback_links.append(full_url)

    if priority_links:
        return priority_links[0]
    if fallback_links:
        return fallback_links[0]
    return None


def safe_slug(page_url: str) -> str:
    path = urlparse(page_url).path.rstrip("/")
    tail = path.split("/")[-1] if path else "document"
    if tail.lower().endswith(".pdf"):
        tail = tail[:-4]
    return slugify(tail or "document")


def save_review_html(slug: str, html: str) -> Path:
    out_path = REVIEW_DIR / f"{slug}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    urls = collect_urls_from_sitemap(SITEMAP_URL)
    candidates = [u for u in urls if looks_relevant(u)]
    manifest = {
        "downloaded": [],
        "review": [],
        "errors": [],
    }

    print(f"[INFO] total sitemap urls: {len(urls)}")
    print(f"[INFO] relevant candidates: {len(candidates)}")

    for i, page_url in enumerate(candidates, 1):
        try:
            slug = safe_slug(page_url)
            out_path = RAW_DIR / f"{slug}.pdf"

            if _is_pdf_href(page_url):
                download_file(page_url, out_path)
                manifest["downloaded"].append({"page_url": page_url, "pdf_url": page_url, "file": out_path.name})
                print(f"[{i}/{len(candidates)}] saved direct pdf: {out_path.name}")
            else:
                html = fetch_text(page_url)
                pdf_url = extract_pdf_link(html, page_url)

                if pdf_url:
                    download_file(pdf_url, out_path)
                    manifest["downloaded"].append(
                        {"page_url": page_url, "pdf_url": pdf_url, "file": out_path.name}
                    )
                    print(f"[{i}/{len(candidates)}] saved pdf: {out_path.name}")
                else:
                    review_path = save_review_html(slug, html)
                    manifest["review"].append(
                        {
                            "page_url": page_url,
                            "reason": "no_pdf_link_found",
                            "review_file": review_path.name,
                        }
                    )
                    print(f"[{i}/{len(candidates)}] saved review html: {review_path.name}")

            time.sleep(1.5)

        except Exception as e:
            manifest["errors"].append({"page_url": page_url, "error": str(e)})
            print(f"[ERROR] {page_url} -> {e}")

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] manifest written: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
