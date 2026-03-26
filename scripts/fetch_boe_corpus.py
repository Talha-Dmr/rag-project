import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

BASE = "https://www.bankofengland.co.uk"
SITEMAP_URL = "https://www.bankofengland.co.uk/_api/sitemap/getsitemap"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "finreg" / "uk_boe"
RAW_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "AcademicRAGBot/1.0 (research use)"
})


def fetch(url: str, timeout: int = 30) -> requests.Response:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_xml(url: str) -> bytes:
    return fetch(url).content


def fetch_text(url: str) -> str:
    return fetch(url).text


def download_file(url: str, out_path: Path):
    r = fetch(url, timeout=60)
    out_path.write_bytes(r.content)


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
    slug = page_url.rstrip("/").split("/")[-1]
    slug = slug.replace(" ", "_")
    return slug


def main():
    urls = collect_urls_from_sitemap(SITEMAP_URL)
    candidates = [u for u in urls if looks_relevant(u)]

    print(f"[INFO] total sitemap urls: {len(urls)}")
    print(f"[INFO] relevant candidates: {len(candidates)}")

    for i, page_url in enumerate(candidates, 1):
        try:
            html = fetch_text(page_url)
            pdf_url = extract_pdf_link(html, page_url)
            slug = safe_slug(page_url)

            if pdf_url:
                out_path = RAW_DIR / f"{slug}.pdf"
                download_file(pdf_url, out_path)
                print(f"[{i}/{len(candidates)}] saved pdf: {out_path.name}")
            else:
                out_path = RAW_DIR / f"{slug}.html"
                out_path.write_text(html, encoding="utf-8")
                print(f"[{i}/{len(candidates)}] saved html: {out_path.name}")

            time.sleep(1.5)

        except Exception as e:
            print(f"[ERROR] {page_url} -> {e}")


if __name__ == "__main__":
    main()