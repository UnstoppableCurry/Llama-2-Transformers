#!/usr/bin/env python3
"""Check docs/ for internal links, landmarks, sitemap, and robots."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE_PREFIX = "/Llama-2-Transformers"
SITE_ORIGIN = "https://unstoppablecurry.github.io"
CONTENT_PAGES = ("index.html", "architecture.html", "usage.html", "limitations.html")
LANDMARK_TAGS = {
    "header": "banner",
    "nav": "navigation",
    "main": "main",
    "footer": "contentinfo",
}
LANDMARK_ROLES = set(LANDMARK_TAGS.values())


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lang = ""
        self.title = ""
        self._in_title = False
        self.landmarks: set[str] = set()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: (value or "") for key, value in attrs}
        if tag == "html":
            self.lang = data.get("lang", "")
        if tag == "title":
            self._in_title = True
        element_id = data.get("id")
        if element_id:
            self.ids.add(element_id)
        role = data.get("role")
        if tag in LANDMARK_TAGS:
            self.landmarks.add(LANDMARK_TAGS[tag])
        if role in LANDMARK_ROLES:
            self.landmarks.add(role)
        if tag == "a" or tag == "link":
            href = data.get("href")
            if href:
                self.hrefs.append(href)
        if tag in {"img", "script", "source"}:
            src = data.get("src")
            if src:
                self.srcs.append(src)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def resolve_internal(from_file: Path, ref: str) -> Path | None:
    if ref.startswith("#"):
        return from_file
    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https", "mailto"}:
        if parsed.netloc and parsed.netloc != "unstoppablecurry.github.io":
            return None
        if parsed.netloc == "unstoppablecurry.github.io":
            path = parsed.path
        else:
            return None
    else:
        path = parsed.path
    if not path:
        return from_file
    if path.startswith(SITE_PREFIX + "/"):
        rel = path[len(SITE_PREFIX) + 1 :]
    elif path == SITE_PREFIX or path == SITE_PREFIX + "/":
        rel = "index.html"
    elif path.startswith("/"):
        raise ValueError(f"{from_file.name}: root-absolute path is not under {SITE_PREFIX}: {ref}")
    else:
        rel = str((from_file.parent / path).resolve().relative_to(DOCS.resolve()))
    if rel.endswith("/") or rel == "":
        rel = (rel + "index.html") if rel else "index.html"
    return DOCS / rel


def check_pages() -> list[str]:
    errors: list[str] = []
    html_files = sorted(DOCS.rglob("*.html"))
    if not html_files:
        return ["docs/ contains no HTML files"]

    for path in html_files:
        raw = path.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(raw)
        rel = path.relative_to(DOCS)

        if not parser.lang.startswith("zh"):
            errors.append(f"{rel}: html lang should be Chinese, got {parser.lang!r}")
        if not parser.title.strip():
            errors.append(f"{rel}: missing <title>")

        missing = LANDMARK_ROLES - parser.landmarks
        if missing:
            errors.append(f"{rel}: missing landmarks {sorted(missing)}")

        if "main" in parser.ids and "skip-link" not in raw:
            errors.append(f"{rel}: has #main but no skip link")

        for ref in parser.hrefs + parser.srcs:
            parsed = urlparse(ref)
            if parsed.scheme in {"mailto"}:
                continue
            if parsed.scheme in {"http", "https"} and parsed.netloc != "unstoppablecurry.github.io":
                continue
            try:
                target = resolve_internal(path, ref)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if target is None:
                continue
            if parsed.fragment and target.exists():
                target_parser = PageParser()
                target_parser.feed(target.read_text(encoding="utf-8"))
                if parsed.fragment not in target_parser.ids:
                    errors.append(f"{rel}: fragment {ref} not found in {target.relative_to(DOCS)}")
            if not target.exists():
                errors.append(f"{rel}: broken internal link {ref} -> {target}")

    return errors


def check_robots() -> list[str]:
    path = DOCS / "robots.txt"
    if not path.exists():
        return ["docs/robots.txt is missing"]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if "User-agent:" not in text:
        errors.append("robots.txt missing User-agent")
    expected = f"Sitemap: {SITE_ORIGIN}{SITE_PREFIX}/sitemap.xml"
    if expected not in text:
        errors.append(f"robots.txt should contain {expected!r}")
    return errors


def check_sitemap() -> list[str]:
    path = DOCS / "sitemap.xml"
    if not path.exists():
        return ["docs/sitemap.xml is missing"]
    errors: list[str] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [f"sitemap.xml is not valid XML: {exc}"]
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [el.text or "" for el in tree.findall(".//sm:loc", ns)]
    if not locs:
        locs = [el.text or "" for el in tree.findall(".//loc")]
    expected = {
        f"{SITE_ORIGIN}{SITE_PREFIX}/",
        f"{SITE_ORIGIN}{SITE_PREFIX}/architecture.html",
        f"{SITE_ORIGIN}{SITE_PREFIX}/usage.html",
        f"{SITE_ORIGIN}{SITE_PREFIX}/limitations.html",
    }
    extra = set(locs) - expected
    missing = expected - set(locs)
    if missing:
        errors.append(f"sitemap.xml missing {sorted(missing)}")
    if extra:
        errors.append(f"sitemap.xml has unexpected loc {sorted(extra)}")
    if any("404" in loc for loc in locs):
        errors.append("sitemap.xml should not list 404.html")
    return errors


def main() -> int:
    if not DOCS.is_dir():
        print("docs/ directory is missing", file=sys.stderr)
        return 1
    errors = check_pages() + check_robots() + check_sitemap()
    if errors:
        print("docs check failed:")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("docs check passed: internal links, landmarks, sitemap, robots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
