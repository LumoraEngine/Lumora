#!/usr/bin/env python3
"""Build the website: copy site/ and render the guides in docs/ into pages.

  python3 scripts/site/build.py --out _site

docs/GUIDE.md (the user guide) and docs/SCRIPTING.md (scripting) are the single sources of
their pages; they read well on GitHub and become docs.html and scripting.html here. Needs the
`markdown` package (pip install markdown).
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/v3nn7/Lumora/blob/main/"


@dataclass(frozen=True)
class Guide:
    source: str       # under docs/
    page: str         # written to the site root
    title: str
    description: str


GUIDES = [
    Guide("GUIDE.md", "docs.html", "Lumora Guide",
          "How to install Lumora, build scenes in the editor, light and render them, use physics, "
          "animation and play mode, and edit together over LAN or playit.gg."),
    Guide("SCRIPTING.md", "scripting.html", "Lumora Scripting",
          "Write behaviours for Lumora games in visual scripts, Lua or C++: events, the engine API, "
          "network games and animator controllers."),
]

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{description}">
  <meta name="theme-color" content="#0d0f14" media="(prefers-color-scheme: dark)">
  <meta name="theme-color" content="#f6f7fb" media="(prefers-color-scheme: light)">
  <link rel="icon" href="assets/logo.svg" type="image/svg+xml">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="site-header">
  <div class="wrap">
    <a class="brand" href="./"><img src="assets/logo.svg" alt="">LUMORA</a>
    <nav class="nav" aria-label="Site">
      <a href="./#features">Features</a>
      <a href="./#download">Download</a>
{nav}
      <a href="https://github.com/v3nn7/Lumora">GitHub</a>
    </nav>
  </div>
</header>
<div class="wrap docs">
  <nav class="toc" aria-label="Contents">
{toc}
  </nav>
  <article class="doc" id="content">
{body}
    <p class="note">This page is generated from <a href="{repo}docs/{source}">docs/{source}</a>.
      Found a mistake? Edit it on GitHub.</p>
  </article>
</div>
<footer class="site-footer">
  <div class="wrap">
    <span>Lumora Engine · free software under the <a href="{repo}LICENSE">GNU GPL v3 or later</a></span>
    <span><a href="./">Home</a> · <a href="https://github.com/v3nn7/Lumora">Source</a> ·
      <a href="{repo}CONTRIBUTING.md">Contributing</a></span>
  </div>
</footer>
</body>
</html>
"""

NAV_LABELS = {"docs.html": "Guide", "scripting.html": "Scripting"}


def render_nav(current: Guide) -> str:
    lines = []
    for guide in GUIDES:
        mark = ' aria-current="page"' if guide is current else ""
        lines.append(f'      <a href="{guide.page}"{mark}>{NAV_LABELS[guide.page]}</a>')
    return "\n".join(lines)


def render_toc(tokens: list[dict]) -> str:
    """Sidebar from the guide's headings: each h2 is a group, its h3s are the links."""
    lines: list[str] = []
    for top in tokens:  # the single h1
        for section in top.get("children", []):
            lines.append(f'    <p><a href="#{section["id"]}">{section["name"]}</a></p>')
            for sub in section.get("children", []):
                lines.append(f'    <a href="#{sub["id"]}">{sub["name"]}</a>')
    return "\n".join(lines)


def rewrite_links(body: str) -> str:
    """Relative links in the guides point at repository files. A guide that is a page of the
    website links to that page (keeping the #section); anything else goes to GitHub."""
    pages = {guide.source: guide.page for guide in GUIDES}

    def fix(match: re.Match[str]) -> str:
        target = html.unescape(match.group(1))
        if re.match(r"^(https?:|mailto:|#)", target):
            return match.group(0)
        path, _, anchor = target.partition("#")
        if path in pages:
            return f'href="{pages[path]}{"#" + anchor if anchor else ""}"'
        return f'href="{REPO}docs/{target}"'

    return re.sub(r'href="([^"]+)"', fix, body)


def render(guide: Guide) -> str:
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "attr_list", "toc"],
        extension_configs={"toc": {"toc_depth": "1-3"}},
    )
    body = md.convert((ROOT / "docs" / guide.source).read_text(encoding="utf-8"))
    body = rewrite_links(body)
    # The guide's first paragraph is its introduction.
    body = body.replace("<p>", '<p class="lead">', 1)
    return PAGE.format(title=guide.title, description=html.escape(guide.description), nav=render_nav(guide),
                       toc=render_toc(md.toc_tokens), body=body, repo=REPO, source=guide.source)


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    for guide in GUIDES:
        (out / guide.page).write_text(render(guide), encoding="utf-8")
    print(f"site written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    build(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
