"""Convert the article source (.md with colour markers) into Feishu docx XML.

Marker syntax used in the source file
-------------------------------------
[[P]]text[[/]]      purple bold, section-heading colour
[[B]]text[[/]]      blue bold (task definition / method / key quantities)
[[G]]text[[/]]      green bold (gold answers, confirmed-correct predictions)
[[R]]text[[/]]      red bold   (misclassifications, confirmed method errors)
[[O]]text[[/]]      orange bold (refusals, invalid output, limits, open items)
[[Y]]text[[/]]      grey (captions, record ids, sources)
[[H-B]]text[[/]]    light-blue background highlight, blue text
[[H-P]]text[[/]]    light-purple background highlight, purple text

Blocks
------
# Title                 -> emitted as <title> (skipped in body)
## / ### headings       -> styled h2 / h3
- item                  -> <ul><li>
1. item                 -> <ol><li>
> quote                 -> <blockquote><p>…</p></blockquote>
| a | b |               -> table (with light-purple header)
<callout emoji="x">…</callout>
![](IMG:name)           -> <img path="@relative/src"/>
plain paragraph         -> <p text-indent="1">
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path('/Users/gxy/投稿管理')

COLORS = {
    'P': ('rgb(100,37,208)', None),
    'B': ('rgb(36,91,219)', None),
    'G': ('rgb(46,161,33)', None),
    'R': ('rgb(216,57,49)', None),
    'O': ('rgb(222,120,2)', None),
    'Y': ('rgb(143,149,158)', None),
    'H-B': ('rgb(36,91,219)', 'rgba(186,206,253,0.7)'),
    'H-P': ('rgb(100,37,208)', 'rgba(205,178,250,0.7)'),
}
# Lower-case markers render without bold; fades used for captions and notes.
NO_BOLD = {'p', 'b', 'g', 'r', 'o', 'y'}
TOKEN = re.compile(r'\[\[(P|B|G|R|O|Y|H-B|H-P|p|b|g|r|o|y)\]\](.*?)\[\[/\]\]', re.S)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def inline(text: str) -> str:
    """Colour markers + **bold** + `code` + [label](url) -> Feishu inline XML."""
    out = []
    pos = 0
    for m in TOKEN.finditer(text):
        out.append(_plain(text[pos:m.start()]))
        key, body = m.group(1), m.group(2)
        colour, background = COLORS[key.upper()]
        attrs = f'text-color="{colour}"'
        if background:
            attrs = f'background-color="{background}" ' + attrs
        span = f'<span {attrs}>{inline(body)}</span>'
        out.append(span if key in NO_BOLD else f'<b>{span}</b>')
        pos = m.end()
    out.append(_plain(text[pos:]))
    return ''.join(out)


PLAIN = re.compile(r'(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))', re.S)


def _plain(text: str) -> str:
    parts = []
    for chunk in PLAIN.split(text):
        if not chunk:
            continue
        if chunk.startswith('**') and chunk.endswith('**'):
            parts.append(f'<b>{esc(chunk[2:-2])}</b>')
        elif chunk.startswith('`') and chunk.endswith('`'):
            parts.append(f'<code>{esc(chunk[1:-1])}</code>')
        elif chunk.startswith('[') and '](' in chunk:
            label, url = chunk[1:-1].split('](', 1)
            parts.append(f'<a href="{html.escape(url, quote=True)}">{esc(label)}</a>')
        else:
            parts.append(esc(chunk))
    return ''.join(parts)


def paragraph(text: str, indent: bool = True) -> str:
    attr = ' text-indent="1"' if indent else ''
    return f'<p{attr}>{inline(text)}</p>'


def table_block(rows, header=True) -> str:
    def cell(tag, text):
        style = ' background-color="rgba(205,178,250,0.55)"' if tag == 'th' else ''
        return f'<{tag}{style}><p>{inline(text)}</p></{tag}>'
    thead = ''
    body = rows
    if header:
        thead = '<thead><tr>' + ''.join(cell('th', c) for c in rows[0]) + '</tr></thead>'
        body = rows[1:]
    tbody = '<tbody>' + ''.join(
        '<tr>' + ''.join(cell('td', c) for c in r) + '</tr>' for r in body) + '</tbody>'
    return f'<table>{thead}{tbody}</table>'


def convert(source: str, img_root: Path, cwd_rel: str) -> tuple[str, list[str]]:
    lines = source.splitlines()
    out: list[str] = []
    images: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('# ') and not stripped.startswith('##'):
            i += 1
            continue
        if stripped.startswith('### '):
            out.append(f'<h3><b><span background-color="rgba(186,206,253,0.7)" '
                       f'text-color="rgb(36,91,219)">{inline(stripped[4:])}</span></b></h3>')
            i += 1
            continue
        if stripped.startswith('## '):
            out.append(f'<h2><b><span background-color="rgba(205,178,250,0.7)" '
                       f'text-color="rgb(100,37,208)">{inline(stripped[3:])}</span></b></h2>')
            i += 1
            continue
        if stripped.startswith('<callout'):
            head = stripped
            inner: list[str] = []
            # callout body may be inline on the same line or following lines until </callout>
            while '</callout>' not in head:
                i += 1
                head += '\n' + lines[i]
            body = head.split('>', 1)[1].rsplit('</callout>', 1)[0].strip()
            inner = [paragraph(p.strip()) for p in body.split('\n\n') if p.strip()]
            attrs = 'emoji="💡" background-color="rgb(240,244,255)" border-color="rgb(130,167,252)"'
            m = re.search(r'emoji="([^"]+)"', head)
            if m:
                attrs = attrs.replace('emoji="💡"', f'emoji="{m.group(1)}"')
            out.append(f'<callout {attrs}>' + ''.join(inner) + '</callout>')
            i += 1
            continue
        if stripped.startswith('![](') and stripped.endswith(')'):
            target = stripped[4:-1]
            name = target.split('IMG:')[-1].strip()
            rel = (img_root.resolve() / f'{name}.png').resolve().relative_to(ROOT.resolve())
            out.append(f'<img path="@./{rel}" name="{name}.png"/>')
            images.append(name)
            i += 1
            continue
        if stripped.startswith('```'):
            lang = stripped[3:].strip() or 'text'
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            i += 1
            body = esc('\n'.join(code).strip())
            out.append(f'<pre lang="{lang}"><code>{body}</code></pre>')
            continue
        if stripped.startswith('>'):
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            body = ''.join(paragraph(q, indent=False) for q in quote if q)
            out.append(f'<blockquote>{body}</blockquote>')
            continue
        if stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'-{0,}', c or '-') for c in cells) or any(c == '' for c in cells):
                    rows.append(cells)
                i += 1
            out.append(table_block(rows))
            continue
        if stripped.startswith('- '):
            items = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                items.append(paragraph(lines[i].strip()[2:], indent=False))
                i += 1
            out.append('<ul>' + ''.join(f'<li>{p}</li>' for p in items) + '</ul>')
            continue
        if re.match(r'^\d+\. ', stripped):
            items = []
            while i < len(lines) and re.match(r'^\d+\. ', lines[i].strip()):
                items.append(paragraph(re.sub(r'^\d+\. ', '', lines[i].strip()), indent=False))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{p}</li>' for p in items) + '</ol>')
            continue
        if stripped.startswith('图注：') or stripped.startswith('图：'):
            out.append(paragraph(f'[[Y]]{stripped}[[/]]'))
            i += 1
            continue
        out.append(paragraph(stripped))
        i += 1
    return ''.join(out), images


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    body, images = convert(args.source.read_text(), args.images, '')
    args.out.write_text(body)
    print(f'wrote {args.out} ({len(body)} chars, {len(images)} images)')


if __name__ == '__main__':
    main()
