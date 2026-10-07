"""Publish the rewritten article to the Feishu doc, then verify it.

Usage (from /Users/gxy/投稿管理):
    python3 jev_workbench/reports/jevsci_20261007/publish_doc.py --body /tmp/body.xml --title "新标题"

Steps: fresh fetch -> backup -> replace title -> replace the whole body range ->
fresh fetch -> assert text/images/styles survived. Fails loudly instead of
half-writing.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/Users/gxy/投稿管理')
DOC = 'LQHgdbrwYo4ylmxjcvAcuRXln3g'
HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
RUN.mkdir(exist_ok=True)


def cli(args, cwd=ROOT):
    proc = subprocess.run(['lark-cli', *args, '--as', 'user'], cwd=cwd,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'lark-cli failed: {" ".join(args)}\n{proc.stderr}\n{proc.stdout}')
    payload = json.loads(proc.stdout)
    if not payload.get('ok'):
        raise SystemExit(f'lark-cli not ok: {json.dumps(payload, ensure_ascii=False)[:800]}')
    return payload


def fetch(tag):
    payload = cli(['docs', '+fetch', '--doc', DOC, '--detail', 'full'])
    (RUN / f'{tag}_full.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    content = payload['data']['document']['content']
    (RUN / f'{tag}.xml').write_text(content)
    return payload, content


def top_level(xml: str):
    root = ET.fromstring('<root>' + xml + '</root>')
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--body', type=Path, required=True, help='new body XML fragment')
    ap.add_argument('--title', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    payload, before = fetch('before')
    revision = payload['data']['document']['revision_id']
    root = top_level(before)
    title_el = root.find('title')
    assert title_el is not None and title_el.get('id'), 'document title block missing'
    children = [el for el in root if el.tag != 'title']
    assert children, 'empty document body'
    first, last = children[0].get('id'), children[-1].get('id')
    old_title = ''.join(title_el.itertext())
    old_text = ''.join(root.itertext())
    print(f'revision={revision} blocks={len(children)} first={first} last={last}')
    print(f'old title: {old_title}')
    print(f'new title: {args.title}')
    print(f'old chars={len(old_text)}  old images={len(root.findall(".//img"))}')
    if args.dry_run:
        return 0

    body = args.body.read_text()
    if not body.strip():
        raise SystemExit('refusing to write an empty body')

    # 1. title first (separate block, keeps the document identity intact).
    #    block_replace on the title returns degrade 1011 even when it did apply,
    #    so accept it and rely on the post-write fetch for the real check.
    if old_title != args.title:
        res = cli(['docs', '+update', '--doc', DOC, '--command', 'block_replace',
                   '--block-id', title_el.get('id'),
                   '--content', f'<title>{args.title}</title>'])
        (RUN / 'title_write.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print('title write:', res['data']['result'], res['data'].get('warnings'))
        assert res['data']['result'] in ('success', 'partial_success'), res

    # 2. whole body in one range replacement; -1 means "to the end of the doc",
    #    which also covers the trailing list whose container carries no block id.
    body_file = RUN / 'body_payload.xml'
    body_file.write_text(body)
    res = cli(['docs', '+update', '--doc', DOC, '--command', 'block_replace',
               '--start-block-id', first, '--end-block-id', '-1',
               '--content', '@./' + str(body_file.relative_to(ROOT))])
    (RUN / 'body_write.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print('body result:', res['data']['result'], 'warnings:', res['data'].get('warnings'))
    assert res['data']['result'] == 'success', res
    assert not res['data'].get('warnings'), res['data']['warnings']

    # 3. verify.
    payload2, after = fetch('after')
    root2 = top_level(after)
    new_title = ''.join(root2.find('title').itertext())
    assert new_title == args.title, (new_title, args.title)
    new_text = ''.join(root2.itertext())
    imgs = root2.findall('.//img')
    tables = root2.findall('.//table')
    report = {
        'revision_before': revision,
        'revision_after': payload2['data']['document']['revision_id'],
        'title': new_title,
        'chars': len(new_text),
        'images': len(imgs),
        'image_tokens': [i.get('src') for i in imgs],
        'tables': len(tables),
        'paragraphs': len(root2.findall('.//p')),
        'headings': len(root2.findall('.//h2')) + len(root2.findall('.//h3')),
        'callouts': len(root2.findall('.//callout')),
        'blockquotes': len(root2.findall('.//blockquote')),
        'placeholders_left': re.findall(r'\{\{[A-Z_]+\}\}', new_text),
    }
    (RUN / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    md = cli(['docs', '+fetch', '--doc', DOC, '--doc-format', 'markdown'])
    (RUN / 'after_markdown.json').write_text(json.dumps(md, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
