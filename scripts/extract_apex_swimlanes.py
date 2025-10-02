#!/usr/bin/env python3
"""
scripts/extract_apex_swimlanes.py

Extract apex=true TODO checklist items grouped by swimlane from a markdown TODO file.
Optionally create GitHub issues for each apex item (idempotent: checks for existing issues with same title).

Usage:
  python3 scripts/extract_apex_swimlanes.py TODO.md > APEX_SWIMLANES.md
  python3 scripts/extract_apex_swimlanes.py TODO.md --create-issues --repo owner/repo --assignee me --labels apex,swimlane:parsing

Environment:
  - If `gh` CLI is available it will prefer it. Otherwise a `GITHUB_TOKEN` env var is required to use the REST API.

"""
import sys
import re
import collections
import argparse
import os
import json
import csv
import subprocess
from urllib import request, parse, error
import yaml
import pathlib
from datetime import datetime

# Optional: cryptography for local encrypted audit
try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except Exception:
    Fernet = None
    _HAS_CRYPTO = False


def parse_todo(path):
    # matches checkbox styles and optional HTML comment metadata
    line_re = re.compile(r'^[\s\-\*]*([☐☒]|- \[[ xX]\])\s*(?P<text>.*?)(?:\s*<!--\s*bmad:(?P<meta>.*?)\s*-->)?\s*$')
    meta_split_re = re.compile(r'\s*;\s*')
    kv_re = re.compile(r'\s*([^=]+)\s*=\s*(.+)\s*')

    swimlanes = collections.OrderedDict()
    # map of text -> inference metadata when auto-promote used
    inference_map = {}

    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            m = line_re.match(ln.rstrip('\n'))
            if not m:
                # also accept bracket-style prefix like [apex][swimlane=parsing] Title
                stripped = ln.strip()
                m2 = re.match(r'^[\-\*\s]*([☐☒]|- \[[ xX]\])\s*(?P<text>.*)$', stripped)
                if m2:
                    text = m2.group('text').strip()
                    # try inline bracket metadata
                    br = re.match(r'^\[(?P<flags>[^\]]+)\]\s*(?P<rest>.*)$', text)
                    meta_raw = ''
                    if br:
                        raw = br.group('flags')
                        # convert comma/semicolon separated to semi-colon style
                        meta_raw = raw.replace(',', ';')
                        text = br.group('rest').strip()
                    else:
                        continue
                    meta = {}
                else:
                    continue
            else:
                text = m.group('text').strip()
                meta_raw = m.group('meta') or ''
                meta = {}
                if meta_raw:
                    for part in meta_split_re.split(meta_raw):
                        if not part.strip():
                            continue
                        kv = kv_re.match(part)
                        if kv:
                            k = kv.group(1).strip().lower()
                            v = kv.group(2).strip().lower()
                            meta[k] = v
                        else:
                            meta[part.strip().lower()] = 'true'

            # Determine apex
            apex = meta.get('apex', 'false').lower() in ('1', 'true', 'yes', 'y', 't')
            if not apex:
                # also accept a textual prefix [apex] in the content
                if text.lower().startswith('[apex]'):
                    apex = True
                    text = text[len('[apex]'):].strip()

            # if not apex, skip for now — auto-promote handled elsewhere
            if not apex:
                # leave non-apex items; the caller may request auto-promote
                # store them under a special key for possible later promotion
                if '_unmarked' not in swimlanes:
                    swimlanes['_unmarked'] = []
                swimlanes['_unmarked'].append(text)
                continue

            swimlane = meta.get('swimlane', 'unspecified')
            if swimlane not in swimlanes:
                swimlanes[swimlane] = []
            swimlanes[swimlane].append(text)

    return swimlanes, inference_map


def infer_swimlane(text):
    """Heuristic keyword-based swimlane inference. Returns (swimlane, confidence).
    Confidence is a float between 0.0 and 1.0.
    """
    t = text.lower()
    # Mapping of swimlane -> keywords (ordered roughly by importance)
    kws = {
        'ir': ['sea-of-nodes', 'sea of nodes', 'seaofnodes', 'ir backend', 'ir backend', 'band', 'sea', 'nodes', 'sea-of-nodes', 'ir'],
        'stage0': ['stage0', 'emitter', 'transpile', 'transpiler', 'stage0_cli', 'emitter.cpp', 'emitter'],
        'patterns': ['pattern', 'lowering', 'tblgen', 'n-way', 'nway', 'lowering', 'pattern-matching'],
        'tests': ['test', 'regression', 'harness', 'integration', 'unit test', 'regression-suite'],
        'build': ['cmake', 'build', 'ci', 'workflow', 'build system']
    }
    scores = {}
    for lane, words in kws.items():
        score = 0
        for w in words:
            if w in t:
                score += 1
        scores[lane] = score
    # pick best
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return ('unspecified', 0.0)
    # confidence heuristic: normalize by number of words checked (clamped)
    total_keywords = sum(len(v) for v in kws.values())
    conf = min(0.95, 0.4 + (best[1] / max(1, total_keywords)))
    return (best[0], round(conf, 2))


def emit_markdown(swimlanes):
    out_lines = []
    out_lines.append('# APEX Features — Swimlanes')
    out_lines.append('')
    for lane, items in swimlanes.items():
        out_lines.append(f'## {lane}')
        out_lines.append('')
        for it in items:
            out_lines.append(f'- {it}')
        out_lines.append('')
    return '\n'.join(out_lines)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-\s_]", '', s)
    s = re.sub(r"[\s_]+", '-', s)
    return s


def load_story_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # expect keys: filename_template, body_template (both with placeholders)
    return data


def render_template(template_str: str, context: dict) -> str:
    # very small placeholder replacement using {{key}}
    def repl(m):
        key = m.group(1).strip()
        return str(context.get(key, ''))
    return re.sub(r"{{\s*([^}]+)\s*}}", repl, template_str)


def emit_stories(swimlanes, template_path, story_dir):
    tpl = load_story_template(template_path)
    filename_tpl = tpl.get('filename_template', '{{slug}}.story.md')
    body_tpl = tpl.get('body_template', '# {{title}}\n\n{{body}}')
    pathlib.Path(story_dir).mkdir(parents=True, exist_ok=True)
    created = []
    for lane, items in swimlanes.items():
        for it in items:
            title = it
            slug = slugify(title)
            context = {'title': title, 'swimlane': lane, 'todo': it, 'slug': slug}
            filename = render_template(filename_tpl, context)
            body = render_template(body_tpl, context)
            out_path = os.path.join(story_dir, filename)
            # avoid overwriting existing files
            if os.path.exists(out_path):
                # append a numeric suffix
                i = 1
                base, ext = os.path.splitext(filename)
                while os.path.exists(os.path.join(story_dir, f"{base}-{i}{ext}")):
                    i += 1
                out_path = os.path.join(story_dir, f"{base}-{i}{ext}")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(body)
            # return metadata for each created story
            created.append({'swimlane': lane, 'title': title, 'slug': slug, 'story_path': out_path, '_orig_todo': it})
    return created


def _get_audit_key_path():
    return os.path.expanduser('~/.bmad_core_audit_key')


def get_audit_key():
    """Return a Fernet key (bytes). Prefer env var BMAD_AUDIT_KEY, else a file in user home, else generate and persist."""
    if not _HAS_CRYPTO:
        return None
    env = os.environ.get('BMAD_AUDIT_KEY')
    if env:
        return env.encode('utf-8') if isinstance(env, str) else env
    path = _get_audit_key_path()
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                k = f.read().strip()
                return k
        # generate and persist
        k = Fernet.generate_key()
        # write with restrictive perms
        with open(path, 'wb') as f:
            f.write(k)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return k
    except Exception:
        return None


def write_encrypted_audit(mapping, out_dir='.bmad-core/audit'):
    """Write encrypted JSON mapping to an .enc file under out_dir using Fernet key."""
    if not _HAS_CRYPTO:
        raise RuntimeError('cryptography package not available')
    key = get_audit_key()
    if not key:
        raise RuntimeError('Unable to obtain audit key')
    f = Fernet(key)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    fname = os.path.join(out_dir, f'audit_{ts}.json.enc')
    data = json.dumps(mapping, ensure_ascii=False, indent=2).encode('utf-8')
    token = f.encrypt(data)
    with open(fname, 'wb') as fh:
        fh.write(token)
    try:
        os.chmod(fname, 0o600)
    except Exception:
        pass
    return fname


def gh_cli_available():
    try:
        subprocess.run(['gh', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def issue_exists_via_api(repo, title, token):
    # list issues and check matching title (simple, paginated first page only)
    url = f'https://api.github.com/repos/{repo}/issues?state=all&per_page=100'
    req = request.Request(url)
    req.add_header('Authorization', f'token {token}')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    try:
        with request.urlopen(req) as resp:
            data = json.load(resp)
            for it in data:
                if it.get('title', '').strip().lower() == title.strip().lower():
                    return True
    except error.HTTPError as e:
        print('HTTP error while checking issues:', e, file=sys.stderr)
    except Exception as e:
        print('Error while checking issues:', e, file=sys.stderr)
    return False


def create_issue_via_api(repo, title, body, labels, assignees, token):
    url = f'https://api.github.com/repos/{repo}/issues'
    payload = {'title': title, 'body': body}
    if labels:
        payload['labels'] = labels
    if assignees:
        payload['assignees'] = assignees
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, method='POST')
    req.add_header('Authorization', f'token {token}')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    req.add_header('Content-Type', 'application/json')
    try:
        with request.urlopen(req) as resp:
            return json.load(resp)
    except error.HTTPError as e:
        err = e.read().decode('utf-8')
        print('GitHub API error creating issue:', e, err, file=sys.stderr)
    except Exception as e:
        print('Error creating issue via API:', e, file=sys.stderr)
    return None


def create_issue_via_gh_cli(repo, title, body, labels, assignees, dry_run=False):
    cmd = ['gh', 'issue', 'create', '--repo', repo, '--title', title, '--body', body]
    if labels:
        cmd += ['--label', ','.join(labels)]
    if assignees:
        cmd += ['--assignee', ','.join(assignees)]
    if dry_run:
        print('DRY RUN gh:', ' '.join(cmd), file=sys.stderr)
        return None
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print('gh CLI error:', proc.stderr, file=sys.stderr)
        return None
    return proc.stdout.strip()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('todo', help='Path to TODO.md')
    ap.add_argument('--create-issues', action='store_true', help='Create GitHub issues for apex items')
    ap.add_argument('--repo', help='GitHub repo owner/name (required for issue creation)')
    ap.add_argument('--assignee', help='Assignee username for issues (comma separated)')
    ap.add_argument('--labels', help='Comma-separated labels to add to issues')
    ap.add_argument('--dry-run', action='store_true', help='Do not actually create issues; show what would be done')
    ap.add_argument('--emit-stories', action='store_true', help='Emit BMAD story files from template')
    ap.add_argument('--story-dir', default='.stories', help='Directory to write emitted stories')
    ap.add_argument('--manifest', default=None, help='Path to write CSV manifest of created tasks')
    ap.add_argument('--quiet', action='store_true', help='Suppress non-error informational logging')
    ap.add_argument('--echo', action='store_true', help='Allow echoing raw TODO text into created issues (opt-in). Default: do not echo')
    ap.add_argument('--auto-promote', action='store_true', help='Auto-promote unmarked checklist items to APEX using heuristic inference')
    args = ap.parse_args(argv[1:])

    QUIET = args.quiet

    def info(*a, **k):
        if not QUIET:
            print(*a, file=sys.stderr, **k)

    def short_id(s):
        import hashlib
        h = hashlib.sha1(s.encode('utf-8')).hexdigest()
        return h[:8]

    swimlanes, inference_map = parse_todo(args.todo)

    # Auto-promote unmarked tasks if requested
    if args.auto_promote and '_unmarked' in swimlanes:
        unmarked = swimlanes.pop('_unmarked')
        promoted = []
        for text in unmarked:
            lane, conf = infer_swimlane(text)
            # require minimal confidence threshold to avoid noise
            if conf >= 0.2:
                if lane not in swimlanes:
                    swimlanes[lane] = []
                swimlanes[lane].append(text)
                promoted.append({'text': text, 'swimlane': lane, 'confidence': conf})
            else:
                # low confidence: put under 'unspecified' so it's visible
                if 'unspecified' not in swimlanes:
                    swimlanes['unspecified'] = []
                swimlanes['unspecified'].append(text)
        if promoted:
            info(f'Auto-promoted {len(promoted)} unmarked items to APEX (use --auto-promote to enable)')

    md = emit_markdown(swimlanes)

    created_items = []

    if args.create_issues:
        if not args.repo:
            print('Error: --repo is required when --create-issues is used', file=sys.stderr)
            sys.exit(2)
        labels = [l.strip() for l in args.labels.split(',')] if args.labels else []
        assignees = [a.strip() for a in args.assignee.split(',')] if args.assignee else []
        use_gh = gh_cli_available()
        token = os.environ.get('GITHUB_TOKEN')
        if not use_gh and not token:
            print('Error: neither gh CLI found nor GITHUB_TOKEN provided. Cannot create issues.', file=sys.stderr)
            sys.exit(2)

        # For each swimlane/item, create an issue titled by the item text (shortened) with body referencing TODO
        # Build a list of tuples so we can correlate later with emitted stories
        for lane, items in swimlanes.items():
            for it in items:
                # Privacy: by default do not echo raw TODO text into external services.
                if args.echo:
                    title = it if len(it) <= 120 else it[:117] + '...'
                    body = f'APEX TODO (swimlane: {lane})\n\nFrom TODO.md:\n\n> {it}\n\n(Generated by scripts/extract_apex_swimlanes.py)'
                else:
                    sid = short_id(it)
                    title = f'APEX task ({lane}) [{sid}]'
                    # Provide metadata and pointer to repository TODO; avoid echoing user text
                    repo = args.repo or 'REPO'
                    body = (
                        f'APEX task (swimlane: {lane})\n\n'
                        f'Task id: {sid}\n\n'
                        f'Source: {repo} - TODO.md\n\n'
                        f'(Automatically generated; original TODO text not included for privacy)'
                    )
                labels_aug = labels + [f'swimlane:{lane}', 'apex']
                created_meta = {'swimlane': lane, 'todo': it, 'title': title, 'labels': labels_aug}
                created_items.append(created_meta)
                if use_gh:
                    if args.dry_run:
                        info('DRY: would create issue via gh for', title)
                        created_meta['issue'] = None
                        continue
                    # We still should check existence via API if token present, else best-effort
                    if token and issue_exists_via_api(args.repo, title, token):
                        info('Issue already exists (skipping):', title)
                        created_meta['issue'] = 'exists'
                        continue
                    res = create_issue_via_gh_cli(args.repo, title, body, labels_aug, assignees, dry_run=args.dry_run)
                    if res:
                        info('Created (gh):', title)
                        created_meta['issue'] = res
                else:
                    # use REST API
                    if issue_exists_via_api(args.repo, title, token):
                        info('Issue already exists (skipping):', title)
                        created_meta['issue'] = 'exists'
                        continue
                    if args.dry_run:
                        info('DRY: would create issue via API for', title)
                        created_meta['issue'] = None
                        continue
                    created = create_issue_via_api(args.repo, title, body, labels_aug, assignees, token)
                    if created:
                        info('Created (api):', created.get('html_url'))
                        created_meta['issue'] = created.get('number') or created.get('html_url')

    # Optionally emit stories and collect created story metadata
    if args.emit_stories:
        tpl = os.path.join('.bmad-core', 'templates', 'apex-story-tmpl.yaml')
        story_created = emit_stories(swimlanes, tpl, args.story_dir)
        info(f'Emitted {len(story_created)} story files to {args.story_dir}')
        # Try to correlate created_items with emitted stories by title/slug
        # If there were no created_items (no issue creation), build mapping from story_created
        if not created_items:
            for s in story_created:
                created_items.append({'swimlane': s.get('swimlane'), 'todo': s.get('_orig_todo'), 'title': s.get('title'), 'slug': s.get('slug'), 'story_path': s.get('story_path'), 'labels': []})
        else:
            # Attach story paths to created_items where titles match (best effort)
            for ci in created_items:
                for s in story_created:
                    if ci.get('todo') and s.get('_orig_todo') and ci.get('todo').strip() == s.get('_orig_todo').strip():
                        ci['slug'] = s.get('slug')
                        ci['story_path'] = s.get('story_path')
                        break

    # If manifest requested, write CSV
    if args.manifest:
        try:
            os.makedirs(os.path.dirname(args.manifest) or '.', exist_ok=True)
            with open(args.manifest, 'w', newline='', encoding='utf-8') as mf:
                writer = csv.writer(mf)
                writer.writerow(['task_id', 'swimlane', 'slug', 'story_path', 'issue', 'labels'])
                for ci in created_items:
                    tid = short_id(ci.get('todo', ci.get('title', '')))
                    writer.writerow([tid, ci.get('swimlane'), ci.get('slug', ''), ci.get('story_path', ''), ci.get('issue', ''), ','.join(ci.get('labels') or [])])
            info(f'Wrote manifest CSV to {args.manifest}')
        except Exception as e:
            info('Failed to write manifest:', e)

    # Finally, print the markdown document to stdout
    # Optionally emit stories
    # write encrypted audit unless disabled via env
    disable_audit = os.environ.get('BMAD_DISABLE_LOCAL_AUDIT') in ('1', 'true', 'True')
    if not disable_audit and created_items:
        try:
            mapping = {}
            for ci in created_items:
                tid = short_id(ci.get('todo', ci.get('title', '')))
                mapping[tid] = {
                    'swimlane': ci.get('swimlane'),
                    'slug': ci.get('slug', ''),
                    'story_path': ci.get('story_path', ''),
                    'issue': ci.get('issue', ''),
                    'orig': ci.get('todo', '')
                }
            try:
                fname = write_encrypted_audit(mapping)
                info(f'Encrypted local audit written: {fname}')
            except Exception as e:
                info('Skipping encrypted audit (error):', e)
        except Exception as e:
            info('Failed building/writing audit:', e)

    print(md)


if __name__ == '__main__':
    main(sys.argv)
