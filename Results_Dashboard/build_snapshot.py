"""Package supplied project outputs for a completely local, offline-capable view."""
from pathlib import Path
import json
import re
import shutil
from urllib.parse import quote, unquote, urlsplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DIST = HERE / 'dist'
GITHUB = 'https://github.com/sudipphuyal/rwth_hackathon/blob/main/'
GITHUB_RAW = 'https://raw.githubusercontent.com/sudipphuyal/rwth_hackathon/main/'
DIST.mkdir(exist_ok=True)
evidence = json.loads((HERE / 'data-evidence.json').read_text())
procurement = json.loads((HERE / 'data-procurement.json').read_text())
paths = [s['path'] for s in evidence['sources'].values()]
paths += [s['relativePath'] for s in procurement['sources']]
source_files = {}
copied_sources = []
LINKS = re.compile(r'(!?)\[([^\]]*)\]\(([^)\n]+)\)')
(DIST / 'sources').mkdir(exist_ok=True)

def copy_source(relative):
    if relative in source_files:
        return
    source = PROJECT / relative
    if not source.is_file() or source.stat().st_size > 2_000_000:
        return
    target = f"{len(copied_sources) + 1:02d}-{source.name.replace(' ', '-')}"
    shutil.copyfile(source, DIST / 'sources' / target)
    source_files[relative] = 'sources/' + target
    copied_sources.append((source, DIST / 'sources' / target))

def local_link(source, url):
    parts = urlsplit(url.strip('<>'))
    if parts.scheme or parts.netloc or not parts.path:
        return None
    path = unquote(parts.path)
    line = re.search(r':(\d+)$', path)
    fragment = parts.fragment
    if line:
        path = path[:line.start()]
        fragment = fragment or 'L' + line.group(1)
    return (source.parent / path).resolve(), fragment

for relative in dict.fromkeys(paths):
    copy_source(relative)

# Follow report references so small figures, audits and related documents work offline.
for source, _ in copied_sources:
    if source.suffix == '.md':
        for match in LINKS.finditer(source.read_text()):
            link = local_link(source, match.group(3))
            if link and link[0].is_relative_to(PROJECT):
                copy_source(link[0].relative_to(PROJECT).as_posix())

def portable_markdown(text, source):
    """Keep links valid after reports move into the dashboard evidence bundle."""
    def rewrite(match):
        image, label, url = match.groups()
        link = local_link(source, url)
        if link is None:
            return match.group(0)
        target, fragment = link
        try:
            relative = target.relative_to(PROJECT).as_posix()
        except ValueError:
            return f'{label} (historical source not included in this repository)'
        if not target.is_file():
            return f'{label} (historical source not included in this repository)'
        if relative in source_files:
            url = quote(Path(source_files[relative]).name)
        else:
            url = (GITHUB_RAW if image else GITHUB) + quote(relative, safe='/')
        if fragment:
            url += '#' + fragment
        return f'{image}[{label}]({url})'
    return LINKS.sub(rewrite, text)

for source, target in copied_sources:
    if source.suffix == '.md':
        target.write_text(portable_markdown(source.read_text(), source))
payload = {'evidence': evidence, 'procurement': procurement, 'sourceFiles': source_files}
(DIST / 'data.js').write_text('window.DASHBOARD_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/') + ';\n')
print(f"Local data packaged: {len(evidence['dailySeries']['rows'])} daily rows, {len(procurement['dailyRows'])} procurement scenarios, {len(source_files)} source files.")
