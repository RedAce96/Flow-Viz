"""Portable HTML reporting from manifests and registered artifacts only."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .artifacts import ArtifactRegistry


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    plan = _read_json(root / "plan.json")
    registry = ArtifactRegistry.load(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "index.html"
    statuses = manifest.get("workflows", {})
    status_rows = "".join(
        "<tr><td>{}</td><td class='{}'>{}</td><td><pre>{}</pre></td></tr>".format(
            html.escape(name), html.escape(item.get("status", "unknown")),
            html.escape(item.get("status", "unknown")), html.escape(item.get("message", "")),
        )
        for name, item in sorted(statuses.items())
    )
    findings = "".join(
        f"<li class='{html.escape(str(item['severity']).lower())}'>"
        f"<strong>{html.escape(str(item['severity']))} {html.escape(item['code'])}</strong>: "
        f"{html.escape(item['message'])}</li>"
        for item in plan.get("findings", [])
    ) or "<li>No preflight findings.</li>"
    gallery: list[str] = []
    products: list[str] = []
    for artifact in registry.artifacts:
        target = Path("..") / artifact.path
        link = html.escape(target.as_posix())
        label = html.escape(artifact.id)
        interpretation = html.escape(artifact.interpretation)
        products.append(
            f"<li><a href='{link}'>{label}</a> [{html.escape(artifact.kind)}] — {interpretation}</li>"
        )
        if artifact.kind == "figure" and Path(artifact.path).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            gallery.append(
                f"<figure><a href='{link}'><img src='{link}' alt='{label}'></a>"
                f"<figcaption>{label}: {interpretation}</figcaption></figure>"
            )
    provenance = html.escape(json.dumps(manifest.get("provenance", {}), indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PeleC post-processing report: {html.escape(manifest.get('case_id', 'unknown'))}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5df;padding:.5rem;text-align:left}}
pre{{white-space:pre-wrap}}img{{max-width:100%;height:auto}}figure{{margin:1.5rem 0}}
.completed{{color:#166534}}.failed,.blocker{{color:#b91c1c}}.warning{{color:#a16207}}
.skipped-by-dependency,.unavailable{{color:#6b21a8}}code{{background:#eef2f6;padding:.1rem .25rem}}
</style></head><body>
<h1>PeleC post-processing report</h1>
<p><strong>Case:</strong> {html.escape(manifest.get('case_id', 'unknown'))}<br>
<strong>Run:</strong> {html.escape(manifest.get('run_id', root.name))}<br>
<strong>Status:</strong> {html.escape(manifest.get('status', 'unknown'))}</p>
<h2>Preflight findings and assumptions</h2><ul>{findings}</ul>
<h2>Workflow status</h2><table><thead><tr><th>Workflow</th><th>Status</th><th>Message</th></tr></thead>
<tbody>{status_rows}</tbody></table>
<h2>Figure gallery</h2>{''.join(gallery) or '<p>No figures registered.</p>'}
<h2>Numerical products and evidence</h2><ul>{''.join(products) or '<li>No artifacts registered.</li>'}</ul>
<h2>Interpretation limits</h2><p>Each artifact description states its interpretation contract. Warnings above remain part of the scientific record and must accompany copied products.</p>
<h2>Provenance</h2><pre>{provenance}</pre>
</body></html>"""
    temporary = report_dir / ".index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path
