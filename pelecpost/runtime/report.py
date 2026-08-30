"""Portable HTML reporting from manifests and registered artifacts only."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re

from .artifacts import ArtifactRegistry


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "item"


def generate_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    plan = _read_json(root / "plan.json")
    registry = ArtifactRegistry.load(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "index.html"
    statuses = manifest.get("workflows", {})
    contracts = plan.get("analysis_contracts", {})
    status_rows = "".join(
        "<tr><td>{}</td><td class='{}'>{}</td><td><pre>{}</pre></td></tr>".format(
            html.escape(name), html.escape(item.get("status", "unknown")),
            html.escape(item.get("status", "unknown")), html.escape(item.get("message", "")),
        )
        for name, item in sorted(statuses.items())
    )
    findings_items = []
    for item in plan.get("findings", []):
        owner = item.get("analysis_id")
        owner_link = (
            f" <a href='#contract-{_anchor(owner)}'>analysis contract</a>"
            if owner in contracts else ""
        )
        findings_items.append(
            f"<li id='finding-{_anchor(str(item['code']))}-{_anchor(str(owner or 'run'))}' "
            f"class='{html.escape(str(item['severity']).lower())}'>"
            f"<strong>{html.escape(str(item['severity']))} {html.escape(item['code'])}</strong>: "
            f"{html.escape(item['message'])}{owner_link}</li>"
        )
    findings = "".join(findings_items) or "<li>No preflight findings.</li>"
    contract_sections = []
    limitation_items: list[str] = []
    for analysis_id, contract in contracts.items():
        assumptions = "".join(f"<li>{html.escape(value)}</li>" for value in contract["assumptions"])
        limitations = "".join(f"<li>{html.escape(value)}</li>" for value in contract["limitations"])
        outputs = ", ".join(html.escape(value) for value in contract["expected_artifact_ids"])
        contract_sections.append(
            f"<article id='contract-{_anchor(analysis_id)}'><h3>{html.escape(analysis_id)}: "
            f"{html.escape(contract['physical_question'])}</h3><p><strong>Recipe:</strong> "
            f"{html.escape(contract['recipe'])}<br><strong>Declared outputs:</strong> {outputs}</p>"
            f"<h4>Assumptions and quality gates</h4><ul>{assumptions}</ul>"
            f"<h4>Interpretation limits</h4><ul>{limitations}</ul></article>"
        )
        limitation_items.extend(
            f"<li><a href='#contract-{_anchor(analysis_id)}'>{html.escape(analysis_id)}</a>: "
            f"{html.escape(value)}</li>" for value in contract["limitations"]
        )
    gallery: list[str] = []
    products: list[str] = []
    evidence_rows: list[str] = []
    for artifact in registry.artifacts:
        target = Path("..") / artifact.path
        link = html.escape(target.as_posix())
        label = html.escape(artifact.id)
        interpretation = html.escape(artifact.interpretation)
        contract_link = (
            f"<a href='#contract-{_anchor(artifact.recipe_instance)}'>"
            f"{html.escape(artifact.recipe_instance)}</a>"
            if artifact.recipe_instance in contracts else html.escape(artifact.recipe_instance)
        )
        products.append(
            f"<li><a href='{link}'>{label}</a> [{html.escape(artifact.kind)}; "
            f"{contract_link}] — {interpretation}</li>"
        )
        evidence_rows.append(
            f"<tr><td><a href='{link}'>{label}</a></td><td>{contract_link}</td>"
            f"<td>{interpretation}</td><td><a href='#preflight'>preflight gates</a></td></tr>"
        )
        if artifact.kind == "figure" and Path(artifact.path).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            gallery.append(
                f"<figure><a href='{link}'><img src='{link}' alt='{label}'></a>"
                f"<figcaption>{label}: {interpretation}</figcaption></figure>"
            )
    provenance = html.escape(json.dumps(manifest.get("provenance", {}), indent=2, sort_keys=True))
    inventory = html.escape(json.dumps(plan.get("inventory", {}), indent=2, sort_keys=True))
    resources = html.escape(json.dumps(plan.get("sampling", {}), indent=2, sort_keys=True))
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
<h2>Project and input inventory</h2><p>The resolved project files are registered below. This inventory was
captured before computation and uses metadata-only readers.</p><pre>{inventory}</pre>
<h2 id="preflight">Preflight findings</h2><ul>{findings}</ul>
<details><summary>Selections, sampling, geometry, and resource calculations</summary><pre>{resources}</pre></details>
<h2>Analysis contracts</h2>{''.join(contract_sections) or '<p>No analyses were selected.</p>'}
<h2>Workflow status</h2><table><thead><tr><th>Workflow</th><th>Status</th><th>Message</th></tr></thead>
<tbody>{status_rows}</tbody></table>
<h2>Figure gallery</h2>{''.join(gallery) or '<p>No figures registered.</p>'}
<h2>Numerical products</h2><ul>{''.join(products) or '<li>No artifacts registered.</li>'}</ul>
<h2>Evidence matrix</h2><table><thead><tr><th>Product</th><th>Analysis contract</th><th>Permitted interpretation</th><th>Quality gates</th></tr></thead>
<tbody>{''.join(evidence_rows) or '<tr><td colspan="4">No artifacts registered.</td></tr>'}</tbody></table>
<h2>Interpretation limits</h2><ul>{''.join(limitation_items) or '<li>No recipe limitations registered.</li>'}</ul>
<p>Warnings and assumptions linked above remain part of the scientific record and must accompany copied products.</p>
<h2>Provenance</h2><pre>{provenance}</pre>
</body></html>"""
    temporary = report_dir / ".index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path
