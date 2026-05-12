"""Static HTML graph viewer rendering for Stage 5 artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .contracts import GraphArtifact

_GRAPH_HTML = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Policy Tracer Graph</title>
    <script src=\"https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js\"></script>
    <style>
      :root {
        --bg: #f4f1ea;
        --panel: #fffdf8;
        --ink: #1f2933;
        --muted: #52606d;
        --border: #d9d4c7;
      }
      body {
        margin: 0;
        font-family: Georgia, \"Times New Roman\", serif;
        color: var(--ink);
        background: radial-gradient(circle at 20% 20%, #fff8e8, var(--bg));
      }
      .layout {
        display: grid;
        grid-template-columns: 1fr 320px;
        min-height: 100vh;
      }
      #graph {
        height: 100vh;
      }
      .sidebar {
        border-left: 1px solid var(--border);
        background: var(--panel);
        padding: 16px;
        overflow: auto;
        overflow-x: hidden;
      }
      h1 {
        margin: 0 0 8px;
        font-size: 1.2rem;
      }
      .legend {
        margin: 12px 0;
        display: grid;
        gap: 6px;
      }
      .legend-item {
        display: grid;
        grid-template-columns: 16px 1fr;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
      }
      .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
      }
      pre {
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.8rem;
        color: var(--muted);
      }
      .selection-empty {
        color: var(--muted);
        font-style: italic;
      }
      .selection-grid {
        display: grid;
        gap: 10px;
      }
      .selection-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
        background: #fffdf8;
      }
      .selection-title {
        margin: 0 0 6px;
        font-size: 0.9rem;
      }
      .selection-kv {
        margin: 0;
        display: grid;
        grid-template-columns: 88px 1fr;
        gap: 4px 8px;
        font-size: 0.8rem;
      }
      .selection-kv dd {
        margin: 0;
        min-width: 0;
        overflow-wrap: anywhere;
      }
      .selection-k {
        color: var(--muted);
      }
      .selection-json {
        margin-top: 8px;
        font-size: 0.75rem;
        max-height: 260px;
        max-width: 100%;
        overflow: auto;
        overflow-wrap: anywhere;
      }
      @media (max-width: 900px) {
        .layout {
          grid-template-columns: 1fr;
          grid-template-rows: 65vh auto;
        }
        .sidebar {
          border-left: 0;
          border-top: 1px solid var(--border);
        }
      }
    </style>
  </head>
  <body>
    <div class=\"layout\">
      <div id=\"graph\"></div>
      <aside class=\"sidebar\">
        <h1>Policy Tracer Graph</h1>
        <p>Pan/zoom the graph and click a node or edge for metadata.</p>
        <div class=\"legend\">
          <strong>Entity Type Legend</strong>
          <div class=\"legend-item\">
            <span class=\"dot\" style=\"background:#d68910\"></span>ORG
          </div>
          <div class=\"legend-item\">
            <span class=\"dot\" style=\"background:#1f618d\"></span>PERSON
          </div>
          <div class=\"legend-item\">
            <span class=\"dot\" style=\"background:#117a65\"></span>POLICY
          </div>
          <div class=\"legend-item\">
            <span class=\"dot\" style=\"background:#7d3c98\"></span>JURISDICTION
          </div>
          <div class=\"legend-item\">
            <span class=\"dot\" style=\"background:#b03a2e\"></span>PROGRAM
          </div>
        </div>
        <h2 style=\"font-size:1rem;margin-top:12px\">Selection</h2>
        <div id=\"details\" class=\"selection-empty\">Click a node or edge to inspect details.</div>
      </aside>
    </div>
    <script>
      const entityColors = {
        ORG: '#d68910',
        PERSON: '#1f618d',
        POLICY: '#117a65',
        JURISDICTION: '#7d3c98',
        PROGRAM: '#b03a2e',
      }

      function nodeColor(node) {
        const type = node.data('type')
        if (type === 'PublisherOrganization') return '#6c7a89'
        if (type === 'Claim') return '#2e4053'
        if (type === 'MentionedEntity') {
          const entityType = node.data('properties')?.entity_type
          return entityColors[entityType] || '#566573'
        }
        return '#566573'
      }

      function escapeHtml(value) {
        return String(value)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;')
      }

      function renderSelection(data, kind) {
        const details = document.getElementById('details')
        const properties = data.properties || {}
        const propertiesJson = JSON.stringify(properties, null, 2)
        details.className = 'selection-grid'
        details.innerHTML = `
          <section class="selection-card">
            <h3 class="selection-title">${escapeHtml(kind)} Summary</h3>
            <dl class="selection-kv">
              <dt class="selection-k">id</dt><dd>${escapeHtml(data.id || '')}</dd>
              <dt class="selection-k">type</dt><dd>${escapeHtml(data.type || '')}</dd>
              <dt class="selection-k">label</dt><dd>${escapeHtml(data.label || '')}</dd>
            </dl>
          </section>
          <section class="selection-card">
            <h3 class="selection-title">Properties</h3>
            <pre class="selection-json">${escapeHtml(propertiesJson)}</pre>
          </section>
        `
      }

      const payload = __EMBEDDED_GRAPH_PAYLOAD__
      try {
          const elements = []
          for (const node of payload.nodes) {
            elements.push({ data: node })
          }
          for (const edge of payload.edges) {
            elements.push({ data: edge })
          }

          const cy = cytoscape({
            container: document.getElementById('graph'),
            elements,
            layout: {
              name: 'cose',
              animate: false,
              fit: true,
              padding: 40,
              avoidOverlap: true,
              nodeRepulsion: 120000,
              idealEdgeLength: 130,
              edgeElasticity: 120,
              gravity: 0.2,
              numIter: 1400,
            },
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': (ele) => nodeColor(ele),
                  label: '',
                  color: '#202020',
                  'font-size': 10,
                  'text-wrap': 'wrap',
                  'text-max-width': 140,
                },
              },
              {
                selector: 'edge[type = "RAISED"]',
                style: {
                  width: 2,
                  'line-color': '#5d6d7e',
                  'target-arrow-color': '#5d6d7e',
                  'target-arrow-shape': 'triangle',
                  'curve-style': 'bezier',
                },
              },
              {
                selector: 'edge[type = "MENTIONS"]',
                style: {
                  width: 2,
                  'line-color': '#85929e',
                  'target-arrow-color': '#85929e',
                  'target-arrow-shape': 'triangle',
                  'curve-style': 'bezier',
                },
              },
            ],
          })

          cy.on('tap', 'node', (event) => {
            renderSelection(event.target.data(), 'Node')
          })

          cy.on('tap', 'edge', (event) => {
            renderSelection(event.target.data(), 'Edge')
          })

          cy.on('mouseover', 'node', (event) => {
            const node = event.target
            node.style('label', node.data('label') || '')
          })

          cy.on('mouseout', 'node', (event) => {
            event.target.style('label', '')
          })
      } catch (error) {
        const details = document.getElementById('details')
        details.textContent = 'Failed to render embedded graph data: ' + String(error)
      }
    </script>
  </body>
</html>
"""


def render_graph_html(*, output_directory: Path, filtered_artifact: GraphArtifact) -> None:
    """Render an interactive static viewer with embedded filtered graph data."""
    embedded_payload = json.dumps(asdict(filtered_artifact), ensure_ascii=True)
    html = _GRAPH_HTML.replace(
        "__EMBEDDED_GRAPH_PAYLOAD__",
        embedded_payload,
    )
    (output_directory / "graph.html").write_text(html, encoding="utf-8")
