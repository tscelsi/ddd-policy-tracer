"""Static HTML graph viewer rendering for Stage 5 artifacts."""

from __future__ import annotations

from pathlib import Path

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
        <pre id=\"details\">Click a node or edge to inspect properties.</pre>
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

      fetch('graph.filtered.json')
        .then((response) => response.json())
        .then((payload) => {
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
            layout: { name: 'cose', animate: false, fit: true, padding: 30 },
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': (ele) => nodeColor(ele),
                  label: 'data(label)',
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

          const details = document.getElementById('details')
          cy.on('tap', 'node, edge', (event) => {
            const data = event.target.data()
            details.textContent = JSON.stringify(data, null, 2)
          })
        })
        .catch((error) => {
          const details = document.getElementById('details')
          details.textContent = 'Failed to load graph.filtered.json: ' + String(error)
        })
    </script>
  </body>
</html>
"""


def render_graph_html(*, output_directory: Path) -> None:
    """Render an interactive static viewer that loads graph.filtered.json."""
    (output_directory / "graph.html").write_text(_GRAPH_HTML, encoding="utf-8")
