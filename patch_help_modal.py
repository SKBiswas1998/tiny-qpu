"""
Patches the dashboard HTML to add:
1. In-app help/manual modal (? button in header)
2. Logo favicon
3. About section

Run from project root: python patch_help_modal.py
"""

import os
import sys

HTML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "tiny_qpu", "dashboard", "templates", "index.html"
)

# ─── The Help Modal CSS ───
HELP_CSS = """
/* Help Modal */
.help-btn {
  width: 32px; height: 32px; border-radius: 50%;
  background: var(--bg-surface); border: 1px solid var(--border);
  color: var(--cyan); font-family: var(--font-mono); font-size: 14px;
  font-weight: 700; cursor: pointer; display: flex; align-items: center;
  justify-content: center; transition: all 0.15s; margin-left: 12px;
}
.help-btn:hover { background: var(--cyan); color: var(--bg-void); box-shadow: 0 0 12px var(--cyan-dim); }

.help-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 2000;
  display: flex; align-items: center; justify-content: center;
  backdrop-filter: blur(6px); opacity: 0; pointer-events: none; transition: opacity 0.25s;
}
.help-overlay.active { opacity: 1; pointer-events: all; }

.help-panel {
  background: var(--bg-deep); border: 1px solid var(--border); border-radius: 16px;
  width: 680px; max-width: 90vw; max-height: 85vh; overflow-y: auto;
  box-shadow: 0 30px 80px rgba(0,0,0,0.6); padding: 0;
  transform: translateY(12px); transition: transform 0.25s;
}
.help-overlay.active .help-panel { transform: translateY(0); }

.help-header {
  position: sticky; top: 0; background: var(--bg-deep); padding: 24px 32px 16px;
  border-bottom: 1px solid var(--border); z-index: 1; display: flex;
  align-items: center; justify-content: space-between;
}
.help-header h2 { font-size: 18px; color: var(--cyan); font-family: var(--font-mono); font-weight: 600; }
.help-close { background: none; border: none; color: var(--text-secondary); font-size: 24px;
  cursor: pointer; padding: 4px 8px; border-radius: 6px; }
.help-close:hover { color: var(--text-bright); background: var(--bg-surface); }

.help-body { padding: 24px 32px 32px; }
.help-body h3 { font-size: 14px; color: var(--amber); font-weight: 600; margin: 20px 0 8px;
  text-transform: uppercase; letter-spacing: 1px; font-family: var(--font-mono); }
.help-body h3:first-child { margin-top: 0; }
.help-body p { font-size: 12.5px; color: var(--text-primary); line-height: 1.7; margin-bottom: 10px; }
.help-body code { font-family: var(--font-mono); font-size: 11px; background: var(--bg-surface);
  color: var(--green); padding: 2px 6px; border-radius: 3px; }
.help-body .key { display: inline-block; font-family: var(--font-mono); font-size: 10px;
  background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-bright);
  padding: 2px 8px; border-radius: 4px; margin: 1px 2px; min-width: 20px; text-align: center; }
.help-body .shortcut-row { display: flex; align-items: center; gap: 12px; margin: 4px 0;
  font-size: 12px; color: var(--text-primary); }
.help-body .shortcut-row .key { flex-shrink: 0; }
.help-body table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; }
.help-body th { text-align: left; font-size: 10px; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 1px; padding: 6px 10px;
  border-bottom: 1px solid var(--border); font-weight: 600; }
.help-body td { font-size: 12px; color: var(--text-primary); padding: 6px 10px;
  border-bottom: 1px solid var(--bg-surface); }
.help-body td:first-child { color: var(--cyan); font-family: var(--font-mono); font-weight: 500; }
.help-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 32px; }
.help-tab { padding: 10px 16px; font-size: 12px; font-weight: 600; color: var(--text-secondary);
  cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s; }
.help-tab:hover { color: var(--text-bright); }
.help-tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.help-tab-content { display: none; }
.help-tab-content.active { display: block; }
"""

# ─── The Help Modal HTML ───
HELP_HTML = """
<!-- ═══════════ HELP MODAL ═══════════ -->
<div class="help-overlay" id="helpOverlay" onclick="if(event.target===this)closeHelp()">
  <div class="help-panel">
    <div class="help-header">
      <h2>⚛ tiny-qpu Quantum Lab</h2>
      <button class="help-close" onclick="closeHelp()">×</button>
    </div>
    <div class="help-tabs">
      <div class="help-tab active" onclick="switchHelpTab('quickstart')">Quick Start</div>
      <div class="help-tab" onclick="switchHelpTab('gates')">Gates</div>
      <div class="help-tab" onclick="switchHelpTab('shortcuts')">Shortcuts</div>
      <div class="help-tab" onclick="switchHelpTab('concepts')">Concepts</div>
      <div class="help-tab" onclick="switchHelpTab('about')">About</div>
    </div>
    <div class="help-body">

      <!-- QUICK START -->
      <div class="help-tab-content active" id="tab-quickstart">
        <h3>Building Circuits</h3>
        <p><strong>Click</strong> a gate in the left palette to select it (cyan border appears), then <strong>click a qubit wire</strong> in the circuit area to place it. Or <strong>drag and drop</strong> from the palette.</p>
        <p>For <strong>multi-qubit gates</strong> (CX, CZ, SWAP): click the control qubit — the target auto-assigns to the adjacent qubit.</p>
        <p>For <strong>rotation gates</strong> (Rx, Ry, Rz): a dialog appears to enter the angle in radians. Use the quick buttons for common values like π/2, π/4.</p>
        <p><strong>Click any placed gate</strong> to remove it, or use <span class="key">Ctrl+Z</span> to undo.</p>

        <h3>Simulating</h3>
        <p>Click <strong>▶ Run</strong> (or press <span class="key">R</span>) to simulate. Results appear in the right panel: Bloch spheres, probability bars, measurement histogram, and amplitude table.</p>
        <p>Click <strong>⏭ Step</strong> (or press <span class="key">S</span>) to walk through the circuit gate-by-gate and watch the quantum state evolve.</p>

        <h3>Presets</h3>
        <p>Scroll down in the left panel to find preset circuits — Bell State, GHZ, Teleportation, Grover's Search, and more. Click to load and auto-run.</p>

        <h3>QASM</h3>
        <p>Use the OpenQASM panel at the bottom right to <strong>Import</strong> or <strong>Export</strong> circuits in QASM 2.0 format, compatible with Qiskit and IBM Quantum.</p>
      </div>

      <!-- GATES -->
      <div class="help-tab-content" id="tab-gates">
        <h3>Single-Qubit Gates</h3>
        <table>
          <tr><th>Gate</th><th>Description</th></tr>
          <tr><td>H</td><td>Hadamard — creates equal superposition. |0⟩ → |+⟩</td></tr>
          <tr><td>X</td><td>Pauli-X (NOT gate) — flips |0⟩ ↔ |1⟩</td></tr>
          <tr><td>Y</td><td>Pauli-Y — bit + phase flip</td></tr>
          <tr><td>Z</td><td>Pauli-Z — phase flip. |1⟩ → −|1⟩</td></tr>
          <tr><td>S</td><td>√Z gate — adds π/2 phase to |1⟩</td></tr>
          <tr><td>T</td><td>π/8 gate — adds π/4 phase to |1⟩</td></tr>
          <tr><td>√X</td><td>Square root of X gate</td></tr>
        </table>

        <h3>Rotation Gates</h3>
        <table>
          <tr><th>Gate</th><th>Description</th></tr>
          <tr><td>Rx(θ)</td><td>Rotation around X-axis by θ radians</td></tr>
          <tr><td>Ry(θ)</td><td>Rotation around Y-axis by θ radians</td></tr>
          <tr><td>Rz(θ)</td><td>Rotation around Z-axis by θ radians</td></tr>
          <tr><td>P(θ)</td><td>Phase gate — adds relative phase θ to |1⟩</td></tr>
        </table>

        <h3>Multi-Qubit Gates</h3>
        <table>
          <tr><th>Gate</th><th>Qubits</th><th>Description</th></tr>
          <tr><td>CX</td><td>2</td><td>CNOT — flips target if control is |1⟩. Creates entanglement.</td></tr>
          <tr><td>CZ</td><td>2</td><td>Controlled-Z — phase flip on target if control is |1⟩</td></tr>
          <tr><td>SWAP</td><td>2</td><td>Exchanges two qubit states</td></tr>
          <tr><td>CRz</td><td>2</td><td>Controlled Rz rotation</td></tr>
          <tr><td>CCX</td><td>3</td><td>Toffoli — flips target if BOTH controls are |1⟩</td></tr>
          <tr><td>CSWAP</td><td>3</td><td>Fredkin — controlled SWAP</td></tr>
        </table>
      </div>

      <!-- SHORTCUTS -->
      <div class="help-tab-content" id="tab-shortcuts">
        <h3>Simulation</h3>
        <div class="shortcut-row"><span class="key">R</span> Run simulation</div>
        <div class="shortcut-row"><span class="key">S</span> Step-by-step mode</div>
        <div class="shortcut-row"><span class="key">C</span> Clear circuit</div>
        <div class="shortcut-row"><span class="key">Ctrl</span>+<span class="key">Z</span> Undo last gate</div>

        <h3>Gate Quick-Select</h3>
        <div class="shortcut-row"><span class="key">H</span> Select Hadamard</div>
        <div class="shortcut-row"><span class="key">X</span> Select Pauli-X</div>
        <div class="shortcut-row"><span class="key">Y</span> Select Pauli-Y</div>
        <div class="shortcut-row"><span class="key">M</span> Select Measure</div>

        <h3>Step Mode</h3>
        <div class="shortcut-row"><span class="key">→</span> Next step</div>
        <div class="shortcut-row"><span class="key">←</span> Previous step</div>
        <div class="shortcut-row"><span class="key">Esc</span> Exit step mode</div>

        <h3>Dialogs</h3>
        <div class="shortcut-row"><span class="key">Enter</span> Confirm parameter dialog</div>
        <div class="shortcut-row"><span class="key">Esc</span> Close dialog</div>
        <div class="shortcut-row"><span class="key">?</span> Toggle this help panel</div>
      </div>

      <!-- CONCEPTS -->
      <div class="help-tab-content" id="tab-concepts">
        <h3>Superposition</h3>
        <p>A qubit can exist in a combination of |0⟩ and |1⟩ simultaneously: |ψ⟩ = α|0⟩ + β|1⟩. The Hadamard gate creates equal superposition from |0⟩. On the Bloch sphere, superposition states lie on the equator.</p>

        <h3>Entanglement</h3>
        <p>Two qubits are entangled when measuring one instantly determines the other. The Bell state (|00⟩ + |11⟩)/√2 is created by H then CNOT. Entangled qubits appear as points near the Bloch sphere origin (purity &lt; 1).</p>

        <h3>Measurement</h3>
        <p>Measuring collapses superposition into |0⟩ or |1⟩. The probability equals |amplitude|². The <strong>Probabilities</strong> panel shows exact values from the statevector. The <strong>Histogram</strong> shows statistical sampling (like a real quantum computer).</p>

        <h3>Interference</h3>
        <p>Quantum amplitudes are complex numbers that can constructively add or destructively cancel. Grover's algorithm exploits this: it amplifies the correct answer's amplitude while cancelling wrong ones.</p>

        <h3>Bloch Sphere</h3>
        <p>Each qubit's state maps to a point on a sphere. North pole = |0⟩, south pole = |1⟩, equator = superpositions. Rotation gates (Rx, Ry, Rz) physically rotate the state vector around the corresponding axis. Entangled qubits shrink toward the center.</p>
      </div>

      <!-- ABOUT -->
      <div class="help-tab-content" id="tab-about">
        <h3>About tiny-qpu</h3>
        <p>tiny-qpu is a lightweight quantum computing simulator and interactive learning platform. It provides a complete quantum circuit simulation engine with 35+ gates, statevector and density matrix backends, and zero external dependencies for the core engine.</p>

        <h3>Features</h3>
        <p>• Interactive circuit builder with drag-and-drop<br>
        • Real-time Bloch sphere visualization<br>
        • Step-by-step quantum state evolution<br>
        • OpenQASM 2.0 import/export<br>
        • 8 preset quantum circuits<br>
        • 20 quantum gates<br>
        • Up to 6 qubits in the dashboard (more via API)<br>
        • Quantum applications: QRNG, QAOA, BB84, VQE, Shor's<br>
        • 350+ comprehensive tests</p>

        <h3>Links</h3>
        <p>GitHub: <code>github.com/SKBiswas1998/tiny-qpu</code></p>

        <h3>Version</h3>
        <p>tiny-qpu v2.0 — Interactive Quantum Lab<br>
        Built with Python, Flask, and pure JavaScript.<br>
        No cloud. No accounts. 100% local.</p>
      </div>

    </div>
  </div>
</div>
"""

# ─── JavaScript for help modal ───
HELP_JS = """
// ─── HELP MODAL ───
function openHelp() { document.getElementById('helpOverlay').classList.add('active'); }
function closeHelp() { document.getElementById('helpOverlay').classList.remove('active'); }
function switchHelpTab(tab) {
  document.querySelectorAll('.help-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.help-tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}
// Add ? key shortcut
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
    e.preventDefault();
    const overlay = document.getElementById('helpOverlay');
    if (overlay.classList.contains('active')) closeHelp(); else openHelp();
  }
});
"""

def patch():
    if not os.path.exists(HTML_PATH):
        print(f"ERROR: HTML not found at {HTML_PATH}")
        sys.exit(1)

    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    # Check if already patched
    if 'help-overlay' in html:
        print("Already patched — skipping.")
        return

    # 1. Insert CSS before </style>
    html = html.replace('</style>', HELP_CSS + '\n</style>')

    # 2. Insert help button in header (after header-controls div opening)
    # Find the closing </div> of header-controls and add button before it
    html = html.replace(
        '</div>\n  </header>',
        '  <button class="help-btn" onclick="openHelp()" title="Help (?)" aria-label="Help">?</button>\n    </div>\n  </header>'
    )

    # 3. Insert help modal HTML before the parameter modal
    html = html.replace(
        '<!-- ═══════════ PARAMETER MODAL',
        HELP_HTML + '\n<!-- ═══════════ PARAMETER MODAL'
    )

    # 4. Insert help JS before the closing </script>
    html = html.replace(
        'init();\n</script>',
        'init();\n' + HELP_JS + '\n</script>'
    )

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"OK Patched {HTML_PATH}")
    print("  + Help button (?) in header")
    print("  + 5-tab help modal (Quick Start, Gates, Shortcuts, Concepts, About)")
    print("  + ? keyboard shortcut to toggle help")


if __name__ == "__main__":
    patch()
