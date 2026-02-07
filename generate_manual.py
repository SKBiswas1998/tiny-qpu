"""
Generate the tiny-qpu Interactive Quantum Lab User Manual as PDF.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable, Image, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

# ─── Colors ───
VOID = HexColor("#060a14")
DEEP = HexColor("#0a0e1a")
PANEL = HexColor("#0f1424")
CYAN = HexColor("#00d4ff")
PURPLE = HexColor("#845ef7")
AMBER = HexColor("#fab005")
RED = HexColor("#ff6b6b")
GREEN = HexColor("#51cf66")
TEXT_BRIGHT = HexColor("#e8ecf4")
TEXT_PRIMARY = HexColor("#b4bcd0")
TEXT_SEC = HexColor("#6b7394")
BORDER = HexColor("#1e2848")

# Page background
class DarkPageTemplate:
    """Custom page template with dark background and header/footer."""

    def __init__(self, doc):
        self.doc = doc

    def beforePage(self, canvas_obj, doc):
        pass

    @staticmethod
    def onPage(canvas_obj, doc):
        w, h = letter
        # Dark background
        canvas_obj.setFillColor(VOID)
        canvas_obj.rect(0, 0, w, h, fill=1, stroke=0)

        # Header line
        canvas_obj.setStrokeColor(BORDER)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(54, h - 54, w - 54, h - 54)

        # Header text
        canvas_obj.setFont("Courier-Bold", 9)
        canvas_obj.setFillColor(CYAN)
        canvas_obj.drawString(54, h - 48, "tiny-qpu")
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(TEXT_SEC)
        canvas_obj.drawRightString(w - 54, h - 48, "Interactive Quantum Lab — User Manual")

        # Footer
        canvas_obj.setStrokeColor(BORDER)
        canvas_obj.line(54, 42, w - 54, 42)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(TEXT_SEC)
        canvas_obj.drawCentredString(w / 2, 28, f"Page {doc.page}")

    @staticmethod
    def onFirstPage(canvas_obj, doc):
        w, h = letter
        # Dark background
        canvas_obj.setFillColor(VOID)
        canvas_obj.rect(0, 0, w, h, fill=1, stroke=0)

        # Decorative orbit lines
        canvas_obj.setStrokeColor(BORDER)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.ellipse(w/2 - 200, h/2 + 40, w/2 + 200, h/2 + 160)
        canvas_obj.ellipse(w/2 - 180, h/2 + 20, w/2 + 180, h/2 + 180)

        # Central glow dot
        canvas_obj.setFillColor(CYAN)
        canvas_obj.circle(w/2, h/2 + 120, 6, fill=1, stroke=0)


def build_manual(output_path="tiny_qpu_manual.pdf"):
    """Build the complete user manual PDF."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=72,
        bottomMargin=60,
        leftMargin=54,
        rightMargin=54,
    )

    styles = getSampleStyleSheet()

    # ─── Custom Styles ───

    s_cover_title = ParagraphStyle(
        'CoverTitle', parent=styles['Title'],
        fontName='Courier-Bold', fontSize=42, leading=50,
        textColor=CYAN, alignment=TA_CENTER, spaceAfter=8,
    )

    s_cover_sub = ParagraphStyle(
        'CoverSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=14, leading=20,
        textColor=TEXT_SEC, alignment=TA_CENTER, spaceAfter=4,
        spaceBefore=0,
    )

    s_cover_tagline = ParagraphStyle(
        'CoverTagline', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=16,
        textColor=TEXT_PRIMARY, alignment=TA_CENTER, spaceAfter=0,
    )

    s_h1 = ParagraphStyle(
        'DarkH1', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=24, leading=30,
        textColor=CYAN, spaceBefore=24, spaceAfter=12,
    )

    s_h2 = ParagraphStyle(
        'DarkH2', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=16, leading=22,
        textColor=TEXT_BRIGHT, spaceBefore=18, spaceAfter=8,
    )

    s_h3 = ParagraphStyle(
        'DarkH3', parent=styles['Heading3'],
        fontName='Helvetica-Bold', fontSize=13, leading=18,
        textColor=AMBER, spaceBefore=14, spaceAfter=6,
    )

    s_body = ParagraphStyle(
        'DarkBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10.5, leading=15,
        textColor=TEXT_PRIMARY, alignment=TA_JUSTIFY, spaceAfter=8,
    )

    s_code = ParagraphStyle(
        'DarkCode', parent=styles['Code'],
        fontName='Courier', fontSize=9, leading=13,
        textColor=GREEN, backColor=PANEL,
        borderColor=BORDER, borderWidth=1, borderPadding=8,
        spaceBefore=6, spaceAfter=10,
    )

    s_note = ParagraphStyle(
        'DarkNote', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=10, leading=14,
        textColor=AMBER, leftIndent=16, spaceAfter=10,
        borderColor=AMBER, borderWidth=1, borderPadding=(6, 8, 6, 8),
    )

    s_key = ParagraphStyle(
        'KeyStyle', parent=styles['Normal'],
        fontName='Courier-Bold', fontSize=10, leading=14,
        textColor=CYAN,
    )

    s_table_header = ParagraphStyle(
        'TableHeader', fontName='Helvetica-Bold', fontSize=9.5,
        textColor=CYAN, leading=13,
    )

    s_table_cell = ParagraphStyle(
        'TableCell', fontName='Helvetica', fontSize=9.5,
        textColor=TEXT_PRIMARY, leading=13,
    )

    s_toc = ParagraphStyle(
        'TOC', fontName='Helvetica', fontSize=12, leading=20,
        textColor=TEXT_PRIMARY, leftIndent=20,
    )

    # Helper to make styled tables
    def make_table(headers, rows, col_widths=None):
        data = [[Paragraph(h, s_table_header) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), s_table_cell) for c in row])

        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PANEL),
            ('BACKGROUND', (0, 1), (-1, -1), DEEP),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [DEEP, PANEL]),
        ]))
        return t

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=8)

    # ═══════════════════════════════════════════
    # BUILD DOCUMENT
    # ═══════════════════════════════════════════

    story = []

    # ─── COVER PAGE ───
    story.append(Spacer(1, 140))
    story.append(Paragraph("tiny-qpu", s_cover_title))
    story.append(Paragraph("Interactive Quantum Lab", s_cover_sub))
    story.append(Spacer(1, 8))
    story.append(Paragraph("USER MANUAL", ParagraphStyle(
        'CoverLabel', fontName='Helvetica', fontSize=12, leading=16,
        textColor=TEXT_SEC, alignment=TA_CENTER, letterSpacing=6,
    )))
    story.append(Spacer(1, 40))
    story.append(hr())
    story.append(Paragraph(
        "Build quantum circuits, visualize quantum states, and explore "
        "quantum computing — all in your browser.",
        s_cover_tagline))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Version 2.0 — February 2026", ParagraphStyle(
        'Version', fontName='Courier', fontSize=10, textColor=TEXT_SEC,
        alignment=TA_CENTER,
    )))

    story.append(PageBreak())

    # ─── TABLE OF CONTENTS ───
    story.append(Paragraph("Contents", s_h1))
    story.append(Spacer(1, 8))

    toc_items = [
        "1. Getting Started",
        "2. Interface Overview",
        "3. Building Circuits",
        "4. Running Simulations",
        "5. Understanding Results",
        "6. Step-by-Step Mode",
        "7. Preset Circuits",
        "8. OpenQASM Import/Export",
        "9. Keyboard Shortcuts",
        "10. Gate Reference",
        "11. Quantum Concepts Quick Reference",
        "12. Troubleshooting",
    ]
    for item in toc_items:
        story.append(Paragraph(item, s_toc))

    story.append(PageBreak())

    # ─── 1. GETTING STARTED ───
    story.append(Paragraph("1. Getting Started", s_h1))

    story.append(Paragraph("Requirements", s_h2))
    story.append(Paragraph(
        "tiny-qpu requires Python 3.8 or later and Flask. The dashboard runs "
        "entirely locally — no internet connection needed, no cloud accounts, no data leaves your machine.",
        s_body))

    story.append(Paragraph("Installation", s_h2))
    story.append(Paragraph("pip install flask", s_code))
    story.append(Paragraph(
        "If you have the standalone executable (tiny-qpu.exe), no installation is needed. "
        "Simply double-click the file to launch.",
        s_body))

    story.append(Paragraph("Launching the Dashboard", s_h2))
    story.append(Paragraph(
        "From your terminal or PowerShell, navigate to the tiny-qpu directory and run:",
        s_body))
    story.append(Paragraph("python -c \"from tiny_qpu.dashboard import launch; launch()\"", s_code))
    story.append(Paragraph(
        "Or if using the executable:", s_body))
    story.append(Paragraph("tiny-qpu.exe", s_code))
    story.append(Paragraph(
        "Your default browser will open automatically to http://127.0.0.1:8888 showing the "
        "Interactive Quantum Lab dashboard.",
        s_body))

    story.append(PageBreak())

    # ─── 2. INTERFACE OVERVIEW ───
    story.append(Paragraph("2. Interface Overview", s_h1))
    story.append(Paragraph(
        "The dashboard has three panels arranged left to right, plus a top toolbar:",
        s_body))

    story.append(Paragraph("Header Bar", s_h3))
    story.append(Paragraph(
        "The top bar contains the tiny-qpu brand, qubit count selector (1-6 qubits), "
        "and shot count input (number of measurement samples). Changing the qubit count "
        "immediately resizes the circuit canvas.",
        s_body))

    story.append(Paragraph("Left Panel — Gate Palette", s_h3))
    story.append(Paragraph(
        "Lists all available quantum gates organized by category: single-qubit fixed gates "
        "(H, X, Y, Z, S, T), rotation gates (Rx, Ry, Rz, P), multi-qubit gates "
        "(CX, CZ, SWAP, CRz, CCX, CSWAP), and measurement (M). Below the gates, "
        "you will find preset circuits you can load with one click.",
        s_body))

    story.append(Paragraph("Center Panel — Circuit Builder", s_h3))
    story.append(Paragraph(
        "The main workspace. Horizontal lines represent qubit wires labeled q0, q1, etc. "
        "The toolbar above contains Run, Step, Clear, and Undo buttons, plus a status "
        "indicator showing gate count and circuit depth.",
        s_body))

    story.append(Paragraph("Right Panel — Results", s_h3))
    story.append(Paragraph(
        "Displays simulation results in five sections (top to bottom): Bloch Sphere "
        "visualization for each qubit, Probability bars showing the likelihood of each "
        "basis state, Measurement Histogram showing sampled counts, Amplitude Table with "
        "complex amplitudes and phases, and an OpenQASM editor for import/export.",
        s_body))

    story.append(PageBreak())

    # ─── 3. BUILDING CIRCUITS ───
    story.append(Paragraph("3. Building Circuits", s_h1))

    story.append(Paragraph("Placing Gates", s_h2))
    story.append(Paragraph(
        "There are two ways to add gates to your circuit:", s_body))
    story.append(Paragraph(
        "<b>Click method:</b> Click a gate in the left palette to select it (it will be "
        "highlighted with a cyan border). Then click on any qubit wire in the circuit area "
        "to place it. The gate automatically finds the earliest available column.",
        s_body))
    story.append(Paragraph(
        "<b>Drag and drop:</b> Drag a gate from the palette and drop it onto a qubit wire "
        "in the circuit area.",
        s_body))

    story.append(Paragraph("Multi-Qubit Gates", s_h2))
    story.append(Paragraph(
        "When placing a 2-qubit gate (like CX or SWAP), click on the control qubit — "
        "the target is automatically assigned to the adjacent qubit. For 3-qubit gates "
        "(CCX/Toffoli), the controls and target span three consecutive qubits from your "
        "click position.",
        s_body))
    story.append(Paragraph(
        "Visual rendering: CNOT shows a filled dot (control) connected by a vertical line "
        "to a circled-plus (target). CZ shows two filled dots. SWAP shows two X marks.",
        s_body))

    story.append(Paragraph("Rotation Gates", s_h2))
    story.append(Paragraph(
        "When you place a rotation gate (Rx, Ry, Rz, P, CRz), a parameter dialog appears. "
        "Enter the rotation angle in radians, or click one of the quick-select buttons "
        "for common values:",
        s_body))

    story.append(make_table(
        ["Button", "Value", "Radians"],
        [
            ["pi", "Full rotation", "3.14159..."],
            ["pi/2", "Quarter turn", "1.5708..."],
            ["pi/4", "Eighth turn (T gate angle)", "0.7854..."],
            ["pi/3", "Third rotation", "1.0472..."],
            ["-pi/2", "Negative quarter turn", "-1.5708..."],
            ["0", "Identity (no rotation)", "0"],
        ],
        col_widths=[80, 200, 120],
    ))

    story.append(Paragraph("Removing Gates", s_h2))
    story.append(Paragraph(
        "Click any placed gate in the circuit to remove it. The remaining gates will "
        "automatically repack into the earliest available columns. You can also use "
        "the Undo button or Ctrl+Z to remove the most recently placed gate.",
        s_body))

    story.append(PageBreak())

    # ─── 4. RUNNING SIMULATIONS ───
    story.append(Paragraph("4. Running Simulations", s_h1))

    story.append(Paragraph(
        "Click the <b>Run</b> button (or press R) to simulate your circuit. The dashboard "
        "sends the circuit to the tiny-qpu statevector simulator, which computes the exact "
        "quantum state and samples measurement outcomes.",
        s_body))

    story.append(Paragraph("What Happens When You Click Run", s_h2))
    story.append(Paragraph(
        "1. The circuit is converted to gate operations and sent to the backend.<br/>"
        "2. The statevector simulator applies each gate sequentially to the initial |0...0> state.<br/>"
        "3. The final statevector is used to compute probabilities, Bloch coordinates, and amplitudes.<br/>"
        "4. The statevector is sampled N times (where N = Shots) to generate measurement counts.<br/>"
        "5. All results appear in the right panel.",
        s_body))

    story.append(Paragraph("Shots", s_h2))
    story.append(Paragraph(
        "The Shots parameter controls how many times the final quantum state is measured. "
        "More shots give a more accurate histogram but the probabilities (computed from "
        "the statevector) are always exact. Default is 1024. For quick exploration, 100 "
        "shots is fine. For publication-quality histograms, use 8192 or higher.",
        s_body))

    story.append(Paragraph("Simulation Limits", s_h2))
    story.append(Paragraph(
        "The statevector simulator uses 2<super>n</super> complex numbers (16 bytes each) "
        "for n qubits. Memory and time requirements grow exponentially:",
        s_body))

    story.append(make_table(
        ["Qubits", "Statevector Size", "Simulation Time"],
        [
            ["1-4", "< 1 KB", "Instant"],
            ["5-10", "< 16 KB", "< 1 second"],
            ["10-15", "< 512 KB", "< 5 seconds"],
            ["15-20", "< 16 MB", "Seconds to minutes"],
            ["20-25", "< 512 MB", "Minutes"],
        ],
        col_widths=[100, 150, 150],
    ))

    story.append(Paragraph(
        "The dashboard supports up to 6 qubits for interactive use, which runs instantly.",
        s_note))

    story.append(PageBreak())

    # ─── 5. UNDERSTANDING RESULTS ───
    story.append(Paragraph("5. Understanding Results", s_h1))

    story.append(Paragraph("Bloch Sphere", s_h2))
    story.append(Paragraph(
        "Each qubit is visualized on its own Bloch sphere — a 3D representation of a "
        "single qubit state. The north pole represents |0> and the south pole represents |1>. "
        "Points on the equator represent superposition states.",
        s_body))

    story.append(make_table(
        ["State", "Bloch Position", "Coordinates"],
        [
            ["|0>", "North pole", "(0, 0, 1)"],
            ["|1>", "South pole", "(0, 0, -1)"],
            ["|+> = (|0>+|1>)/sqrt(2)", "Equator (positive x)", "(1, 0, 0)"],
            ["|-> = (|0>-|1>)/sqrt(2)", "Equator (negative x)", "(-1, 0, 0)"],
            ["|i> = (|0>+i|1>)/sqrt(2)", "Equator (positive y)", "(0, 1, 0)"],
            ["Mixed/Entangled", "Near origin (shorter vector)", "Purity < 1"],
        ],
        col_widths=[160, 140, 120],
    ))

    story.append(Paragraph(
        "When a qubit is entangled with other qubits, its individual Bloch vector shrinks "
        "toward the origin. An amber ring around the sphere indicates reduced purity (< 1.0), "
        "which means entanglement is present.",
        s_body))

    story.append(Paragraph("Probability Bars", s_h2))
    story.append(Paragraph(
        "Shows the exact probability of measuring each computational basis state, computed "
        "directly from the statevector (not from sampling). States are sorted by probability, "
        "with color coding: cyan for high probability (> 40%), amber for moderate (> 10%), "
        "and gray for low probability states.",
        s_body))

    story.append(Paragraph("Measurement Histogram", s_h2))
    story.append(Paragraph(
        "Shows the distribution of measurement outcomes from sampling the quantum state "
        "N times (controlled by the Shots parameter). Unlike probabilities, counts include "
        "statistical noise — this is what you would see from a real quantum computer. "
        "Each bar shows the count above it, with bars colored by hash for visual distinction.",
        s_body))

    story.append(Paragraph("Amplitude Table", s_h2))
    story.append(Paragraph(
        "Displays the full statevector in Dirac notation. For each basis state |b>, the "
        "table shows the complex amplitude (real + imaginary parts), the phase angle in "
        "degrees, and the probability (|amplitude|<super>2</super>). Zero-amplitude states are shown "
        "as 0 with no phase.",
        s_body))

    story.append(PageBreak())

    # ─── 6. STEP MODE ───
    story.append(Paragraph("6. Step-by-Step Mode", s_h1))
    story.append(Paragraph(
        "Step mode is the most powerful educational feature in tiny-qpu. It lets you watch "
        "the quantum state evolve gate by gate, seeing exactly how each operation transforms "
        "the statevector, probabilities, and Bloch sphere.",
        s_body))

    story.append(Paragraph("Using Step Mode", s_h2))
    story.append(Paragraph(
        "1. Build your circuit (or load a preset).<br/>"
        "2. Click <b>Step</b> (or press S). An amber step indicator appears below the toolbar.<br/>"
        "3. Use <b>Next</b> (or right arrow key) to advance one gate at a time.<br/>"
        "4. Use <b>Prev</b> (or left arrow key) to go backward.<br/>"
        "5. The right panel updates at each step showing the current quantum state.<br/>"
        "6. The current gate is highlighted in the circuit view.<br/>"
        "7. Click <b>Exit</b> (or press Escape) to return to normal mode.",
        s_body))

    story.append(Paragraph(
        "Step 0 always shows the initial state |0...0> before any gates are applied.",
        s_note))

    story.append(Paragraph("What to Watch For", s_h2))
    story.append(Paragraph(
        "<b>Hadamard gate (H):</b> Watch the Bloch vector rotate from the north pole to "
        "the equator, and the probability split from 100%/0% to 50%/50%.",
        s_body))
    story.append(Paragraph(
        "<b>CNOT after H:</b> The target qubit's Bloch vector collapses toward the origin "
        "(entanglement!), and new basis states appear in the probability bars.",
        s_body))
    story.append(Paragraph(
        "<b>Rotation gates:</b> Watch the Bloch vector sweep smoothly. Rx rotates around "
        "the x-axis, Ry around y, Rz around z.",
        s_body))

    story.append(PageBreak())

    # ─── 7. PRESETS ───
    story.append(Paragraph("7. Preset Circuits", s_h1))
    story.append(Paragraph(
        "The left panel (below the gate palette) contains preset circuits that demonstrate "
        "key quantum computing concepts. Click any preset to load and auto-run it.",
        s_body))

    story.append(make_table(
        ["Preset", "Qubits", "Description"],
        [
            ["Bell State (phi+)", "2", "Creates maximally entangled pair |00> + |11>. The foundation of quantum information."],
            ["GHZ State", "3", "Three-qubit entanglement: |000> + |111>. Extends Bell state to 3 parties."],
            ["Uniform Superposition", "3", "H gates on all qubits. Every basis state equally likely."],
            ["Teleportation", "3", "Quantum teleportation protocol. Transfers qubit 0's state to qubit 2."],
            ["Deutsch-Jozsa", "3", "Determines if a function is constant or balanced in one query."],
            ["QFT (2-qubit)", "2", "Quantum Fourier Transform — core subroutine of Shor's algorithm."],
            ["Grover's Search", "2", "Searches for |11> in 2-qubit space with amplitude amplification."],
            ["Bit-Flip Code", "3", "3-qubit repetition code protecting |1> against single bit-flip errors."],
        ],
        col_widths=[110, 50, 280],
    ))

    story.append(PageBreak())

    # ─── 8. QASM ───
    story.append(Paragraph("8. OpenQASM Import/Export", s_h1))

    story.append(Paragraph("Exporting", s_h2))
    story.append(Paragraph(
        "Click <b>Export</b> in the OpenQASM section (bottom of the right panel) to convert "
        "your current circuit into OpenQASM 2.0 format. You can copy this code to use in "
        "Qiskit, IBM Quantum Experience, or any other QASM-compatible tool.",
        s_body))

    story.append(Paragraph("Importing", s_h2))
    story.append(Paragraph(
        "Paste OpenQASM 2.0 code into the text area and click <b>Import</b>. The circuit "
        "builder will parse the QASM and reconstruct the circuit visually. The qubit count "
        "automatically adjusts to match the QASM definition.",
        s_body))

    story.append(Paragraph("Example QASM (Bell State):", s_h3))
    story.append(Paragraph(
        'OPENQASM 2.0;\n'
        'include "qelib1.inc";\n'
        'qreg q[2];\n'
        'h q[0];\n'
        'cx q[0],q[1];',
        s_code))

    story.append(PageBreak())

    # ─── 9. KEYBOARD SHORTCUTS ───
    story.append(Paragraph("9. Keyboard Shortcuts", s_h1))
    story.append(Paragraph(
        "The dashboard supports keyboard shortcuts for fast interaction. "
        "These only work when you are not typing in an input field or text area.",
        s_body))

    story.append(make_table(
        ["Key", "Action"],
        [
            ["R", "Run simulation"],
            ["S", "Enter step mode"],
            ["C", "Clear circuit"],
            ["Ctrl+Z", "Undo last gate"],
            ["H", "Select Hadamard gate"],
            ["X", "Select Pauli-X gate"],
            ["Y", "Select Pauli-Y gate"],
            ["M", "Select Measure"],
            ["Arrow Right", "Next step (in step mode)"],
            ["Arrow Left", "Previous step (in step mode)"],
            ["Escape", "Exit step mode / close dialogs"],
            ["Enter", "Confirm parameter dialog"],
        ],
        col_widths=[120, 300],
    ))

    story.append(PageBreak())

    # ─── 10. GATE REFERENCE ───
    story.append(Paragraph("10. Gate Reference", s_h1))

    story.append(Paragraph("Single-Qubit Fixed Gates", s_h2))
    story.append(make_table(
        ["Gate", "Matrix", "Effect"],
        [
            ["H (Hadamard)", "1/sqrt(2) [[1,1],[1,-1]]", "Creates equal superposition. Maps |0> to |+>, |1> to |->"],
            ["X (Pauli-X)", "[[0,1],[1,0]]", "Bit flip (quantum NOT). Maps |0> to |1>, |1> to |0>"],
            ["Y (Pauli-Y)", "[[0,-i],[i,0]]", "Bit+phase flip. Maps |0> to i|1>, |1> to -i|0>"],
            ["Z (Pauli-Z)", "[[1,0],[0,-1]]", "Phase flip. Maps |1> to -|1>, leaves |0> unchanged"],
            ["S", "[[1,0],[0,i]]", "sqrt(Z). Adds pi/2 phase to |1>"],
            ["T", "[[1,0],[0,e^(i*pi/4)]]", "pi/8 gate. Adds pi/4 phase to |1>"],
            ["sqrt(X)", "(1/2)[[1+i,1-i],[1-i,1+i]]", "Square root of X gate"],
        ],
        col_widths=[90, 160, 190],
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Rotation Gates", s_h2))
    story.append(make_table(
        ["Gate", "Definition", "Generator"],
        [
            ["Rx(theta)", "exp(-i*theta*X/2)", "Pauli X — rotates around x-axis on Bloch sphere"],
            ["Ry(theta)", "exp(-i*theta*Y/2)", "Pauli Y — rotates around y-axis on Bloch sphere"],
            ["Rz(theta)", "exp(-i*theta*Z/2)", "Pauli Z — rotates around z-axis on Bloch sphere"],
            ["P(theta)", "[[1,0],[0,e^(i*theta)]]", "Phase gate — adds relative phase theta to |1>"],
        ],
        col_widths=[90, 160, 190],
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Multi-Qubit Gates", s_h2))
    story.append(make_table(
        ["Gate", "Qubits", "Description"],
        [
            ["CX (CNOT)", "2", "Controlled-X: flips target if control is |1>. Creates entanglement."],
            ["CZ", "2", "Controlled-Z: applies Z to target if control is |1>."],
            ["SWAP", "2", "Exchanges the states of two qubits."],
            ["CRz(theta)", "2", "Controlled Rz: applies Rz(theta) to target if control is |1>."],
            ["CCX (Toffoli)", "3", "Double-controlled X: flips target if both controls are |1>."],
            ["CSWAP (Fredkin)", "3", "Controlled SWAP: swaps targets if control is |1>."],
        ],
        col_widths=[110, 50, 280],
    ))

    story.append(PageBreak())

    # ─── 11. QUANTUM CONCEPTS ───
    story.append(Paragraph("11. Quantum Concepts Quick Reference", s_h1))

    story.append(Paragraph("Superposition", s_h2))
    story.append(Paragraph(
        "A qubit can be in a combination of |0> and |1> simultaneously: |psi> = alpha|0> + beta|1>, "
        "where |alpha|<super>2</super> + |beta|<super>2</super> = 1. The Hadamard gate creates equal "
        "superposition from |0>.",
        s_body))

    story.append(Paragraph("Entanglement", s_h2))
    story.append(Paragraph(
        "Two qubits are entangled when the state of one cannot be described independently "
        "of the other. The Bell state (|00> + |11>)/sqrt(2) is the simplest example — measuring "
        "one qubit instantly determines the other's outcome, regardless of distance.",
        s_body))

    story.append(Paragraph("Measurement", s_h2))
    story.append(Paragraph(
        "Measuring a qubit collapses its superposition into a definite |0> or |1> outcome. "
        "The probability of each outcome equals |amplitude|<super>2</super>. After measurement, the qubit "
        "is no longer in superposition. In the simulator, the statevector gives exact probabilities "
        "while the histogram shows statistical sampling of those probabilities.",
        s_body))

    story.append(Paragraph("Quantum Interference", s_h2))
    story.append(Paragraph(
        "Quantum amplitudes are complex numbers that can add (constructive interference) "
        "or cancel (destructive interference). Algorithms like Grover's Search exploit "
        "interference to amplify the amplitude of correct answers and suppress wrong ones.",
        s_body))

    story.append(Paragraph("No-Cloning Theorem", s_h2))
    story.append(Paragraph(
        "An unknown quantum state cannot be perfectly copied. This is why quantum teleportation "
        "requires destroying the original state, and why quantum key distribution (BB84) is secure.",
        s_body))

    story.append(PageBreak())

    # ─── 12. TROUBLESHOOTING ───
    story.append(Paragraph("12. Troubleshooting", s_h1))

    story.append(make_table(
        ["Problem", "Solution"],
        [
            ["Browser doesn't open", "Navigate manually to http://127.0.0.1:8888"],
            ["'ModuleNotFoundError: flask'", "Run: pip install flask"],
            ["Simulation returns error", "Check that all multi-qubit gates have valid qubit indices"],
            ["Bloch sphere empty", "Click Run to simulate — results don't auto-update yet"],
            ["QASM import fails", "Ensure format is OPENQASM 2.0 with 'qreg q[N];' declaration"],
            ["Port 8888 in use", "Change port: launch(port=9999)"],
            ["Gates won't place", "Make sure a gate is selected (cyan border in palette)"],
        ],
        col_widths=[180, 260],
    ))

    story.append(Spacer(1, 30))
    story.append(hr())
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "tiny-qpu Interactive Quantum Lab is an open-source project.<br/>"
        "GitHub: github.com/SKBiswas1998/tiny-qpu",
        ParagraphStyle('Footer', fontName='Helvetica', fontSize=10,
                       textColor=TEXT_SEC, alignment=TA_CENTER, leading=15)))

    # ─── BUILD ───
    doc.build(story,
              onFirstPage=DarkPageTemplate.onFirstPage,
              onLaterPages=DarkPageTemplate.onPage)

    print(f"Manual generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_manual()
