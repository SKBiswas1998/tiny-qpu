"""
Quantum Manim Animations
===========================
Beautiful animated visualizations for quantum computing concepts.

Requires: pip install manim

Scenes:
    BlochSphereScene     — Animate qubit rotations on the Bloch sphere
    CircuitAnimScene     — Step-by-step circuit execution
    EntanglementScene    — Bell state creation and measurement
    VQEOptimScene        — VQE energy landscape optimization
    NoiseDegradScene     — Noise degradation visualization
    PESAnimScene         — Animated potential energy surface

Usage:
    # Render from command line:
    manim -pql src/tiny_qpu/animations/scenes.py BlochSphereScene

    # Or from Python:
    from tiny_qpu.animations import render_scene
    render_scene('BlochSphereScene', quality='low')
"""
