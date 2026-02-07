"""
tiny-qpu Interactive Quantum Lab Dashboard.

Launch with: tiny-qpu serve
Or programmatically: from tiny_qpu.dashboard import launch; launch()
"""

from tiny_qpu.dashboard.server import create_app, launch

__all__ = ["create_app", "launch"]
