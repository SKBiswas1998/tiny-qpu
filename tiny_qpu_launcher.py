"""
tiny-qpu Interactive Quantum Lab — Native Windows Application

Uses pywebview to create a real Windows desktop window.
No browser opens. Looks and feels like native software.
"""

import sys
import os
import threading

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    sys.path.insert(0, base_dir)
    os.chdir(os.path.dirname(sys.executable))
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(base_dir, 'src'))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='tiny-qpu',
        description='tiny-qpu Interactive Quantum Lab'
    )
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--browser', action='store_true',
                        help='Open in browser instead of native window')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.browser:
        from tiny_qpu.dashboard import launch
        launch(port=args.port, host=args.host, debug=args.debug, open_browser=True)
        return

    # Native window mode (default)
    try:
        import webview
    except ImportError:
        print("pywebview not found - falling back to browser mode.")
        print("For native window: pip install pywebview")
        from tiny_qpu.dashboard import launch
        launch(port=args.port, host=args.host, debug=args.debug, open_browser=True)
        return

    from tiny_qpu.dashboard.server import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"

    def run_server():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    import time, urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)

    window = webview.create_window(
        title='tiny-qpu Quantum Lab',
        url=url,
        width=1400,
        height=900,
        min_size=(1000, 600),
        resizable=True,
        text_select=True,
    )
    webview.start(debug=args.debug)


if __name__ == '__main__':
    main()
