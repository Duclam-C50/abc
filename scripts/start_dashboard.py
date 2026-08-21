"""Start a local web server to view explanations and evaluation results in real-time."""

import os
import sys
import json
import webbrowser
import socket
from http.server import SimpleHTTPRequestHandler, HTTPServer


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve static files and provide dynamic API endpoints."""

    def end_headers(self):
        # Prevent caching so that results update immediately upon refresh
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        # Resolve project root path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # API: Get latest anomalies data
        if self.path == '/api/anomalies':
            js_path = os.path.join(project_root, "artifacts", "results", "ig", "dashboard_data.js")
            if os.path.exists(js_path):
                try:
                    with open(js_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    # Strip 'const ANOMALIES_DATA = ' and trailing ';'
                    if content.startswith("const ANOMALIES_DATA ="):
                        json_str = content[len("const ANOMALIES_DATA ="):].strip()
                        if json_str.endswith(";"):
                            json_str = json_str[:-1].strip()
                        
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(json_str.encode('utf-8'))
                        return
                except Exception as e:
                    self.send_error(500, f"Error reading anomalies: {e}")
                    return
            self.send_error(404, "dashboard_data.js not found. Run generate_dashboard.py first.")
            return

        # API: Get latest evaluation report
        elif self.path == '/api/eval':
            json_path = os.path.join(project_root, "artifacts", "results", "eval_report.json")
            if os.path.exists(json_path):
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    with open(json_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
                except Exception as e:
                    self.send_error(500, f"Error reading eval report: {e}")
                    return
            self.send_error(404, "eval_report.json not found. Run evaluate.py first.")
            return

        # Redirect root to index.html
        elif self.path == '/' or self.path == '/index.html':
            self.path = '/artifacts/results/ig/index.html'

        # Default static file serving
        super().do_GET()


def find_free_port(start_port: int = 8080) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except socket.error:
                port += 1
    return start_port


def main():
    # Set CWD to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    port = find_free_port(8080)
    server_address = ('localhost', port)
    
    print("=" * 60)
    print("  LSTM AUTOENCODER -- REAL-TIME DASHBOARD SERVER")
    print("=" * 60)
    print(f"  Project Root: {project_root}")
    print(f"  Server URL  : http://localhost:{port}/")
    print("=" * 60)
    print("  Press Ctrl+C to stop the server.")
    print()

    httpd = HTTPServer(server_address, DashboardHandler)
    
    # Automatically open browser
    webbrowser.open(f"http://localhost:{port}/")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping server...")
        httpd.server_close()
        print("  Server stopped.")


if __name__ == "__main__":
    main()
