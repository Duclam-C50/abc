"""Start a local web server to view explanations and evaluation results in real-time."""

import os
import sys
import json
import webbrowser
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


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
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if os.path.join(project_root, "src") not in sys.path:
            sys.path.insert(0, os.path.join(project_root, "src"))

        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.path)
        clean_path = parsed_url.path
        query = parse_qs(parsed_url.query)

        from anomaly_detection.db import init_db, get_db_path, get_anomalies, get_devices, get_anomaly_detail, get_related_process_tree

        # API: Get latest anomalies data with SQLite database pagination & search
        if clean_path == '/api/anomalies':
            try:
                page = int(query.get('page', ['1'])[0])
                limit = int(query.get('limit', ['50'])[0])
                mode = query.get('mode', ['grouped'])[0]
                search = query.get('search', [''])[0]
                device = query.get('device', ['all'])[0]

                db_path = get_db_path(project_root)
                if not os.path.exists(db_path):
                    db_path = init_db(project_root)

                res = get_anomalies(db_path, page=page, limit=limit, mode=mode, search=search, device=device)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f"Error querying anomalies database: {e}")
                return

        # API: Get anomaly detail and XAI explanation (cached or on-demand)
        elif clean_path == '/api/anomaly_detail':
            try:
                key_id = query.get('id', [None])[0] or query.get('row', [None])[0]
                if not key_id:
                    self.send_error(400, "Missing id or row query parameter")
                    return

                db_path = get_db_path(project_root)
                if not os.path.exists(db_path):
                    db_path = init_db(project_root)

                detail = get_anomaly_detail(db_path, str(key_id))
                if not detail:
                    self.send_error(404, f"Anomaly not found: {key_id}")
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(detail, ensure_ascii=False).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f"Error fetching anomaly detail: {e}")
                return

        # API: Trace full related process tree for a specific anomaly
        elif clean_path == '/api/related_tree':
            try:
                key_id = query.get('id', [None])[0] or query.get('row', [None])[0]
                if not key_id:
                    self.send_error(400, "Missing id parameter")
                    return

                db_path = get_db_path(project_root)
                tree_data = get_related_process_tree(db_path, str(key_id))
                if not tree_data:
                    self.send_error(404, f"Related tree not found for: {key_id}")
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(tree_data, ensure_ascii=False).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f"Error calculating related process tree: {e}")
                return

        # API: Get distinct devices and counts from database
        elif clean_path == '/api/devices':
            try:
                db_path = get_db_path(project_root)
                if not os.path.exists(db_path):
                    db_path = init_db(project_root)

                devs = get_devices(db_path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(devs, ensure_ascii=False).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f"Error fetching devices: {e}")
                return

        # API: Get latest evaluation report
        elif clean_path == '/api/eval':
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



        # Redirect root to index.html
        elif clean_path == '/' or clean_path == '/index.html':
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

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if os.path.join(project_root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(project_root, "src"))

    from anomaly_detection.db import init_db
    print("[*] Initializing SQLite database...")
    db_path = init_db(project_root)
    print(f"[+] SQLite database ready: {db_path}")
    
    port = find_free_port(8080)
    server_address = ('localhost', port)
    
    print("=" * 60)
    print("  LSTM AUTOENCODER -- REAL-TIME DASHBOARD SERVER (DATABASE-BACKED)")
    print("=" * 60)
    print(f"  Project Root: {project_root}")
    print(f"  Database    : {db_path}")
    print(f"  Server URL  : http://localhost:{port}/")
    print("=" * 60)
    print("  Press Ctrl+C to stop the server.")
    print()

    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    
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
