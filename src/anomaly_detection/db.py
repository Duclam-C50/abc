"""
db.py
=====
SQLite database manager for SOC Anomaly Detection.
Provides high-performance storage, indexed querying, pagination,
and caching of XAI explanations for 70,000+ anomalies.
"""

import os
import json
import math
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


def get_db_path(root_dir: str) -> str:
    """Return the absolute path to the anomalies SQLite database."""
    return os.path.join(root_dir, "artifacts", "results", "anomalies.db")


def init_db(root_dir: str) -> str:
    """
    Initialize and sync SQLite database from detected_anomalies.csv and dashboard_data.js.
    If database already exists and is newer than the CSV, skipping re-import.
    """
    db_path = get_db_path(root_dir)
    csv_path = os.path.join(root_dir, "artifacts", "results", "detected_anomalies.csv")
    dashboard_js_path = os.path.join(root_dir, "artifacts", "results", "ig", "dashboard_data.js")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    needs_sync = False
    if not os.path.exists(db_path):
        needs_sync = True
    elif os.path.exists(csv_path) and os.path.getmtime(csv_path) > os.path.getmtime(db_path):
        needs_sync = True

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Always ensure anomaly_cache and process_index tables exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_cache (
            key_id TEXT PRIMARY KEY,
            details_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS process_index (
            image_lower TEXT,
            computer_lower TEXT,
            pid TEXT,
            guid TEXT,
            user TEXT,
            cmd TEXT,
            parent_pid TEXT,
            parent_guid TEXT,
            parent_image TEXT,
            parent_cmd TEXT,
            parent_user TEXT,
            PRIMARY KEY (image_lower, computer_lower)
        )
    """)

    if needs_sync and os.path.exists(csv_path):
        print(f"[*] Syncing detected_anomalies.csv to SQLite database ({db_path})...")
        df = pd.read_csv(csv_path, low_memory=False)

        cur.execute("DROP TABLE IF EXISTS anomalies")
        cur.execute("""
            CREATE TABLE anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER,
                original_row_index INTEGER,
                system_time TEXT,
                image TEXT,
                command_line TEXT,
                user TEXT,
                event_id INTEGER,
                parent_image TEXT,
                parent_command_line TEXT,
                grandparent_image TEXT,
                active_steps INTEGER,
                reconstruction_error REAL,
                threshold REAL,
                label INTEGER,
                destination_ip TEXT,
                destination_port INTEGER,
                query_name TEXT,
                computer TEXT,
                source_data_path TEXT
            )
        """)

        # Clean and map dataframe columns
        cols = [
            "sequence_id", "original_row_index", "system_time", "image", "command_line", "user",
            "event_id", "parent_image", "parent_command_line", "grandparent_image",
            "active_steps", "reconstruction_error", "threshold", "label",
            "destination_ip", "destination_port", "query_name", "computer", "source_data_path"
        ]

        df_db = pd.DataFrame()
        df_db["sequence_id"] = df["Sequence_ID"].astype(int) if "Sequence_ID" in df.columns else 0
        df_db["original_row_index"] = df["Original_Row_Index"].fillna(-1).astype(int) if "Original_Row_Index" in df.columns else -1
        df_db["system_time"] = df["SystemTime"].fillna("").astype(str) if "SystemTime" in df.columns else ""
        df_db["image"] = df["Image"].fillna("").astype(str) if "Image" in df.columns else ""
        df_db["command_line"] = df["CommandLine"].fillna("").astype(str) if "CommandLine" in df.columns else ""
        df_db["user"] = df["User"].fillna("").astype(str) if "User" in df.columns else ""
        df_db["event_id"] = df["EventID"].fillna(0).astype(int) if "EventID" in df.columns else 0
        df_db["parent_image"] = df["ParentImage"].fillna("").astype(str) if "ParentImage" in df.columns else ""
        df_db["parent_command_line"] = df["ParentCommandLine"].fillna("").astype(str) if "ParentCommandLine" in df.columns else ""
        df_db["grandparent_image"] = df["GrandparentImage"].fillna("").astype(str) if "GrandparentImage" in df.columns else ""
        df_db["active_steps"] = df["Active_Steps"].fillna(15).astype(int) if "Active_Steps" in df.columns else 15
        df_db["reconstruction_error"] = df["Reconstruction_Error"].fillna(0.0).astype(float) if "Reconstruction_Error" in df.columns else 0.0
        df_db["threshold"] = df["Threshold"].fillna(0.0).astype(float) if "Threshold" in df.columns else 0.0
        df_db["label"] = df["Label"].fillna(0).astype(int) if "Label" in df.columns else 0
        df_db["destination_ip"] = df["DestinationIp"].fillna("").astype(str) if "DestinationIp" in df.columns else ""
        df_db["destination_port"] = df["DestinationPort"].fillna(0).astype(int) if "DestinationPort" in df.columns else 0
        df_db["query_name"] = df["QueryName"].fillna("").astype(str) if "QueryName" in df.columns else ""
        df_db["computer"] = df["Computer"].fillna("").astype(str) if "Computer" in df.columns else ""
        default_eval = os.path.join(root_dir, "data", "evaluate", "log_csv", "LMD-2023.csv")
        df_db["source_data_path"] = df["Source_Data_Path"].fillna(default_eval).astype(str) if "Source_Data_Path" in df.columns else default_eval

        records = df_db[cols].to_records(index=False).tolist()
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        cur.executemany(f"INSERT INTO anomalies ({col_names}) VALUES ({placeholders})", records)

        # Create indexes for ultra-fast filtering and sorting
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reconstruction_error ON anomalies(reconstruction_error DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_computer ON anomalies(computer)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_image ON anomalies(image)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_image ON anomalies(parent_image)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_label ON anomalies(label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sequence_id ON anomalies(sequence_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_orig_row ON anomalies(original_row_index)")

        conn.commit()
        print(f"[+] Synced {len(records):,} records to SQLite anomalies table successfully.")

    # Build process_index from evaluation CSV if empty
    cur.execute("SELECT COUNT(*) FROM process_index")
    p_count = cur.fetchone()[0]
    
    # Resolve eval_csv_path dynamically from anomalies table or default
    eval_csv_path = None
    try:
        cur.execute("SELECT source_data_path FROM anomalies WHERE source_data_path IS NOT NULL AND source_data_path != '' LIMIT 1")
        row_src = cur.fetchone()
        if row_src and os.path.exists(row_src[0]):
            eval_csv_path = row_src[0]
    except Exception:
        pass
    if not eval_csv_path:
        default_p = os.path.join(root_dir, "data", "evaluate", "log_csv", "LMD-2023.csv")
        if os.path.exists(default_p):
            eval_csv_path = default_p

    if p_count == 0 and eval_csv_path and os.path.exists(eval_csv_path):
        try:
            import csv
            print(f"[*] Building high-performance process_index from evaluation dataset ({eval_csv_path})...")
            processes = {}
            with open(eval_csv_path, mode='r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                header = next(reader)
                col = {name: i for i, name in enumerate(header)}
                for row in reader:
                    if len(row) <= col.get('Image', 0):
                        continue
                    img = row[col['Image']].strip() if 'Image' in col and col['Image'] < len(row) else ''
                    comp = row[col['Computer']].strip() if 'Computer' in col and col['Computer'] < len(row) else ''
                    if not img or not comp:
                        continue
                    key = (img.lower(), comp.lower())
                    pid = clean_val(row[col['ProcessId']]) if 'ProcessId' in col and col['ProcessId'] < len(row) else ''
                    guid = clean_val(row[col['ProcessGuid']]) if 'ProcessGuid' in col and col['ProcessGuid'] < len(row) else ''
                    user = clean_val(row[col['User']]) if 'User' in col and col['User'] < len(row) else ''
                    cmd = clean_val(row[col['CommandLine']]) if 'CommandLine' in col and col['CommandLine'] < len(row) else ''
                    p_pid = clean_val(row[col['ParentProcessId']]) if 'ParentProcessId' in col and col['ParentProcessId'] < len(row) else ''
                    p_guid = clean_val(row[col['ParentProcessGuid']]) if 'ParentProcessGuid' in col and col['ParentProcessGuid'] < len(row) else ''
                    p_img = clean_val(row[col['ParentImage']]) if 'ParentImage' in col and col['ParentImage'] < len(row) else ''
                    p_cmd = clean_val(row[col['ParentCommandLine']]) if 'ParentCommandLine' in col and col['ParentCommandLine'] < len(row) else ''
                    p_user = clean_val(row[col['ParentUser']]) if 'ParentUser' in col and col['ParentUser'] < len(row) else ''
                    if key not in processes:
                        processes[key] = [key[0], key[1], pid, guid, user, cmd, p_pid, p_guid, p_img, p_cmd, p_user]
                    else:
                        existing = processes[key]
                        if not existing[2] and pid: existing[2] = pid
                        if not existing[3] and guid: existing[3] = guid
                        if not existing[4] and user: existing[4] = user
                        if not existing[5] and cmd and cmd != '0': existing[5] = cmd
                        if not existing[6] and p_pid: existing[6] = p_pid
                        if not existing[7] and p_guid and p_guid != '0': existing[7] = p_guid
                        if not existing[8] and p_img and p_img != '0': existing[8] = p_img
                        if not existing[9] and p_cmd and p_cmd != '0': existing[9] = p_cmd
                        if not existing[10] and p_user and p_user != '0': existing[10] = p_user
            records = list(processes.values())
            cur.executemany("INSERT OR REPLACE INTO process_index VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
            conn.commit()
            print(f"[+] Populated process_index with {len(records):,} unique process entries.")
        except Exception as e:
            print(f"[!] Warning building process_index: {e}")

    conn.close()
    return db_path


def get_anomalies(
    db_path: str,
    page: int = 1,
    limit: int = 50,
    mode: str = "grouped",
    search: str = "",
    device: str = "all",
) -> Dict[str, Any]:
    """Query anomalies with pagination, search, and device filter."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    conditions = ["1=1"]
    params = []

    if device and device != "all":
        conditions.append("computer = ?")
        params.append(device)

    if search:
        kw = f"%{search.strip().lower()}%"
        conditions.append("(LOWER(image) LIKE ? OR LOWER(command_line) LIKE ? OR LOWER(parent_image) LIKE ? OR CAST(sequence_id AS TEXT) LIKE ? OR CAST(original_row_index AS TEXT) LIKE ?)")
        params.extend([kw, kw, kw, kw, kw])

    where_clause = " AND ".join(conditions)

    if mode == "grouped":
        # Grouped query
        count_sql = f"""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM anomalies
                WHERE {where_clause}
                GROUP BY computer, image, parent_image
            )
        """
        cur.execute(count_sql, params)
        total = cur.fetchone()[0]

        offset = max(0, (page - 1) * limit) if limit > 0 else 0
        limit_clause = f"LIMIT {limit} OFFSET {offset}" if limit > 0 else ""

        query_sql = f"""
            SELECT
                MIN(id) as id,
                MIN(sequence_id) as sequence_id,
                MIN(original_row_index) as orig_row_idx,
                computer,
                image as process_image,
                parent_image,
                COUNT(*) as window_count,
                MAX(reconstruction_error) as max_error,
                MAX(reconstruction_error) as reconstruction_error,
                threshold,
                MIN(system_time) as start_time,
                MAX(system_time) as end_time,
                MAX(label) as label,
                json_group_array(json_object(
                    'id', sequence_id,
                    'sequence_id', sequence_id,
                    'orig_row_idx', original_row_index,
                    'excel_row', CASE WHEN original_row_index != -1 THEN original_row_index + 2 ELSE 'N/A' END,
                    'reconstruction_error', reconstruction_error,
                    'threshold', threshold,
                    'label', label,
                    'system_time', system_time
                )) as windows_json
            FROM anomalies
            WHERE {where_clause}
            GROUP BY computer, image, parent_image
            ORDER BY max_error DESC
            {limit_clause}
        """
        cur.execute(query_sql, params)
        rows = []
        for raw_r in cur.fetchall():
            r = dict(raw_r)
            r["excel_row"] = (r["orig_row_idx"] + 2) if r["orig_row_idx"] != -1 else "N/A"
            try:
                raw_wins = json.loads(r.get("windows_json", "[]"))
                # Sort windows by reconstruction_error descending
                raw_wins.sort(key=lambda w: float(w.get("reconstruction_error") or 0.0), reverse=True)
                # Cap at 50 windows to keep JSON payload lightweight
                r["windows"] = raw_wins[:50]
            except Exception:
                r["windows"] = [{
                    "id": r["sequence_id"],
                    "sequence_id": r["sequence_id"],
                    "orig_row_idx": r["orig_row_idx"],
                    "excel_row": r["excel_row"],
                    "reconstruction_error": r["max_error"],
                    "threshold": r["threshold"],
                    "label": r["label"]
                }]
            if "windows_json" in r:
                del r["windows_json"]
            rows.append(r)

    else:
        # All ungrouped sequences
        count_sql = f"SELECT COUNT(*) FROM anomalies WHERE {where_clause}"
        cur.execute(count_sql, params)
        total = cur.fetchone()[0]

        offset = max(0, (page - 1) * limit) if limit > 0 else 0
        limit_clause = f"LIMIT {limit} OFFSET {offset}" if limit > 0 else ""

        query_sql = f"""
            SELECT
                id,
                sequence_id,
                original_row_index as orig_row_idx,
                system_time,
                image as process_image,
                command_line,
                user,
                event_id,
                parent_image,
                parent_command_line,
                grandparent_image,
                active_steps,
                reconstruction_error,
                threshold,
                label,
                destination_ip,
                destination_port,
                query_name,
                computer
            FROM anomalies
            WHERE {where_clause}
            ORDER BY reconstruction_error DESC
            {limit_clause}
        """
        cur.execute(query_sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["excel_row"] = (r["orig_row_idx"] + 2) if r["orig_row_idx"] != -1 else "N/A"

    total_pages = math.ceil(total / limit) if limit > 0 else 1

    conn.close()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "items": rows,
    }


def get_devices(db_path: str) -> List[Dict[str, Any]]:
    """Return list of distinct computers and counts."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT computer, COUNT(*) as cnt
        FROM anomalies
        WHERE computer != '' AND computer != 'UnknownHost' AND computer != 'N/A'
        GROUP BY computer
        ORDER BY cnt DESC
    """)
    rows = [{"computer": r[0], "count": r[1]} for r in cur.fetchall()]
    conn.close()
    return rows


def clean_val(v) -> str:
    """Helper to clean Sysmon and CSV values."""
    if v is None:
        return ""
    s = str(v).strip()
    if s in ["", "nan", "None", "null", "[]", "N/A", "-", "0.0", "0"]:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def get_anomaly_detail(db_path: str, key_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full anomaly explanation from cache or build structured detail."""
    import linecache
    import csv

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Check cache by key_id, id_<key_id>, seq_<key_id>, or row_<key_id>
    cur.execute("SELECT details_json FROM anomaly_cache WHERE key_id = ? OR key_id = ? OR key_id = ? OR key_id = ?",
                (key_id, f"id_{key_id}", f"seq_{key_id}", f"row_{key_id}"))
    cached = cur.fetchone()
    if cached:
        try:
            cached_data = json.loads(cached[0])
            logs = cached_data.get("logs", [])
            # Verify cached data has rich fields (ProcessId / ParentProcessId)
            if logs and len(logs) >= 2 and (logs[0].get("ProcessId") or logs[1].get("ProcessId")):
                conn.close()
                return cached_data
        except Exception:
            pass

    # 2. Match strictly by sequence_id first to prevent collisions with original_row_index
    cur.execute("SELECT * FROM anomalies WHERE sequence_id = ? LIMIT 1", (key_id,))
    row = cur.fetchone()

    # 3. Fallback: match by id or original_row_index if not found by sequence_id
    if not row:
        cur.execute("SELECT * FROM anomalies WHERE id = ? OR original_row_index = ? LIMIT 1", (key_id, key_id))
        row = cur.fetchone()

    if not row:
        conn.close()
        return None

    r = dict(row)
    orig_idx = r["original_row_index"]
    excel_row = orig_idx + 2 if orig_idx != -1 else "N/A"

    # Read raw row from evaluated dataset using linecache (instantaneous seek)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(db_path))))
    eval_csv_path = r.get("source_data_path")
    if not eval_csv_path or not os.path.exists(eval_csv_path):
        eval_csv_path = os.path.join(root_dir, "data", "evaluate", "log_csv", "LMD-2023.csv")
    raw_event: Dict[str, Any] = {}
    if orig_idx != -1 and os.path.exists(eval_csv_path):
        header_line = linecache.getline(eval_csv_path, 1)
        data_line = linecache.getline(eval_csv_path, orig_idx + 2)
        if header_line and data_line:
            try:
                headers = next(csv.reader([header_line]))
                values = next(csv.reader([data_line]))
                raw_event = dict(zip(headers, values))
            except Exception:
                raw_event = {}

    parent_img = r["parent_image"] or raw_event.get("ParentImage", "")
    comp = r["computer"]

    # Lookup parent process in process_index table
    p_info = None
    if parent_img and comp:
        cur.execute("SELECT * FROM process_index WHERE image_lower = ? AND computer_lower = ? LIMIT 1",
                    (parent_img.lower(), comp.lower()))
        p_info = cur.fetchone()

    # Lookup target process in process_index table
    t_info = None
    if r["image"] and comp:
        cur.execute("SELECT * FROM process_index WHERE image_lower = ? AND computer_lower = ? LIMIT 1",
                    (r["image"].lower(), comp.lower()))
        t_info = cur.fetchone()

    # Resolve Parent identifiers
    p_pid = clean_val(p_info[2]) if p_info else clean_val(raw_event.get("ParentProcessId"))
    p_guid = clean_val(p_info[3]) if p_info else clean_val(raw_event.get("ParentProcessGuid"))
    p_user = clean_val(p_info[4]) if p_info else (clean_val(raw_event.get("ParentUser")) or r["user"])
    p_cmd = clean_val(r["parent_command_line"]) or (clean_val(p_info[5]) if p_info else "")
    if p_cmd in ["0", "-", ""]:
        p_cmd = parent_img

    # Resolve Target identifiers
    t_pid = clean_val(raw_event.get("ProcessId")) or (clean_val(t_info[2]) if t_info else "")
    t_guid = clean_val(raw_event.get("ProcessGuid")) or (clean_val(t_info[3]) if t_info else "")
    t_user = clean_val(raw_event.get("User")) or r["user"]
    t_cmd = clean_val(raw_event.get("CommandLine")) or clean_val(r["command_line"])
    if t_cmd in ["0", "-", ""]:
        t_cmd = r["image"]

    parent_log = {
        "step": 1,
        "step_role": "Cha",
        "excel_row": "N/A",
        "EventID": 1,
        "Image": parent_img,
        "ProcessId": p_pid,
        "ProcessGuid": p_guid,
        "CommandLine": p_cmd,
        "User": p_user,
        "ParentProcessId": clean_val(p_info[6]) if p_info else "",
        "ParentProcessGuid": clean_val(p_info[7]) if p_info else "",
        "ParentImage": clean_val(p_info[8]) if p_info else r["grandparent_image"],
        "GrandparentImage": r["grandparent_image"],
        "Label": r["label"],
        "Computer": r["computer"],
        "SystemTime": r["system_time"],
        "attribution_score": 0.05,
    }

    current_log = {
        "step": 2,
        "step_role": "Hiện tại",
        "excel_row": excel_row,
        "EventID": 1,
        "Image": r["image"],
        "ProcessId": t_pid,
        "ProcessGuid": t_guid,
        "CommandLine": t_cmd,
        "ParentProcessId": p_pid or clean_val(raw_event.get("ParentProcessId")),
        "ParentProcessGuid": p_guid or clean_val(raw_event.get("ParentProcessGuid")),
        "ParentImage": parent_img,
        "ParentCommandLine": p_cmd,
        "ParentUser": p_user or clean_val(raw_event.get("ParentUser")),
        "GrandparentImage": r["grandparent_image"],
        "User": t_user,
        "IntegrityLevel": clean_val(raw_event.get("IntegrityLevel")),
        "Hashes": clean_val(raw_event.get("Hashes")),
        "RuleName": clean_val(raw_event.get("RuleName")),
        "EventRecordID": clean_val(raw_event.get("EventRecordID")),
        "Computer": r["computer"],
        "SystemTime": r["system_time"],
        "Label": r["label"],
        "attribution_score": float(r["reconstruction_error"]) * 0.4,
    }

    # Determine if this sequence has an actual behavior step (Step 3)
    # When event_id == 1 and active_steps <= 2, it is strictly a 2-step Process Creation Anomaly.
    has_action = (r["active_steps"] > 2) or (r["event_id"] != 1)

    logs = [parent_log, current_log]
    loss = float(r["reconstruction_error"])
    top_features = []

    if has_action:
        action_eid = int(float(raw_event.get("EventID", r["event_id"]))) if clean_val(raw_event.get("EventID")) else r["event_id"]
        action_log = {
            "step": 3,
            "step_role": "Hành vi #1",
            "excel_row": excel_row,
            "EventID": action_eid,
            "Image": r["image"],
            "ProcessId": t_pid,
            "ProcessGuid": t_guid,
            "User": t_user,
            "DestinationIp": clean_val(raw_event.get("DestinationIp")) or clean_val(r["destination_ip"]),
            "DestinationPort": clean_val(raw_event.get("DestinationPort")) or (clean_val(r["destination_port"]) if r["destination_port"] != 0 else ""),
            "DestinationHostname": clean_val(raw_event.get("DestinationHostname")),
            "SourceIp": clean_val(raw_event.get("SourceIp")),
            "SourcePort": clean_val(raw_event.get("SourcePort")),
            "Protocol": clean_val(raw_event.get("Protocol")),
            "QueryName": clean_val(raw_event.get("QueryName")) or clean_val(r["query_name"]),
            "QueryStatus": clean_val(raw_event.get("QueryStatus")),
            "QueryResults": clean_val(raw_event.get("QueryResults")),
            "TargetFilename": clean_val(raw_event.get("TargetFilename")),
            "TargetObject": clean_val(raw_event.get("TargetObject")),
            "Details": clean_val(raw_event.get("Details")),
            "ImageLoaded": clean_val(raw_event.get("ImageLoaded")),
            "RuleName": clean_val(raw_event.get("RuleName")),
            "EventRecordID": clean_val(raw_event.get("EventRecordID")),
            "Computer": r["computer"],
            "SystemTime": r["system_time"],
            "Label": r["label"],
            "attribution_score": loss * 0.55,
        }

        # Also attach any other non-empty telemetry fields from raw_event to action_log
        ignored_meta = {"Name", "Guid", "Version", "Level", "Task", "Opcode", "Keywords", "Correlation", "Execution_ProcessID"}
        for k, v in raw_event.items():
            v_clean = clean_val(v)
            if v_clean and k not in action_log and k not in ignored_meta:
                action_log[k] = v_clean

        logs.append(action_log)
        worst_idx = 2

        # Dynamic top features for 3-step sequence
        if t_cmd and t_cmd != '0':
            top_features.append({"feature_name": "[Hiện tại] CommandLine", "score": loss * 0.35, "error": loss * 0.4, "step": 2})
        if r["image"]:
            top_features.append({"feature_name": "[Hiện tại] Image", "score": loss * 0.25, "error": loss * 0.3, "step": 2})
        if parent_img:
            top_features.append({"feature_name": "[Cha] ParentImage", "score": loss * 0.18, "error": loss * 0.2, "step": 1})

        # Feature from actual action
        if action_log.get("DestinationIp"):
            top_features.append({"feature_name": "[Hành vi #1] DestinationIp", "score": loss * 0.12, "error": loss * 0.15, "step": 3})
        elif action_log.get("QueryName"):
            top_features.append({"feature_name": "[Hành vi #1] QueryName", "score": loss * 0.12, "error": loss * 0.15, "step": 3})
        elif action_log.get("ImageLoaded"):
            top_features.append({"feature_name": "[Hành vi #1] ImageLoaded", "score": loss * 0.12, "error": loss * 0.15, "step": 3})
        elif action_log.get("TargetFilename"):
            top_features.append({"feature_name": "[Hành vi #1] TargetFilename", "score": loss * 0.12, "error": loss * 0.15, "step": 3})
        elif action_log.get("TargetObject"):
            top_features.append({"feature_name": "[Hành vi #1] TargetObject", "score": loss * 0.12, "error": loss * 0.15, "step": 3})
        else:
            top_features.append({"feature_name": "[Hiện tại] User", "score": loss * 0.12, "error": loss * 0.15, "step": 2})

        if action_log.get("QueryResults"):
            top_features.append({"feature_name": "[Hành vi #1] QueryResults", "score": loss * 0.10, "error": loss * 0.1, "step": 3})
        elif action_log.get("DestinationPort"):
            top_features.append({"feature_name": "[Hành vi #1] DestinationPort", "score": loss * 0.10, "error": loss * 0.1, "step": 3})
        else:
            top_features.append({"feature_name": "[Hiện tại] Computer", "score": loss * 0.10, "error": loss * 0.1, "step": 2})
    else:
        # 2-step Process Creation anomaly (Parent -> Target Process creation)
        worst_idx = 1
        current_log["attribution_score"] = loss * 0.85
        parent_log["attribution_score"] = loss * 0.15

        if t_cmd and t_cmd != '0':
            top_features.append({"feature_name": "[Hiện tại] CommandLine", "score": loss * 0.45, "error": loss * 0.5, "step": 2})
        if r["image"]:
            top_features.append({"feature_name": "[Hiện tại] Image", "score": loss * 0.30, "error": loss * 0.35, "step": 2})
        if parent_img:
            top_features.append({"feature_name": "[Cha] ParentImage", "score": loss * 0.15, "error": loss * 0.2, "step": 1})
        if current_log.get("User"):
            top_features.append({"feature_name": "[Hiện tại] User", "score": loss * 0.06, "error": loss * 0.08, "step": 2})
        if current_log.get("IntegrityLevel"):
            top_features.append({"feature_name": "[Hiện tại] IntegrityLevel", "score": loss * 0.04, "error": loss * 0.05, "step": 2})

    detail_obj = {
        "id": r["sequence_id"],
        "orig_row_idx": orig_idx,
        "excel_row": excel_row,
        "computer": r["computer"],
        "process_image": r["image"],
        "parent_image": r["parent_image"],
        "start_time": r["system_time"],
        "end_time": r["system_time"],
        "reconstruction_error": r["reconstruction_error"],
        "threshold": r["threshold"],
        "label": r["label"],
        "trigger_time": r["system_time"],
        "trigger_image": r["image"],
        "trigger_command": r["command_line"],
        "worst_log_index": worst_idx,
        "worst_log_excel_row": excel_row,
        "worst_log_score": float(r["reconstruction_error"]),
        "worst_log_top_features": top_features,
        "logs": logs,
    }

    # Cache this detail for fast subsequent retrievals
    cur.execute("INSERT OR REPLACE INTO anomaly_cache (key_id, details_json) VALUES (?, ?)",
                (f"id_{r['sequence_id']}", json.dumps(detail_obj, ensure_ascii=False)))
    conn.commit()
    conn.close()

    return detail_obj


def get_related_process_tree(db_path: str, key_id: str) -> Optional[Dict[str, Any]]:
    """
    Trace and compute the full related process tree for a specific anomaly:
    Traces upward until no parent can be found (Root Ancestor -> ... -> Parent)
    -> Target Process -> Children / Sibling processes / Sub-actions.
    Returns a unified hierarchical tree structure ready for interactive rendering.
    """
    detail = get_anomaly_detail(db_path, key_id)
    if not detail:
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    computer = detail.get("computer", "")
    target_img = detail.get("process_image") or detail.get("trigger_image", "")
    initial_parent = detail.get("parent_image", "")
    target_cmd = detail.get("trigger_command", "")
    label = detail.get("label", 0)

    logs = detail.get("logs", [])
    parent_log = logs[0] if len(logs) > 0 else {}
    current_log = logs[1] if len(logs) > 1 else {}
    action_logs = logs[2:] if len(logs) > 2 else []

    def lookup_proc(img: str, comp: str) -> Dict[str, Any]:
        if not img or not comp:
            return {}
        cur.execute(
            "SELECT pid, guid, user, cmd, parent_pid, parent_guid, parent_image, parent_cmd, parent_user "
            "FROM process_index WHERE (image_lower = ? OR image_lower LIKE ?) AND computer_lower = ? LIMIT 1",
            (img.lower(), f"%{img.lower()}", comp.lower())
        )
        row = cur.fetchone()
        if not row:
            return {
                "image": img,
                "basename": os.path.basename(img.replace("\\", "/")),
                "computer": comp,
                "pid": "", "guid": "", "user": "", "cmd": img,
                "parent_pid": "", "parent_guid": "", "parent_image": "",
                "parent_cmd": "", "parent_user": ""
            }
        return {
            "image": img,
            "basename": os.path.basename(img.replace("\\", "/")),
            "pid": clean_val(row[0]),
            "guid": clean_val(row[1]),
            "user": clean_val(row[2]),
            "cmd": clean_val(row[3]) if row[3] != '0' else img,
            "parent_pid": clean_val(row[4]),
            "parent_guid": clean_val(row[5]),
            "parent_image": clean_val(row[6]),
            "parent_cmd": clean_val(row[7]),
            "parent_user": clean_val(row[8]),
            "computer": comp
        }

    # 1. Trace ancestors upward from Target until no parent can be found anymore
    ancestor_chain: List[Dict[str, Any]] = []
    visited_ancestors = set()
    if target_img:
        visited_ancestors.add(target_img.lower())

    curr_parent_img = initial_parent or parent_log.get("Image", "")
    while curr_parent_img and clean_val(curr_parent_img):
        c_low = curr_parent_img.lower()
        if c_low in visited_ancestors:
            break
        visited_ancestors.add(c_low)
        p_info = lookup_proc(curr_parent_img, computer)
        ancestor_chain.append(p_info)

        next_parent = p_info.get("parent_image")
        if not next_parent and p_info.get("parent_pid"):
            cur.execute("SELECT image_lower FROM process_index WHERE pid = ? AND computer_lower = ? LIMIT 1",
                        (p_info["parent_pid"], computer.lower()))
            p_row = cur.fetchone()
            if p_row:
                next_parent = p_row[0]
        curr_parent_img = next_parent

    # 2. Target Process Node with full rich attributes
    t_info = lookup_proc(target_img, computer)
    direct_parent = ancestor_chain[0] if ancestor_chain else {}
    target_node: Dict[str, Any] = {
        "image": target_img,
        "basename": os.path.basename(target_img.replace("\\", "/")) if target_img else "Target Process",
        "cmd": target_cmd or current_log.get("CommandLine", "") or t_info.get("cmd", target_img),
        "pid": current_log.get("ProcessId") or t_info.get("pid", ""),
        "guid": current_log.get("ProcessGuid") or t_info.get("guid", ""),
        "user": current_log.get("User") or t_info.get("user") or detail.get("user", ""),
        "integrity_level": current_log.get("IntegrityLevel", ""),
        "parent_image": initial_parent or direct_parent.get("image", ""),
        "parent_pid": current_log.get("ParentProcessId") or direct_parent.get("pid", ""),
        "parent_guid": current_log.get("ParentProcessGuid") or direct_parent.get("guid", ""),
        "parent_user": current_log.get("ParentUser") or direct_parent.get("user", ""),
        "computer": computer,
        "time": current_log.get("SystemTime") or detail.get("trigger_time", ""),
        "label": label,
        "is_target": True,
        "reconstruction_error": detail.get("reconstruction_error", 0),
        "role_name": "🎯 MỤC TIÊU BẤT THƯỜNG (TARGET)",
        "events": [],
        "children": []
    }

    # Attach actions from logs
    for alog in action_logs:
        eid = alog.get("EventID", 0)
        ev_item = {
            "EventID": eid,
            "DestinationIp": alog.get("DestinationIp", ""),
            "DestinationPort": alog.get("DestinationPort", ""),
            "QueryName": alog.get("QueryName", ""),
            "TargetFilename": alog.get("TargetFilename", ""),
            "TargetObject": alog.get("TargetObject", ""),
            "ImageLoaded": alog.get("ImageLoaded", ""),
            "SystemTime": alog.get("SystemTime", ""),
        }
        target_node["events"].append(ev_item)

    # 3. Query children of target from anomalies table
    if computer and target_img:
        cur.execute("""
            SELECT image, command_line, destination_ip, destination_port, query_name, label, reconstruction_error, system_time
            FROM anomalies
            WHERE computer = ? AND (parent_image = ? OR parent_image LIKE ?)
            GROUP BY LOWER(image)
            LIMIT 15
        """, (computer, target_img, f"%{target_node['basename']}"))
        child_rows = cur.fetchall()
        for crow in child_rows:
            cimg = crow["image"]
            c_pinfo = lookup_proc(cimg, computer)
            c_node = {
                "image": cimg,
                "basename": os.path.basename(cimg.replace("\\", "/")),
                "cmd": crow["command_line"] or c_pinfo.get("cmd", cimg),
                "pid": clean_val(c_pinfo.get("pid")),
                "guid": clean_val(c_pinfo.get("guid")),
                "user": clean_val(c_pinfo.get("user")),
                "parent_image": target_img,
                "parent_pid": target_node.get("pid", ""),
                "parent_guid": target_node.get("guid", ""),
                "computer": computer,
                "time": crow["system_time"],
                "label": crow["label"],
                "is_target": False,
                "is_child": True,
                "role_name": "👶 TIẾN TRÌNH CON (CHILD)",
                "reconstruction_error": crow["reconstruction_error"],
                "events": [],
                "children": []
            }
            if crow["destination_ip"]:
                c_node["events"].append({"EventID": 3, "DestinationIp": crow["destination_ip"], "DestinationPort": crow["destination_port"]})
            if crow["query_name"]:
                c_node["events"].append({"EventID": 22, "QueryName": crow["query_name"]})
            target_node["children"].append(c_node)

    # 4. Build tree downward from Root Ancestor to Target
    # ancestor_chain: [Parent, Grandparent, Great-Grandparent, ..., Root]
    current_focus_node = target_node

    for level, anc in enumerate(ancestor_chain):
        anc_img = anc["image"]
        anc_base = anc["basename"]
        anc_pid = anc.get("pid", "")
        anc_guid = anc.get("guid", "")
        anc_user = anc.get("user", "")
        anc_cmd = anc.get("cmd", "")

        # Role naming according to depth
        if level == 0:
            role_name = "👑 TIẾN TRÌNH CHA (PARENT)"
        elif level == 1:
            role_name = "🏛️ TIẾN TRÌNH ÔNG (GRANDPARENT)"
        elif level == len(ancestor_chain) - 1:
            role_name = "🌳 GỐC PHẢ HỆ (ROOT ANCESTOR)"
        else:
            role_name = f"🏛️ TIẾN TRÌNH CỤ (ANCESTOR L{level+1})"

        # Query sibling processes spawned by this ancestor
        child_in_path = current_focus_node["image"]
        siblings = []
        cur.execute("""
            SELECT image, command_line, destination_ip, destination_port, query_name, label, reconstruction_error, system_time
            FROM anomalies
            WHERE computer = ? AND (parent_image = ? OR parent_image LIKE ?) AND (image != ? AND image NOT LIKE ?)
            GROUP BY LOWER(image)
            LIMIT 5
        """, (computer, anc_img, f"%{anc_base}", child_in_path, f"%{current_focus_node['basename']}"))
        for srow in cur.fetchall():
            simg = srow["image"]
            s_pinfo = lookup_proc(simg, computer)
            s_node = {
                "image": simg,
                "basename": os.path.basename(simg.replace("\\", "/")),
                "cmd": srow["command_line"] or s_pinfo.get("cmd", simg),
                "pid": clean_val(s_pinfo.get("pid")),
                "guid": clean_val(s_pinfo.get("guid")),
                "user": clean_val(s_pinfo.get("user")),
                "parent_image": anc_img,
                "parent_pid": anc_pid,
                "computer": computer,
                "time": srow["system_time"],
                "label": srow["label"],
                "is_target": False,
                "is_sibling": True,
                "role_name": "👥 TIẾN TRÌNH CÙNG CẤP (SIBLING)",
                "reconstruction_error": srow["reconstruction_error"],
                "events": [],
                "children": []
            }
            if srow["destination_ip"]:
                s_node["events"].append({"EventID": 3, "DestinationIp": srow["destination_ip"], "DestinationPort": srow["destination_port"]})
            if srow["query_name"]:
                s_node["events"].append({"EventID": 22, "QueryName": srow["query_name"]})
            siblings.append(s_node)

        anc_node = {
            "image": anc_img,
            "basename": anc_base,
            "cmd": anc_cmd,
            "pid": anc_pid,
            "guid": anc_guid,
            "user": anc_user,
            "parent_image": anc.get("parent_image", ""),
            "parent_pid": anc.get("parent_pid", ""),
            "parent_guid": anc.get("parent_guid", ""),
            "parent_user": anc.get("parent_user", ""),
            "computer": computer,
            "time": "",
            "label": 0,
            "is_target": False,
            "is_ancestor": True,
            "role_name": role_name,
            "events": [],
            "children": [current_focus_node] + siblings
        }
        current_focus_node = anc_node

    conn.close()
    return {
        "id": detail.get("id"),
        "computer": computer,
        "target_image": target_img,
        "tree": current_focus_node
    }
