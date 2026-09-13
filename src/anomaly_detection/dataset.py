"""
dataset.py
==========
Utilities for constructing process-tree hierarchical sequences and sliding windows.
Supports 15-step process-action sequences:
  - Step 0: Parent process event (Cha)
  - Step 1: Current process creation event (Hiện tại)
  - Steps 2..14: Direct actions of current process (Hành vi: network, registry, file, dns...)
"""

import os
import csv
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def clean_id(val: Any) -> str:
    """Clean GUID / PID string, removing whitespace and trailing .0."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ["nan", "none", "", "0", "0.0", "-"]:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def clean_str(val: Any) -> str:
    """Clean string, removing whitespace and empty placeholders."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ["nan", "none", "", "0", "0.0", "-"]:
        return ""
    return s


class ProcessNode:
    """Represents a process execution node in the process tree."""

    def __init__(
        self,
        guid: str,
        pid: str,
        image: str,
        cmd: str,
        parent_guid: str,
        parent_pid: str,
        user: str,
        time_str: str,
        row_idx: int = -1,
        label: int = 0,
        parent_image: str = "",
        parent_cmd: str = "",
        computer: str = "",
    ):
        self.guid = clean_id(guid)
        self.pid = clean_id(pid)
        self.computer = clean_str(computer)
        self.image = str(image) if image and str(image).lower() != "nan" else ""
        self.cmd = str(cmd) if cmd and str(cmd).lower() != "nan" else ""
        self.parent_guid = clean_id(parent_guid)
        self.parent_pid = clean_id(parent_pid)
        self.parent_image = str(parent_image) if parent_image and str(parent_image).lower() != "nan" else ""
        self.parent_cmd = str(parent_cmd) if parent_cmd and str(parent_cmd).lower() != "nan" else ""
        self.user = str(user) if user and str(user).lower() != "nan" else ""
        self.time_str = str(time_str) if time_str and str(time_str).lower() != "nan" else ""
        self.row_idx = row_idx
        self.label = label
        self.depth = 1
        self.children: List["ProcessNode"] = []
        self.events: List[Dict[str, Any]] = []
        self.parent_node: Optional["ProcessNode"] = None

    @property
    def basename(self) -> str:
        if not self.image:
            return "UnknownProcess"
        return os.path.basename(self.image.replace("\\", "/"))


def load_records_from_csv(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["_row_idx"] = idx
            records.append(row)
    return records


def parse_event(d: Dict[str, Any]) -> Dict[str, Any]:
    """Parse raw CSV or EVTX record into standardized dictionary."""
    eid_raw = d.get("EventID", 0)
    try:
        eid = int(float(str(eid_raw).strip()))
    except (ValueError, TypeError):
        eid = 0

    image = d.get("Image") or d.get("SourceImage") or ""
    p_image = d.get("ParentImage") or ""
    cmd = d.get("CommandLine") or ""
    guid = d.get("ProcessGuid") or ""
    p_guid = d.get("ParentProcessGuid") or ""
    pid = d.get("ProcessId") or d.get("ProcessID") or ""
    p_pid = d.get("ParentProcessId") or d.get("ParentProcessID") or ""
    user = d.get("User") or d.get("AccountName") or ""
    time_str = d.get("SystemTime") or d.get("@timestamp") or d.get("UtcTime") or ""
    target_file = d.get("TargetFilename") or ""
    dest_ip = d.get("DestinationIp") or d.get("DestAddress") or ""
    dest_port_raw = d.get("DestinationPort") or d.get("DestPort") or 0
    try:
        dest_port = int(float(str(dest_port_raw).strip()))
    except (ValueError, TypeError):
        dest_port = 0
    query_name = d.get("QueryName") or ""
    protocol = d.get("Protocol") or ""
    dest_host = d.get("DestinationHostname") or d.get("DestinationHostName") or ""
    target_obj = d.get("TargetObject") or ""
    img_loaded = d.get("ImageLoaded") or ""
    signed = d.get("Signed") or ""
    signature = d.get("Signature") or ""
    sig_status = d.get("SignatureStatus") or ""
    src_image = d.get("SourceImage") or ""
    tgt_image = d.get("TargetImage") or ""
    query_status_raw = d.get("QueryStatus") or 0
    try:
        query_status = int(float(str(query_status_raw).strip()))
    except (ValueError, TypeError):
        query_status = 0
    p_cmd = d.get("ParentCommandLine") or ""
    computer = clean_str(d.get("Computer") or d.get("ComputerName") or "")
    lbl_raw = d.get("Label", 0)
    try:
        label = int(float(str(lbl_raw).strip()))
    except (ValueError, TypeError):
        label = 0

    # Base standardized dictionary
    res = {
        "EventID": eid,
        "Image": str(image),
        "ParentImage": str(p_image),
        "CommandLine": str(cmd),
        "ParentCommandLine": str(p_cmd),
        "ProcessGuid": str(guid),
        "ParentProcessGuid": str(p_guid),
        "ProcessId": str(pid),
        "ParentProcessId": str(p_pid),
        "User": str(user),
        "Computer": computer,
        "SystemTime": str(time_str),
        "TargetFilename": str(target_file),
        "DestinationIp": str(dest_ip),
        "DestinationPort": dest_port,
        "Protocol": str(protocol),
        "DestinationHostname": str(dest_host),
        "TargetObject": str(target_obj),
        "ImageLoaded": str(img_loaded),
        "Signed": str(signed),
        "Signature": str(signature),
        "SignatureStatus": str(sig_status),
        "SourceImage": str(src_image),
        "TargetImage": str(tgt_image),
        "QueryName": str(query_name),
        "QueryStatus": query_status,
        "Label": label,
        "_row_idx": d.get("_row_idx", -1),
    }

    # Retain all other rich forensic attributes from raw data (e.g. Hashes, IntegrityLevel, ParentUser, CallTrace...)
    ignored_raw = {"Event", "_row_idx", "Label", "EventID"}
    for k, v in d.items():
        if k not in res and k not in ignored_raw:
            v_str = str(v).strip() if v is not None else ""
            if v_str and v_str.lower() not in ("nan", "none", "0", "0.0", "-"):
                res[k] = v_str

    return res


def build_process_trees(parsed_records: List[Dict[str, Any]]) -> List[ProcessNode]:
    """Build process hierarchy trees from parsed records, strictly grouped by (Computer, ProcessGuid)."""
    nodes_by_key: Dict[Tuple[str, str], ProcessNode] = {}
    all_nodes: List[ProcessNode] = []

    # Pass 1: Create ProcessNode for each Process Creation event (EventID == 1)
    for r in parsed_records:
        eid = r["EventID"]
        guid = clean_id(r["ProcessGuid"])
        comp = clean_str(r.get("Computer", ""))

        if eid == 1 or not nodes_by_key:
            node = ProcessNode(
                guid=guid,
                pid=clean_id(r["ProcessId"]),
                image=r["Image"],
                cmd=r["CommandLine"],
                parent_guid=r["ParentProcessGuid"],
                parent_pid=r["ParentProcessId"],
                user=r["User"],
                time_str=r["SystemTime"],
                row_idx=r["_row_idx"],
                label=r["Label"],
                parent_image=r.get("ParentImage", ""),
                parent_cmd=r.get("ParentCommandLine", ""),
                computer=comp,
            )
            if guid:
                nodes_by_key[(comp, guid)] = node
            all_nodes.append(node)

    # Pass 2: Attach non-EID 1 events strictly by (Computer, ProcessGuid)
    for r in parsed_records:
        eid = r["EventID"]
        if eid == 1:
            continue

        guid = clean_id(r["ProcessGuid"])
        comp = clean_str(r.get("Computer", ""))
        key = (comp, guid)

        if guid and key in nodes_by_key:
            node = nodes_by_key[key]
            node.events.append(r)
            if r["Label"] > node.label:
                node.label = r["Label"]

    # Pass 3: Link Parent -> Child relationships strictly within the same Computer
    root_nodes: List[ProcessNode] = []

    for node in all_nodes:
        parent_node = None
        if node.parent_guid:
            parent_key = (node.computer, clean_id(node.parent_guid))
            if parent_key in nodes_by_key:
                parent_node = nodes_by_key[parent_key]

        if parent_node and parent_node != node:
            node.parent_node = parent_node
            parent_node.children.append(node)
        else:
            root_nodes.append(node)

    return root_nodes


def assign_tree_depth(node: ProcessNode, current_depth: int = 1):
    """Assign process tree depth (Root = 1, Child = Parent + 1)."""
    node.depth = current_depth
    for child in node.children:
        assign_tree_depth(child, current_depth + 1)


def build_parent_dict(parent_node: Optional[ProcessNode], node: ProcessNode) -> Dict[str, Any]:
    if parent_node:
        p_img = parent_node.image or parent_node.parent_image or ""
        p_cmd = parent_node.cmd or parent_node.parent_cmd or ""
        lbl = parent_node.label
        time_s = parent_node.time_str
        row_i = parent_node.row_idx
        guid = parent_node.guid
        pid = parent_node.pid
        user = parent_node.user
        comp = parent_node.computer
    else:
        p_img = node.parent_image or node.parent_guid or ""
        p_cmd = node.parent_cmd or ""
        lbl = 0
        time_s = node.time_str
        row_i = -1
        guid = node.parent_guid
        pid = node.parent_pid
        user = node.user
        comp = node.computer

    return {
        "EventID": 1 if p_img else 0,
        "Image": p_img,
        "CommandLine": p_cmd,
        "User": user,
        "Computer": comp,
        "ParentImage": "",
        "ParentCommandLine": "",
        "Label": lbl,
        "SystemTime": time_s,
        "_row_idx": row_i,
        "ProcessGuid": guid,
        "ProcessId": pid,
        "ParentProcessGuid": "",
        "ParentProcessId": "",
        "TargetFilename": "",
        "DestinationIp": "",
        "DestinationPort": 0,
        "Protocol": "",
        "DestinationHostname": "",
        "TargetObject": "",
        "ImageLoaded": "",
        "Signed": "",
        "Signature": "",
        "SignatureStatus": "",
        "SourceImage": "",
        "TargetImage": "",
        "QueryName": "",
        "QueryStatus": 0,
    }


def build_current_dict(node: ProcessNode, parent_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "EventID": 1,
        "Image": node.image,
        "CommandLine": node.cmd,
        "User": node.user,
        "Computer": node.computer,
        "ParentImage": parent_dict.get("Image", ""),
        "ParentCommandLine": parent_dict.get("CommandLine", ""),
        "Label": node.label,
        "SystemTime": node.time_str,
        "_row_idx": node.row_idx,
        "ProcessGuid": node.guid,
        "ProcessId": node.pid,
        "ParentProcessGuid": node.parent_guid,
        "ParentProcessId": node.parent_pid,
        "TargetFilename": "",
        "DestinationIp": "",
        "DestinationPort": 0,
        "Protocol": "",
        "DestinationHostname": "",
        "TargetObject": "",
        "ImageLoaded": "",
        "Signed": "",
        "Signature": "",
        "SignatureStatus": "",
        "SourceImage": "",
        "TargetImage": "",
        "QueryName": "",
        "QueryStatus": 0,
    }


def generate_tree_sequences(roots: List[ProcessNode], stride: int = 1) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[Dict[str, Any]]]:
    """
    Construct 15-step process-action sequences:
      - Step 0 (idx 0): Parent process event (Cha)
      - Step 1 (idx 1): Current process creation event (Hiện tại - Event 1)
      - Steps 2..14 (13 steps): Direct actions of the current process (network, registry, file, dns...)
        (excluding child process creations; child processes have their own sequences).
      - If n_actions <= 13: 1 sequence with padding up to 15.
      - If n_actions > 13: sliding window of 13 actions with stride.
    """
    sequences = []
    labels = []
    metadatas = []

    def traverse(parent_node: Optional[ProcessNode], curr_node: ProcessNode):
        p_dict = build_parent_dict(parent_node, curr_node)
        c_dict = build_current_dict(curr_node, p_dict)

        # Only direct non-EID 1 events of curr_node
        actions = sorted(curr_node.events, key=lambda x: str(x.get("SystemTime", "")))
        n_act = len(actions)

        if n_act <= 13:
            seq_rows = [p_dict, c_dict] + actions
            lbl = max([curr_node.label] + [a.get("Label", 0) for a in actions]) if actions else curr_node.label
            sequences.append(seq_rows)
            labels.append(lbl)
            metadatas.append({
                "parent": p_dict,
                "current": c_dict,
                "active_steps": len(seq_rows),
                "depth": curr_node.depth,
                "trigger_event": actions[-1] if actions else c_dict,
            })
        else:
            for s_idx in range(0, n_act - 12, stride):
                win = actions[s_idx : s_idx + 13]
                seq_rows = [p_dict, c_dict] + win
                lbl = max([curr_node.label] + [a.get("Label", 0) for a in win])
                sequences.append(seq_rows)
                labels.append(lbl)
                metadatas.append({
                    "parent": p_dict,
                    "current": c_dict,
                    "active_steps": 15,
                    "depth": curr_node.depth,
                    "trigger_event": win[-1],
                })
            if (n_act - 13) % stride != 0:
                win = actions[-13:]
                seq_rows = [p_dict, c_dict] + win
                lbl = max([curr_node.label] + [a.get("Label", 0) for a in win])
                sequences.append(seq_rows)
                labels.append(lbl)
                metadatas.append({
                    "parent": p_dict,
                    "current": c_dict,
                    "active_steps": 15,
                    "depth": curr_node.depth,
                    "trigger_event": win[-1],
                })

        # Recurse into children (each child becomes curr_node, and this curr_node becomes parent_node)
        for child in curr_node.children:
            traverse(curr_node, child)

    for r in roots:
        traverse(None, r)

    return sequences, np.array(labels, dtype=np.int32), metadatas


def build_tree_dataset(data_source: Any, stride: int = 1) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[Dict[str, Any]]]:
    """Load records from CSV or DataFrame and build 15-step process-action sequences."""
    if isinstance(data_source, str):
        raw_records = load_records_from_csv(data_source)
    elif isinstance(data_source, pd.DataFrame):
        raw_records = []
        for idx, row in data_source.iterrows():
            d = row.to_dict()
            d["_row_idx"] = idx
            raw_records.append(d)
    else:
        raw_records = list(data_source)

    parsed_records = [parse_event(r) for r in raw_records]
    roots = build_process_trees(parsed_records)
    for r in roots:
        assign_tree_depth(r, 1)

    return generate_tree_sequences(roots, stride=stride)


def vectorize_tree_sequences(raw_seqs: List[List[Dict[str, Any]]], preprocessor: Any, fit: bool = False, seq_len: int = 15) -> np.ndarray:
    """
    Vectorize 15-step process-action sequences into an array of shape (N, seq_len, n_features).
    Padded steps (length < seq_len) are strictly np.zeros(n_features) for clean masking.
    """
    all_events = [ev for seq in raw_seqs for ev in seq]
    df_events = pd.DataFrame(all_events)

    if fit:
        X_events = preprocessor.fit_transform(df_events)
    else:
        X_events = preprocessor.transform(df_events)

    n_features = X_events.shape[1]
    n_seqs = len(raw_seqs)
    X_3d = np.zeros((n_seqs, seq_len, n_features), dtype=np.float32)

    offset = 0
    for i, seq in enumerate(raw_seqs):
        k = min(len(seq), seq_len)
        X_3d[i, :k, :] = X_events[offset : offset + k]
        offset += len(seq)

    return X_3d


# ---------------------------------------------------------------------------
# Legacy sliding window & lineage helpers (retained for backward compatibility)
# ---------------------------------------------------------------------------
def make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    if len(data) <= seq_len:
        raise ValueError(f"data length ({len(data)}) must be greater than seq_len ({seq_len}).")
    windows = sliding_window_view(data, window_shape=(seq_len, data.shape[1]))
    return windows[:-1, 0, :, :]


def make_seq_labels(labels: np.ndarray, seq_len: int) -> np.ndarray:
    windows = sliding_window_view(labels, seq_len)
    return np.any(windows[:-1] != 0, axis=1).astype(np.int32)


def make_lineage_sequences(df: pd.DataFrame, X_numeric: np.ndarray) -> np.ndarray:
    n_samples, n_features = X_numeric.shape
    guid_to_idx = {}
    if 'ProcessGuid' in df.columns and 'EventID' in df.columns:
        event_ids = df['EventID'].values
        process_guids = df['ProcessGuid'].values
        for idx in range(n_samples):
            if event_ids[idx] == 1:
                guid = process_guids[idx]
                if pd.notna(guid) and guid != '0' and guid != '-':
                    guid_to_idx[guid] = idx

    blank_vector = np.zeros(n_features)
    sequences = np.zeros((n_samples, 3, n_features))
    parent_guids = df['ParentProcessGuid'].values if 'ParentProcessGuid' in df.columns else [None] * n_samples

    for idx in range(n_samples):
        sequences[idx, 2, :] = X_numeric[idx]
        parent_guid = parent_guids[idx]
        parent_idx = -1
        if pd.notna(parent_guid) and parent_guid != '0' and parent_guid != '-':
            parent_idx = guid_to_idx.get(parent_guid, -1)

        if parent_idx != -1:
            sequences[idx, 1, :] = X_numeric[parent_idx]
            gp_guid = parent_guids[parent_idx]
            gp_idx = -1
            if pd.notna(gp_guid) and gp_guid != '0' and gp_guid != '-':
                gp_idx = guid_to_idx.get(gp_guid, -1)
            if gp_idx != -1:
                sequences[idx, 0, :] = X_numeric[gp_idx]
            else:
                sequences[idx, 0, :] = blank_vector
        else:
            sequences[idx, 1, :] = blank_vector
            sequences[idx, 0, :] = blank_vector

    return sequences


def make_lineage_seq_labels(labels: np.ndarray) -> np.ndarray:
    return (labels != 0).astype(np.int32)
