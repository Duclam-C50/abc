#!/usr/bin/env python
"""
evtx_to_csv.py
==============
Script chuyển đổi tệp tin nhật ký sự kiện Windows (.evtx) sang định dạng CSV,
trích xuất ĐẦY ĐỦ TẤT CẢ CÁC TRƯỜNG (System Metadata, EventData, UserData).

Tính năng chính:
----------------
1. Trích xuất toàn diện:
   - Toàn bộ System Metadata: EventRecordID, SystemTime, EventID, Level, Task,
     Opcode, Keywords, Channel, Computer, UserID, Execution_ProcessID,
     Execution_ThreadID, Provider_Name, Provider_Guid, ActivityID, v.v.
   - Toàn bộ EventData (đặc thù của từng loại Event ID như Process Creation,
     Network Connection, File Create, Registry, DNS...): Image, CommandLine,
     CurrentDirectory, User, Hashes, ParentProcessId, DestinationIp, QueryName...
   - Toàn bộ UserData (nếu có trong các sự kiện ứng dụng).
   - Tự động map các alias phổ biến (Name -> Provider_Name, ProcessID, ThreadID).
2. Cơ chế 2-Pass Streaming:
   - Pass 1: Quét nhanh qua file để phát hiện tất cả các trường độc nhất xuất hiện
     trong bất kỳ bản ghi nào (vì mỗi loại Event ID có tập trường khác nhau).
   - Pass 2: Ghi dữ liệu ra CSV theo dòng, bộ nhớ O(1), không bị tràn RAM dù file vài GB.
3. Hiệu năng vượt trội:
   - Sử dụng thư viện `evtx` (Rust pyevtx-rs đa luồng), tốc độ hàng chục nghìn
     bản ghi mỗi giây.
4. Hỗ trợ hàng loạt:
   - Chuyển đổi 1 file đơn lẻ hoặc quét toàn bộ thư mục chứa file .evtx.
   - Tùy chọn gộp (merge) nhiều file evtx vào 1 file CSV duy nhất.
5. Mã hóa UTF-8-BOM (utf-8-sig):
   - Đảm bảo tương thích hoàn hảo cả khi đọc bằng Python/Pandas lẫn khi mở trực tiếp
     bằng Microsoft Excel mà không bị lỗi font tiếng Việt hoặc ký tự đặc biệt.

Cách sử dụng:
-------------
# 1. Chuyển đổi 1 file cụ thể:
python scripts/evtx_to_csv.py data/train/log_raw/normal.evtx -o data/train/log_csv/normal.csv

# 2. Tự động lưu cùng tên (thay đuôi .csv):
python scripts/evtx_to_csv.py data/train/log_raw/normal.evtx

# 3. Chuyển đổi toàn bộ file .evtx trong 1 thư mục:
python scripts/evtx_to_csv.py data/train/log_raw/ -o data/train/log_csv/

# 4. Giới hạn số dòng để xem thử (preview):
python scripts/evtx_to_csv.py data/train/log_raw/normal.evtx --limit 100
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Set, Any, Optional, Generator, Tuple

# Reconfigure stdout/stderr for Windows console to handle UTF-8 / Vietnamese characters
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Kiểm tra thư viện evtx
try:
    import evtx
except ImportError:
    print(
        "[LỖI] Thư viện 'evtx' chưa được cài đặt trong môi trường Python!\n"
        "Vui lòng chạy lệnh sau để cài đặt:\n"
        "    pip install evtx\n",
        file=sys.stderr,
    )
    sys.exit(1)


# Thứ tự ưu tiên cho các cột System metadata ở đầu bảng CSV
SYSTEM_COLUMNS_ORDER = [
    "EventRecordID",
    "SystemTime",
    "EventID",
    "Level",
    "Task",
    "Opcode",
    "Keywords",
    "Channel",
    "Computer",
    "UserID",
    "Execution_ProcessID",
    "Execution_ThreadID",
    "Provider_Name",
    "Provider_Guid",
    "ActivityID",
    "RelatedActivityID",
    "Correlation",
    "System_Version",
]

# Thứ tự ưu tiên cho các cột thường gặp trong Sysmon / Windows Security Log
PAYLOAD_COLUMNS_PRIORITY = [
    "RuleName",
    "UtcTime",
    "ProcessGuid",
    "ProcessId",
    "Image",
    "FileVersion",
    "Description",
    "Product",
    "Company",
    "OriginalFileName",
    "CommandLine",
    "CurrentDirectory",
    "User",
    "LogonGuid",
    "LogonId",
    "TerminalSessionId",
    "IntegrityLevel",
    "Hashes",
    "ParentProcessGuid",
    "ParentProcessId",
    "ParentImage",
    "ParentCommandLine",
    "ParentUser",
    "Protocol",
    "Initiated",
    "SourceIsIpv6",
    "SourceIp",
    "SourceHostname",
    "SourcePort",
    "SourcePortName",
    "DestinationIsIpv6",
    "DestinationIp",
    "DestinationHostname",
    "DestinationPort",
    "DestinationPortName",
    "ImageLoaded",
    "Signed",
    "Signature",
    "SignatureStatus",
    "SourceProcessGuid",
    "SourceProcessId",
    "SourceThreadId",
    "SourceImage",
    "TargetProcessGuid",
    "TargetProcessId",
    "TargetImage",
    "GrantedAccess",
    "CallTrace",
    "SourceUser",
    "TargetUser",
    "NewThreadId",
    "StartAddress",
    "StartModule",
    "StartFunction",
    "TargetFilename",
    "IsExecutable",
    "Archived",
    "EventType",
    "TargetObject",
    "Details",
    "PipeName",
    "QueryName",
    "QueryStatus",
    "QueryResults",
    "Name",
    "ProcessID",
    "ThreadID",
]


def extract_record_data(
    data: Dict[str, Any], include_raw_json: bool = False
) -> Dict[str, Any]:
    """
    Trích xuất toàn bộ trường dữ liệu từ một bản ghi EVTX (đã parse sang JSON).
    Đảm bảo không bỏ sót bất kỳ thông tin nào của System, EventData và UserData.
    """
    row: Dict[str, Any] = {}
    event = data.get("Event", {})
    system = event.get("System", {})

    # 1. System Metadata
    row["EventRecordID"] = system.get("EventRecordID", "")

    eid = system.get("EventID", "")
    if isinstance(eid, dict):
        row["EventID"] = eid.get("#text", "")
    else:
        row["EventID"] = eid

    tc = system.get("TimeCreated", {})
    if isinstance(tc, dict):
        row["SystemTime"] = tc.get("#attributes", {}).get("SystemTime", "")
    else:
        row["SystemTime"] = tc or ""

    row["Channel"] = system.get("Channel", "")
    row["Computer"] = system.get("Computer", "")
    row["Level"] = system.get("Level", "")
    row["Task"] = system.get("Task", "")
    row["Opcode"] = system.get("Opcode", "")
    row["Keywords"] = system.get("Keywords", "")
    row["System_Version"] = system.get("Version", "")

    # Provider
    prov = system.get("Provider", {})
    if isinstance(prov, dict):
        pattr = prov.get("#attributes", {})
        row["Provider_Name"] = pattr.get("Name", "")
        row["Provider_Guid"] = pattr.get("Guid", "")
    elif isinstance(prov, str):
        row["Provider_Name"] = prov
        row["Provider_Guid"] = ""

    # Execution (ProcessID và ThreadID của Windows Event Logging Service)
    exec_info = system.get("Execution", {})
    if isinstance(exec_info, dict):
        eattr = exec_info.get("#attributes", {})
        row["Execution_ProcessID"] = eattr.get("ProcessID", "")
        row["Execution_ThreadID"] = eattr.get("ThreadID", "")

    # Security
    sec = system.get("Security", {})
    if isinstance(sec, dict):
        sattr = sec.get("#attributes", {})
        row["UserID"] = sattr.get("UserID", "")

    # Correlation
    corr = system.get("Correlation")
    if isinstance(corr, dict):
        cattr = corr.get("#attributes", {})
        row["ActivityID"] = cattr.get("ActivityID", "")
        row["RelatedActivityID"] = cattr.get("RelatedActivityID", "")
        row["Correlation"] = json.dumps(corr, ensure_ascii=False)
    elif corr:
        row["Correlation"] = str(corr)
        row["ActivityID"] = ""
        row["RelatedActivityID"] = ""
    else:
        row["Correlation"] = ""
        row["ActivityID"] = ""
        row["RelatedActivityID"] = ""

    # 2. EventData (Dynamic Payload)
    ed = event.get("EventData")
    if isinstance(ed, dict):
        for k, v in ed.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
    elif isinstance(ed, list):
        for idx, item in enumerate(ed):
            if isinstance(item, dict):
                name = item.get("#attributes", {}).get("Name")
                val = item.get("#text", item.get("Data", ""))
                if name:
                    row[name] = val
                else:
                    row[f"Data_{idx}"] = json.dumps(item, ensure_ascii=False)
            else:
                row[f"Data_{idx}"] = item

    # 3. UserData (nếu có)
    ud = event.get("UserData")
    if isinstance(ud, dict):
        for k, v in ud.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    col_name = f"{k}_{sub_k}"
                    row[col_name] = (
                        json.dumps(sub_v, ensure_ascii=False)
                        if isinstance(sub_v, (dict, list))
                        else sub_v
                    )
            else:
                row[k] = (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )

    # 4. Compatibility Aliases (hỗ trợ tương thích tối đa với ML / Sysmon models)
    if "Name" not in row or not row["Name"]:
        row["Name"] = row.get("Provider_Name", "")
    if "ProcessID" not in row or row["ProcessID"] == "":
        row["ProcessID"] = row.get("Execution_ProcessID", "")
    if "ThreadID" not in row or row["ThreadID"] == "":
        row["ThreadID"] = row.get("Execution_ThreadID", "")

    # 5. Tùy chọn lưu Raw JSON nguyên bản
    if include_raw_json:
        row["RawJSON"] = json.dumps(data, ensure_ascii=False)

    return row


def discover_columns(
    input_evtx: str,
    number_of_threads: int = 0,
    limit: Optional[int] = None,
    include_raw_json: bool = False,
    quiet: bool = False,
) -> Tuple[List[str], int]:
    """
    Pass 1: Quét toàn bộ file .evtx để phát hiện danh sách TẤT CẢ các cột độc nhất.
    Trả về (danh sách cột đã sắp xếp theo thứ tự tối ưu, tổng số bản ghi).
    """
    parser = evtx.PyEvtxParser(input_evtx, number_of_threads=number_of_threads)
    discovered_keys: Set[str] = set()
    total_records = 0

    if not quiet:
        print(f"[*] Đang quét phát hiện các trường dữ liệu từ '{input_evtx}'...")

    t0 = time.time()
    for record in parser.records_json():
        if isinstance(record, Exception):
            continue

        try:
            record_dict = json.loads(record["data"])
            row = extract_record_data(
                record_dict, include_raw_json=include_raw_json
            )
            discovered_keys.update(row.keys())
            total_records += 1

            if limit and total_records >= limit:
                break
        except Exception:
            continue

    elapsed = time.time() - t0

    # Sắp xếp các cột theo thứ tự trực quan, logic:
    # 1. Các cột System Metadata chuẩn
    # 2. Các cột Payload/Sysmon thường gặp
    # 3. Các cột còn lại (sắp xếp bảng chữ cái)
    ordered_columns: List[str] = []
    seen: Set[str] = set()

    # Nhóm 1
    for col in SYSTEM_COLUMNS_ORDER:
        if col in discovered_keys and col not in seen:
            ordered_columns.append(col)
            seen.add(col)

    # Nhóm 2
    for col in PAYLOAD_COLUMNS_PRIORITY:
        if col in discovered_keys and col not in seen:
            ordered_columns.append(col)
            seen.add(col)

    # Nhóm 3: Các cột còn lại
    remaining = sorted(list(discovered_keys - seen))
    ordered_columns.extend(remaining)

    if not quiet:
        print(
            f"[+] Hoàn tất quét ({elapsed:.2f}s): "
            f"Tìm thấy {len(ordered_columns)} trường dữ liệu độc nhất trong {total_records:,} bản ghi."
        )

    return ordered_columns, total_records


def convert_evtx_to_csv(
    input_evtx: str,
    output_csv: str,
    columns: Optional[List[str]] = None,
    number_of_threads: int = 0,
    limit: Optional[int] = None,
    encoding: str = "utf-8-sig",
    include_raw_json: bool = False,
    sysmon_only: bool = False,
    add_label: Optional[int] = None,
    quiet: bool = False,
) -> int:
    """
    Chuyển đổi một file .evtx sang .csv hoàn chỉnh với đầy đủ các trường.

    Args:
        input_evtx: Đường dẫn tệp .evtx đầu vào.
        output_csv: Đường dẫn tệp .csv đầu ra.
        columns: Danh sách cột cụ thể (nếu None sẽ tự động quét phát hiện ở Pass 1).
        number_of_threads: Số luồng xử lý (0 = tự động theo số core CPU).
        limit: Giới hạn số bản ghi xử lý (dùng để test/preview).
        encoding: Bảng mã file CSV (mặc định utf-8-sig để mở tiếng Việt trên Excel không lỗi).
        include_raw_json: Thêm cột RawJSON chứa toàn bộ JSON gốc của sự kiện.
        sysmon_only: Chỉ giữ lại các sự kiện thuộc kênh Sysmon.
        add_label: Gán thêm cột Label với giá trị cố định.
        quiet: Giảm bớt thông báo tiến trình.

    Returns:
        Số bản ghi đã được ghi thành công.
    """
    if not os.path.exists(input_evtx):
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {input_evtx}")

    # Đảm bảo thư mục đầu ra tồn tại
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pass 1: Quét phát hiện tất cả các cột nếu chưa được chỉ định
    if columns is None:
        columns, expected_count = discover_columns(
            input_evtx=input_evtx,
            number_of_threads=number_of_threads,
            limit=limit,
            include_raw_json=include_raw_json,
            quiet=quiet,
        )
    else:
        expected_count = None

    if not columns:
        if not quiet:
            print(f"[!] Không tìm thấy dữ liệu trong file '{input_evtx}'.")
        return 0

    if add_label is not None and "Label" not in columns:
        columns = list(columns) + ["Label"]

    # Pass 2: Ghi dữ liệu ra CSV
    if not quiet:
        print(f"[*] Đang xuất dữ liệu ra CSV: '{output_csv}'...")

    t0 = time.time()
    parser = evtx.PyEvtxParser(input_evtx, number_of_threads=number_of_threads)
    written_count = 0
    error_count = 0

    with open(output_csv, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, restval="", extrasaction="ignore"
        )
        writer.writeheader()

        for record in parser.records_json():
            if isinstance(record, Exception):
                error_count += 1
                continue

            try:
                record_dict = json.loads(record["data"])
                row = extract_record_data(
                    record_dict, include_raw_json=include_raw_json
                )

                if sysmon_only:
                    prov = str(row.get("Provider_Name", "")).lower()
                    chan = str(row.get("Channel", "")).lower()
                    if "sysmon" not in prov and "sysmon" not in chan:
                        continue

                if add_label is not None:
                    row["Label"] = add_label

                writer.writerow(row)
                written_count += 1

                if not quiet and written_count % 10000 == 0:
                    speed = written_count / (time.time() - t0)
                    print(
                        f"    -> Đã ghi {written_count:,} dòng ({speed:.0f} dòng/giây)..."
                    )

                if limit and written_count >= limit:
                    break
            except Exception as e:
                error_count += 1
                continue

    elapsed = time.time() - t0
    speed = written_count / elapsed if elapsed > 0 else 0
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)

    if not quiet:
        print(
            f"[THÀNH CÔNG] Đã chuyển đổi '{input_evtx}' -> '{output_csv}'\n"
            f"   + Số dòng: {written_count:,} bản ghi\n"
            f"   + Số cột: {len(columns)} trường\n"
            f"   + Thời gian: {elapsed:.2f} giây ({speed:.0f} bản ghi/giây)\n"
            f"   + Kích thước CSV: {file_size_mb:.2f} MB"
        )
        if error_count > 0:
            print(f"   + Bản ghi lỗi/bỏ qua: {error_count}")

    return written_count


def convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    merge: bool = False,
    merged_filename: str = "merged_events.csv",
    number_of_threads: int = 0,
    limit: Optional[int] = None,
    encoding: str = "utf-8-sig",
    include_raw_json: bool = False,
    quiet: bool = False,
) -> None:
    """
    Chuyển đổi toàn bộ các file .evtx trong một thư mục sang .csv.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Thư mục không tồn tại: {input_dir}")

    evtx_files: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".evtx"):
                    evtx_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(".evtx"):
                evtx_files.append(os.path.join(input_dir, file))

    if not evtx_files:
        print(f"[!] Không tìm thấy file .evtx nào trong thư mục '{input_dir}'.")
        return

    print(
        f"[*] Tìm thấy {len(evtx_files)} file .evtx trong '{input_dir}'"
        + (" (đệ quy)" if recursive else "")
    )

    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if merge:
        # Gộp tất cả các file evtx thành 1 CSV duy nhất
        # Pass 1 gộp: Quét tất cả các file để tập hợp siêu tập hợp cột (union of all columns)
        print("[*] Đang quét tất cả các file để thu thập toàn bộ các trường...")
        all_columns_set: Set[str] = set()
        for f_path in evtx_files:
            cols, _ = discover_columns(
                f_path,
                number_of_threads=number_of_threads,
                include_raw_json=include_raw_json,
                quiet=True,
            )
            all_columns_set.update(cols)

        # Sắp xếp các cột
        ordered_columns: List[str] = []
        seen: Set[str] = set()
        for col in SYSTEM_COLUMNS_ORDER + PAYLOAD_COLUMNS_PRIORITY:
            if col in all_columns_set and col not in seen:
                ordered_columns.append(col)
                seen.add(col)
        ordered_columns.extend(sorted(list(all_columns_set - seen)))

        merged_csv_path = os.path.join(output_dir, merged_filename)
        print(
            f"[*] Đang ghi file gộp '{merged_csv_path}' với {len(ordered_columns)} trường..."
        )

        total_merged = 0
        with open(merged_csv_path, "w", newline="", encoding=encoding) as out_f:
            writer = csv.DictWriter(
                out_f,
                fieldnames=ordered_columns,
                restval="",
                extrasaction="ignore",
            )
            writer.writeheader()

            for f_path in evtx_files:
                print(f"    -> Đang xử lý: '{os.path.basename(f_path)}'...")
                parser = evtx.PyEvtxParser(
                    f_path, number_of_threads=number_of_threads
                )
                for record in parser.records_json():
                    if isinstance(record, Exception):
                        continue
                    try:
                        record_dict = json.loads(record["data"])
                        row = extract_record_data(
                            record_dict, include_raw_json=include_raw_json
                        )
                        writer.writerow(row)
                        total_merged += 1
                    except Exception:
                        continue

        print(
            f"[THÀNH CÔNG GỘP] Đã gộp {len(evtx_files)} file evtx vào '{merged_csv_path}' ({total_merged:,} dòng)."
        )
    else:
        # Chuyển đổi từng file riêng rẽ
        for idx, f_path in enumerate(evtx_files, 1):
            base_name = os.path.splitext(os.path.basename(f_path))[0]
            out_csv = os.path.join(output_dir, f"{base_name}.csv")
            print(f"\n--- [{idx}/{len(evtx_files)}] ---")
            convert_evtx_to_csv(
                input_evtx=f_path,
                output_csv=out_csv,
                number_of_threads=number_of_threads,
                limit=limit,
                encoding=encoding,
                include_raw_json=include_raw_json,
                quiet=quiet,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Công cụ chuyển đổi Windows EVTX sang CSV với đầy đủ tất cả các trường.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ lệnh:
  # Chuyển đổi 1 file EVTX:
  python scripts/evtx_to_csv.py data/train/log_raw/normal.evtx -o data/train/log_csv/normal.csv

  # Chuyển đổi toàn bộ thư mục log_raw sang log_csv:
  python scripts/evtx_to_csv.py data/train/log_raw/ -o data/train/log_csv/

  # Gộp tất cả file EVTX trong thư mục thành 1 file CSV:
  python scripts/evtx_to_csv.py data/train/log_raw/ -o data/train/log_csv/ --merge
        """,
    )

    parser.add_argument(
        "input",
        help="Đường dẫn đến file .evtx hoặc thư mục chứa các file .evtx",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Đường dẫn file .csv đầu ra hoặc thư mục lưu file .csv (mặc định cùng thư mục với file gốc)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Quét đệ quy các thư mục con (khi input là thư mục)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Gộp tất cả file .evtx tìm được thành một file CSV duy nhất",
    )
    parser.add_argument(
        "--merged-name",
        default="merged_events.csv",
        help="Tên file CSV gộp khi dùng tùy chọn --merge (mặc định: merged_events.csv)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="Số luồng phân tích cú pháp đa luồng (mặc định: 0 = tự động tận dụng toàn bộ nhân CPU)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Chỉ xuất N bản ghi đầu tiên (dùng để thử nghiệm nhanh)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Bảng mã của file CSV (mặc định: utf-8-sig để hỗ trợ hiển thị đẹp trên Excel và UTF-8)",
    )
    parser.add_argument(
        "--include-raw-json",
        action="store_true",
        help="Bổ sung thêm cột 'RawJSON' chứa toàn bộ cấu trúc JSON gốc của sự kiện",
    )
    parser.add_argument(
        "--sysmon-only",
        action="store_true",
        help="Chỉ giữ lại các sự kiện thuộc Microsoft-Windows-Sysmon",
    )
    parser.add_argument(
        "--add-label",
        type=int,
        default=None,
        help="Gán thêm cột Label với giá trị cố định (ví dụ: 0 cho normal, 1 cho malicious)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Chế độ im lặng (giảm thiểu thông báo in ra terminal)",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    if os.path.isdir(input_path):
        convert_directory(
            input_dir=input_path,
            output_dir=os.path.abspath(args.output) if args.output else None,
            recursive=args.recursive,
            merge=args.merge,
            merged_filename=args.merged_name,
            number_of_threads=args.threads,
            limit=args.limit,
            encoding=args.encoding,
            include_raw_json=args.include_raw_json,
            quiet=args.quiet,
        )
    elif os.path.isfile(input_path):
        if args.output is None:
            base_no_ext = os.path.splitext(input_path)[0]
            output_csv = f"{base_no_ext}.csv"
        else:
            output_arg = os.path.abspath(args.output)
            if os.path.isdir(output_arg) or args.output.endswith(
                (os.sep, "/", "\\")
            ):
                base_name = (
                    os.path.splitext(os.path.basename(input_path))[0] + ".csv"
                )
                output_csv = os.path.join(output_arg, base_name)
            else:
                output_csv = output_arg

        convert_evtx_to_csv(
            input_evtx=input_path,
            output_csv=output_csv,
            number_of_threads=args.threads,
            limit=args.limit,
            encoding=args.encoding,
            include_raw_json=args.include_raw_json,
            sysmon_only=args.sysmon_only,
            add_label=args.add_label,
            quiet=args.quiet,
        )
    else:
        print(f"[LỖI] Đường dẫn không hợp lệ: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
