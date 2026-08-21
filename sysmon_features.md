# Tài liệu các trường dữ liệu Sysmon sử dụng trong huấn luyện Machine Learning

Dự án sử dụng mô hình học máy **LSTM Autoencoder** (Phát hiện bất thường mạng/hệ thống) để phân tích chuỗi sự kiện Sysmon. Dữ liệu huấn luyện thô được lấy từ sự kiện Windows Sysmon Event Log sau khi chuyển đổi sang dạng CSV. 

Mô hình xử lý tổng cộng **71 trường dữ liệu** (sau khi đã loại bỏ 23 trường gây nhiễu hoặc mã định danh ngẫu nhiên).

---

## 1. Các trường dữ liệu được giữ lại để huấn luyện (71 trường)

Các trường được giữ lại được phân nhóm theo mục đích giám sát của Sysmon:

### A. Siêu dữ liệu sự kiện chung (Common Event Metadata)
* `EventID`: ID sự kiện Sysmon (Ví dụ: `1` cho tạo tiến trình, `3` cho kết nối mạng, `11` cho tạo tệp tin, v.v.). Đây là trường quan trọng nhất để mô hình nhận diện loại hành vi.
* `Version`, `Level`, `Task`: Các metadata kỹ thuật của bản ghi Windows Event Log.
* `Computer`: Tên máy tính xảy ra sự kiện (giúp học các mô hình hành vi đặc trưng của từng máy).
* `UserID`: Mã định danh bảo mật (SID) của người dùng thực hiện hành vi.

### B. Giám sát Tiến trình (Event ID 1: Process Creation & Event ID 5: Process Terminated)
Các trường này cung cấp ngữ cảnh đầy đủ về tiến trình đang chạy và tiến trình cha sinh ra nó:
* `ProcessId`, `ProcessID`, `ThreadID`: Định danh số hiệu tiến trình và luồng hệ thống.
* `Image`: Đường dẫn đầy đủ của tệp thực thi (Ví dụ: `C:\Windows\System32\cmd.exe`).
* `FileVersion`, `Description`, `Product`, `Company`, `OriginalFileName`: Thông tin chi tiết của tệp thực thi được trích xuất từ phần Header của file PE (giúp phát hiện mã độc giả mạo tên tệp tin hệ thống).
* `CommandLine`: Dòng lệnh đầy đủ được sử dụng để chạy tiến trình (Ví dụ: lệnh thực thi Powershell tải mã độc).
* `CurrentDirectory`: Thư mục làm việc hiện tại của tiến trình.
* `User`: Tên tài khoản chạy tiến trình.
* `LogonId`: ID của phiên đăng nhập (Session Logon ID).
* `TerminalSessionId`: ID phiên Terminal/RDP.
* `IntegrityLevel`: Mức độ phân quyền bảo mật của tiến trình (`System`, `High`, `Medium`, `Low`).
* `ParentProcessId`: PID của tiến trình cha.
* `ParentImage`: Đường dẫn tệp thực thi của tiến trình cha.
* `ParentCommandLine`: Dòng lệnh chạy tiến trình cha.
* `ParentUser`: Người dùng chạy tiến trình cha.

### C. Giám sát Kết nối Mạng (Event ID 3: Network Connection)
Sử dụng khi tiến trình tạo ra bất kỳ kết nối mạng TCP/UDP nào:
* `Protocol`: Giao thức truyền tải (`TCP`, `UDP`).
* `Initiated`: Xác định tiến trình nội bộ có chủ động kết nối ra ngoài không (`true`/`false`).
* `SourceIsIpv6`, `SourceIp`, `SourceHostname`, `SourcePort`, `SourcePortName`: Thông tin máy gửi (thường là localhost).
* `DestinationIsIpv6`, `DestinationIp`, `DestinationHostname`, `DestinationPort`, `DestinationPortName`: Địa chỉ IP, tên miền và cổng của máy đích kết nối tới (giúp phát hiện kết nối tới máy chủ điều khiển C2).
* `State`: Trạng thái kết nối mạng.

### D. Giám sát Nạp Thư viện DLL (Event ID 7: Image Loaded)
* `ImageLoaded`: Đường dẫn của thư viện DLL được nạp vào tiến trình.
* `Signed`, `Signature`, `SignatureStatus`: Xác minh chữ ký số của thư viện DLL (giúp phát hiện kỹ thuật DLL Hijacking hoặc nạp DLL không có chữ ký hợp lệ).

### E. Giám sát Tạo Luồng từ xa (Event ID 8: CreateRemoteThread)
Hành vi tạo luồng trong tiến trình khác (thường dùng cho kỹ thuật Process Injection):
* `NewThreadId`: ID của luồng mới được tạo.
* `StartAddress`, `StartModule`, `StartFunction`: Địa chỉ bắt đầu, mô-đun và tên hàm của luồng mới (giúp phát hiện việc thực thi mã lạ ngoài vùng thư viện chuẩn).

### F. Giám sát Truy cập Tiến trình (Event ID 10: Process Access)
Phát hiện việc một tiến trình mở tiến trình khác để đọc/ghi bộ nhớ (Ví dụ: Mimikatz truy cập bộ nhớ `lsass.exe` để trích xuất mật khẩu):
* `SourceProcessId`, `SourceThreadId`, `SourceImage`: Tiến trình yêu cầu truy cập.
* `TargetProcessId`, `TargetImage`: Tiến trình bị truy cập.
* `GrantedAccess`: Quyền hạn truy cập được cấp (Ví dụ: `0x1F1FFF`).
* `CallTrace`: Ngữ cảnh các hàm hệ thống được gọi dẫn đến hành vi này (rất hữu ích cho phân tích hành vi nâng cao).
* `SourceUser`, `TargetUser`: Người dùng của hai tiến trình liên quan.

### G. Giám sát Tệp tin (Event ID 11: FileCreate, Event ID 23: FileDelete, Event ID 26: FileDeleteDetected)
* `TargetFilename`: Đường dẫn tệp tin được tạo hoặc xóa trên đĩa cứng.
* `IsExecutable`, `Archived`: Trạng thái tệp tin có khả năng thực thi hay không.

### H. Giám sát Registry (Event ID 12: RegistryEvent Object Create/Delete, Event ID 13: Value Set)
Giám sát việc sửa đổi các khóa hệ thống (ví dụ: thêm mã độc vào khóa Autorun để khởi động cùng Windows):
* `EventType`: Loại thao tác (`CreateKey`, `DeleteKey`, `SetValue`...).
* `TargetObject`: Đường dẫn khóa Registry bị tác động (Ví dụ: `HKLM\Software\Microsoft\Windows\CurrentVersion\Run\Malware`).
* `Details`: Dữ liệu cụ thể được ghi vào khóa Registry đó.

### I. Giám sát Named Pipes (Event ID 17: Pipe Created, Event ID 18: Pipe Connected)
* `PipeName`: Tên đường ống Named Pipe (nhiều mã độc sử dụng các tên Named Pipe mặc định để giao tiếp nội bộ giữa các thành phần độc hại).

### J. Giám sát DNS (Event ID 22: DNS Query)
* `QueryName`, `QueryStatus`, `QueryResults`: Tên miền được phân giải, trạng thái kết quả và địa chỉ IP trả về (giúp phát hiện kỹ thuật DGA - Domain Generation Algorithm).

### K. Các trường khác
* `Contents`: Nội dung thô đi kèm của một số sự kiện.
* `Version2`, `SchemaVersion`, `Configuration`, `ConfigurationFileHash`: Liên quan đến trạng thái cấu hình của chính Sysmon.

---

## 2. Các trường bị LOẠI BỎ (COLUMNS_TO_DROP trong config.py)

Trong quá trình tiền xử lý dữ liệu trước khi đưa vào mô hình học máy, các trường sau đã bị loại bỏ vì các lý do cụ thể:

| Tên trường bị loại bỏ | Lý do loại bỏ |
| :--- | :--- |
| `ID` | Chỉ là số thứ tự dòng hoặc định danh cơ sở dữ liệu tạm thời, không chứa thông tin đặc trưng hành vi. |
| `EventRecordID` | Số thứ tự tăng dần của Windows Event Log, mang tính đặc thù của máy thu thập log, không mang ý nghĩa bảo mật. |
| `ProcessGuid`, `ParentProcessGuid`, `SourceProcessGUID`, `TargetProcessGUID`, `SourceProcessGuid`, `TargetProcessGuid`, `LogonGuid`, `Guid` | Các chuỗi mã định danh toàn cầu duy nhất (UUID/GUID) được sinh ngẫu nhiên. Nếu giữ lại, mô hình sẽ bị **quá khớp (overfitting)** vì học thuộc lòng các chuỗi ngẫu nhiên này thay vì học hành vi tổng quát. |
| `Name`, `Opcode`, `Keywords`, `Correlation`, `Execution_ProcessID`, `Channel` | Metadata của hệ thống Windows Event Log để phân loại danh mục, không bổ sung giá trị hành vi cho tiến trình. |
| `SystemTime`, `UtcTime`, `CreationUtcTime`, `PreviousCreationUtcTime` | Các nhãn thời gian (timestamp) tuyệt đối. Thời gian tuyệt đối không phản ánh tính độc hại của hành vi (mô hình học máy chuỗi sẽ học dựa trên thứ tự xuất hiện của chuỗi sự kiện thông qua `SEQ_LEN`, không dựa vào mốc thời gian lịch sử). |
| `Hashes`, `Hash` | Mã băm của file thực thi (MD5, SHA256). Mặc dù Hash rất hữu ích để kiểm tra chữ ký mã độc (Signature), nhưng đối với học máy phát hiện bất thường (Anomaly Detection), chuỗi Hash có số lượng giá trị duy nhất quá lớn (high cardinality). Việc đưa Hash vào Label Encoder sẽ tạo ra các nhãn số cực kỳ lớn và ngẫu nhiên, gây nhiễu nghiêm trọng cho mạng Autoencoder. |
| `Label` | Nhãn phân loại dữ liệu (`Normal` hoặc `Malicious`). Vì mô hình sử dụng kiến trúc **LSTM Autoencoder** là mô hình học bán giám sát/không giám sát (Semi-supervised Anomaly Detection). Mô hình chỉ được huấn luyện trên dữ liệu bình thường (`Normal.csv`) để học cách tái tạo hành vi bình thường. Nhãn `Label` chỉ được giữ lại trong tập test để tính toán điểm đánh giá (ROC AUC), không đưa vào lúc chạy hàm `fit`. |
