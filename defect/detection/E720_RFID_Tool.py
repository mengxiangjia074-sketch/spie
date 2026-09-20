#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 E720 超高频 RFID 读写模块 — 控制程序  V1.1
===============================================================================
 功能：标签识别 | 数据读写 | 功率/区域/信道设置

 硬件连接（USB 转 TTL，3.3V 电平）：
   USB-TTL TXD → E720 TTL_RXD(引脚6)
   USB-TTL RXD → E720 TTL_TXD(引脚7)
   USB-TTL GND → E720 GND

 通讯协议：E720 V4.3.3  |  默认波特率：115200 bps
===============================================================================
"""

import sys
import time
import serial
import serial.tools.list_ports


# ============================================================================
#  第一部分：常量定义
# ============================================================================

# --- 帧结构 ---
FRAME_HEADER = 0xBB      # 帧头
FRAME_END    = 0x7E      # 帧尾
FRAME_HEADER_BYTES = bytes([FRAME_HEADER])

# --- 帧类型 ---
TYPE_COMMAND      = 0x00  # 命令帧：上位机 → 模块
TYPE_RESPONSE     = 0x01  # 响应帧：模块 → 上位机
TYPE_NOTIFICATION = 0x02  # 通知帧：模块主动上报（读到标签）

# --- 命令码 ---
CMD_GET_INFO         = 0x03  # 获取模块信息
CMD_SET_REGION       = 0x07  # 设置工作地区
CMD_GET_REGION       = 0x08  # 获取工作地区
CMD_GET_SELECT       = 0x0B  # 获取 Select 参数
CMD_SET_SELECT       = 0x0C  # 设置 Select 参数
CMD_GET_QUERY        = 0x0D  # 获取 Query 参数
CMD_SET_QUERY        = 0x0E  # 设置 Query 参数
CMD_SET_SELECT_MODE  = 0x12  # 设置 Select 模式
CMD_SINGLE_INVENTORY = 0x22  # 单次寻卡
CMD_MULTI_INVENTORY  = 0x27  # 多次寻卡（连续读）
CMD_STOP_INVENTORY   = 0x28  # 停止多次寻卡
CMD_READ_DATA        = 0x39  # 读标签存储区
CMD_WRITE_DATA       = 0x49  # 写标签存储区
CMD_KILL_TAG         = 0x65  # 灭活标签（慎用）
CMD_LOCK_TAG         = 0x82  # 锁定标签存储区
CMD_GET_CHANNEL      = 0xAA  # 获取工作信道
CMD_SET_CHANNEL      = 0xAB  # 设置工作信道
CMD_SET_AUTO_FREQ    = 0xAD  # 设置自动跳频
CMD_SET_POWER        = 0xB6  # 设置发射功率
CMD_GET_POWER        = 0xB7  # 获取发射功率
CMD_ERROR            = 0xFF  # 错误响应

# --- 标签存储区 (MemBank) ---
MEMBANK_RFU  = 0x00  # RFU 保留区
MEMBANK_EPC  = 0x01  # EPC 存储区
MEMBANK_TID  = 0x02  # TID 存储区
MEMBANK_USER = 0x03  # User 用户区（最常用）

# --- 工作地区 ---
REGIONS = {
    "China2": 0x01, "China1": 0x04, "US": 0x02,
    "Europe": 0x03, "Korea": 0x06,
}
REGION_NAMES = {v: k for k, v in REGIONS.items()}

# --- 发射功率表 (dBm → 参数值) ---
POWER_TABLE = {
    15: 0x05DC, 16: 0x0640,
    17: 0x06A4, 18: 0x0708, 19: 0x076C, 20: 0x07D0,
    21: 0x0834, 22: 0x0898, 23: 0x08FC, 24: 0x0960,
    25: 0x09C4, 26: 0x0A28,
}

# --- 错误码 ---
ERROR_READ_OUT_OF_RANGE = 0xA3
ERROR_CODES = {
    0x09: "未找到指定标签",         0x10: "未找到指定标签(Write)",
    0x12: "未找到指定标签(Kill)",   0x13: "未找到指定标签(Lock)",
    0x15: "无标签响应/CRC错误",    0x16: "访问密码错误",
    ERROR_READ_OUT_OF_RANGE: "读超出存储区范围",
    0xB3: "写超出存储区范围",
    0xC4: "存储区已锁定",           0xD0: "灭活密码未设置",
}

DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT  = 1.0


# ============================================================================
#  第二部分：协议帧工具
# ============================================================================

def calc_checksum(data: bytes) -> int:
    """计算校验和：从 Type 到最后一个 Parameter 累加，取低 8 位"""
    return sum(data) & 0xFF


def build_frame(command: int, params: bytes = b"") -> bytes:
    """
    构建 E720 指令帧
    格式：BB | Type(00) | Cmd | PL_H | PL_L | Params... | Checksum | 7E
    """
    pl_msb = (len(params) >> 8) & 0xFF
    pl_lsb = len(params) & 0xFF
    middle = bytes([TYPE_COMMAND, command, pl_msb, pl_lsb]) + params
    checksum = calc_checksum(middle)
    return bytes([FRAME_HEADER]) + middle + bytes([checksum, FRAME_END])


def parse_frame(data: bytes) -> dict | None:
    """解析响应帧，返回包含 type/command/payload/checksum 的字典"""
    if len(data) < 7:
        return None
    idx = data.find(FRAME_HEADER_BYTES)
    if idx < 0:
        return None
    data = data[idx:]
    if len(data) < 7:
        return None
    if data[0] != FRAME_HEADER:
        return None

    frame_type = data[1]
    command = data[2]
    payload_len = (data[3] << 8) | data[4]
    # Header + Type + Cmd + PL(2) + Payload + Checksum + End
    total = 7 + payload_len
    if len(data) < total:
        return None

    payload = data[5:5 + payload_len]
    checksum_rx = data[5 + payload_len]
    if data[5 + payload_len + 1] != FRAME_END:
        return None

    checksum_calc = calc_checksum(data[1:5 + payload_len])
    return {
        "type": frame_type,
        "command": command,
        "payload_len": payload_len,
        "payload": payload,
        "checksum_received": checksum_rx,
        "checksum_calc": checksum_calc,
        "checksum_ok": (checksum_calc == checksum_rx),
        "start_index": idx,
        "total_len": total,
    }


def parse_notification_frame(frame: dict) -> dict | None:
    """从通知帧中提取标签 EPC 信息"""
    if (frame["type"] != TYPE_NOTIFICATION
            or frame["command"] != CMD_SINGLE_INVENTORY
            or frame.get("checksum_ok") is False
            or len(frame["payload"]) < 7):
        return None
    p = frame["payload"]
    rssi_raw = p[0]
    # 通知载荷固定为 RSSI(1) + PC(2) + EPC(N) + 标签 CRC(2)。
    # EPC 长度由标签 PC 决定，不能固定按常见的 12 字节截取。
    epc = p[3:-2]
    if not epc:
        return None
    return {
        "rssi": rssi_raw if rssi_raw < 128 else rssi_raw - 256,
        "pc": p[1:3],
        "epc": epc,
        "epc_hex": epc.hex().upper(),
        "crc": p[-2:],
    }


def parse_read_response(frame: dict) -> dict | None:
    """从读数据响应帧中提取数据"""
    if frame["command"] != CMD_READ_DATA or len(frame["payload"]) < 15:
        return None
    p = frame["payload"]
    ul = p[0]
    epc = p[3:3 + ul - 2]
    data = p[3 + ul - 2:]
    return {
        "ul": ul, "pc": p[1:3],
        "epc": epc, "epc_hex": epc.hex().upper(),
        "data": data, "data_hex": data.hex().upper(),
    }


def format_hex(data: bytes, sep: str = " ") -> str:
    """字节 → 可读十六进制字符串"""
    return sep.join(f"{b:02X}" for b in data)


def hex_to_bytes(hex_str: str) -> bytes:
    """十六进制字符串 → 字节（支持空格分隔）"""
    return bytes.fromhex("".join(hex_str.split()))


# ============================================================================
#  第三部分：E720 模块通讯类
# ============================================================================

class E720Module:
    """E720 超高频 RFID 模块 — 串口通讯封装"""

    def __init__(self):
        self.ser: serial.Serial | None = None
        self.port = ""
        self.connected = False
        self.buffer = b""

    # --- 连接 ---

    def connect(self, port: str, baudrate: int = DEFAULT_BAUDRATE,
                timeout: float = DEFAULT_TIMEOUT) -> bool:
        """连接模块，成功返回 True（失败时打印具体原因）"""
        # Windows: COM>=10 需要 \\\\.\\ 前缀
        ports_to_try = [port]
        if port.upper().startswith("COM") and not port.startswith("\\\\.\\"):
            try:
                num = int(port[3:])
                if num >= 10:
                    ports_to_try.append(f"\\\\.\\{port}")
            except ValueError:
                pass

        last_err = ""
        for p in ports_to_try:
            try:
                self.ser = serial.Serial(port=p, baudrate=baudrate,
                                         bytesize=serial.EIGHTBITS,
                                         parity=serial.PARITY_NONE,
                                         stopbits=serial.STOPBITS_ONE,
                                         timeout=timeout)
                self.port = port  # 对外显示原始端口名
                self.connected = True
                self.buffer = b""
                return True
            except serial.SerialException as e:
                last_err = str(e)
            except (ValueError, OSError) as e:
                last_err = str(e)

        # 全部失败，打印错误原因
        if "Access is denied" in last_err or "拒绝访问" in last_err:
            print(f"  [X] {port} 被其他程序占用，请关闭串口助手/监视器等后重试")
        elif "FileNotFoundError" in last_err or "系统找不到" in last_err or "does not exist" in last_err:
            print(f"  [X] {port} 不存在，请检查设备管理器中端口号是否正确")
        else:
            print(f"  [X] 无法打开 {port}: {last_err}")
        return False

    def disconnect(self):
        """断开连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False
        self.buffer = b""

    # --- 底层收发 ---

    def send_command(self, command: int, params: bytes = b"",
                     verbose: bool = False) -> bytes | None:
        """发送指令并读取原始响应"""
        if not self.connected or not self.ser or not self.ser.is_open:
            return None
        frame = build_frame(command, params)
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()
        if verbose:
            print(f"  [TX] {format_hex(frame)}")
        time.sleep(0.1)
        return self._read_raw()

    def _read_raw(self) -> bytes:
        """从串口读取全部可用数据"""
        all_data = b""
        timeout = self.ser.timeout if self.ser and self.ser.timeout is not None else 1.0
        deadline = time.time() + timeout
        hard_deadline = time.time() + max(timeout + 1.0, 2.0)
        while time.time() < deadline and time.time() < hard_deadline:
            waiting = self.ser.in_waiting if self.ser else 0
            if waiting > 0:
                chunk = self.ser.read(waiting)
                if chunk:
                    all_data += chunk
                    deadline = time.time() + 0.3
            else:
                time.sleep(0.05)
        return all_data

    def send_and_parse(self, command: int, params: bytes = b"",
                       verbose: bool = False) -> list[dict]:
        """发送指令，解析所有响应帧并返回"""
        raw = self.send_command(command, params, verbose)
        if not raw:
            return []
        if verbose and raw:
            print(f"  [RX] {format_hex(raw)}")
        frames = []
        offset = 0
        while offset < len(raw):
            pos = raw.find(FRAME_HEADER_BYTES, offset)
            if pos < 0:
                break
            f = parse_frame(raw[pos:])
            if f:
                offset = pos + f["total_len"]
                if not f["checksum_ok"]:
                    if verbose:
                        print("       └ 校验和错误, 已忽略该帧")
                    continue
                frames.append(f)
                if verbose:
                    type_name = {0: "CMD", 1: "RSP", 2: "NTF"}.get(f["type"], "?")
                    print(f"       └ {type_name} cmd=0x{f['command']:02X} "
                          f"pl={format_hex(f['payload'])} csum=OK")
            else:
                offset = pos + 1
        return frames

    # --- 模块信息 ---

    def get_module_info(self, info_type: int) -> str | None:
        """获取模块信息 (0=硬件版本, 1=软件版本, 2=制造商)"""
        for f in self.send_and_parse(CMD_GET_INFO, bytes([info_type])):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_GET_INFO:
                if len(f["payload"]) > 1:
                    return f["payload"][1:].decode("ascii", errors="replace")
        return None

    def get_module_summary(self) -> dict:
        """一次性获取模块基本信息摘要"""
        return {
            "hw": self.get_module_info(0x00) or "?",
            "sw": self.get_module_info(0x01) or "?",
            "mfr": self.get_module_info(0x02) or "?",
            "power": self.get_power(),
            "region": self.get_region(),
            "channel": self.get_channel(),
        }

    # --- Select 操作 ---

    def _send_select(self, params: bytes) -> bool:
        """发送 Select 指令，返回是否成功"""
        for f in self.send_and_parse(CMD_SET_SELECT, params):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_SET_SELECT:
                if f["payload"] and f["payload"][0] == 0x00:
                    return True
        return False

    def select_clear(self) -> bool:
        """
        清除 Select 筛选条件（无掩码模式）
        对应文档示例: BB 00 0C 00 07 23 00 00 00 00 60 00 96 7E
        """
        # SelParam=0x23, Ptr=0x00000000, MaskLen=0x60, Truncate=0x00
        params = bytes([0x23, 0x00, 0x00, 0x00, 0x00, 0x60, 0x00])
        return self._send_select(params)

    def select_epc(self, epc: bytes) -> bool:
        """
        按 EPC 卡号筛选指定标签
        对应文档示例: BB 00 0C 00 13 23 00 00 00 00 60 00 {EPC 12字节} 34 7E
        注意: SelParam=0x23 且 Ptr=0x00000000，与文档完全一致
        """
        if len(epc) != 12:
            print(f"  [!] EPC 长度错误：期望12字节，实际{len(epc)}字节")
            return False
        # SelParam=0x23, Ptr=0x00000000, MaskLen=0x60(96bit=12字节), Truncate=0x00
        params = bytes([0x23, 0x00, 0x00, 0x00, 0x00, 0x60, 0x00]) + epc
        return self._send_select(params)

    def set_select_mode(self, mode: int = 0x02) -> bool:
        """
        设置 Select 模式
        mode=0x00: 所有操作前都发送 Select
        mode=0x01: 不发送 Select
        mode=0x02: 仅 Read/Write/Lock/Kill 前发送 Select（推荐用于单标签读写）
        """
        for f in self.send_and_parse(CMD_SET_SELECT_MODE, bytes([mode])):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_SET_SELECT_MODE:
                return f["payload"] and f["payload"][0] == 0x00
        return False

    # --- 寻卡 ---

    def single_inventory(self) -> list[dict]:
        """单次寻卡，返回标签列表 [{epc_hex, rssi, epc, pc, crc}]"""
        results = {}
        for f in self.send_and_parse(CMD_SINGLE_INVENTORY):
            if f["type"] == TYPE_NOTIFICATION:
                tag = parse_notification_frame(f)
                if tag:
                    previous = results.get(tag["epc_hex"])
                    if previous is None or tag["rssi"] > previous["rssi"]:
                        results[tag["epc_hex"]] = tag
            elif f["type"] == TYPE_RESPONSE and f["command"] == CMD_ERROR:
                code = f["payload"][0] if f["payload"] else 0
                if code != 0x15:
                    print(f"  [!] {ERROR_CODES.get(code, f'错误 0x{code:02X}')}")
        return sorted(
            results.values(), key=lambda item: item["rssi"], reverse=True)

    def scan(self, duration: float = 3.0) -> list[dict]:
        """持续扫描指定秒数，自动去重，返回标签列表"""
        if not self.connected or not self.ser or not self.ser.is_open:
            return []
        # 发送连续寻卡
        params = bytes([0x22, 0xFF, 0xFF])  # 65535 次
        self.send_command(CMD_MULTI_INVENTORY, params)

        seen = {}
        deadline = time.time() + duration
        while time.time() < deadline:
            waiting = self.ser.in_waiting
            if waiting > 0:
                data = self.ser.read(waiting)
                self.buffer += data
                while len(self.buffer) >= 7:
                    pos = self.buffer.find(FRAME_HEADER_BYTES)
                    if pos < 0:
                        self.buffer = b""
                        break
                    if pos > 0:
                        self.buffer = self.buffer[pos:]
                    if len(self.buffer) < 5:
                        break
                    payload_len = (self.buffer[3] << 8) | self.buffer[4]
                    total_len = 7 + payload_len
                    if len(self.buffer) < total_len:
                        break
                    f = parse_frame(self.buffer[:total_len])
                    if not f:
                        self.buffer = self.buffer[1:]
                        continue
                    self.buffer = self.buffer[total_len:]
                    if f["checksum_ok"] and f["type"] == TYPE_NOTIFICATION:
                        tag = parse_notification_frame(f)
                        if tag and tag["epc_hex"] not in seen:
                            seen[tag["epc_hex"]] = tag
                            print(f"  >> {tag['epc_hex']}  RSSI:{tag['rssi']:d}dBm")
            else:
                time.sleep(0.05)

        self.send_and_parse(CMD_STOP_INVENTORY)
        return list(seen.values())

    # --- 读写 ---

    def read_selected_tag_result(
        self,
        mem_bank: int,
        start_addr: int,
        word_count: int,
        verbose: bool = False,
    ) -> dict:
        """读取已通过 Select 选中的标签，并保留模块错误码。"""
        log = print if verbose else lambda *a, **kw: None
        pwd = bytes(4)
        params = (
            pwd
            + bytes(
                [
                    mem_bank,
                    (start_addr >> 8) & 0xFF,
                    start_addr & 0xFF,
                    (word_count >> 8) & 0xFF,
                    word_count & 0xFF,
                ]
            )
        )
        log(
            f"  发送读指令: Bank={mem_bank} Addr={start_addr} "
            f"Len={word_count}Word..."
        )
        frames = self.send_and_parse(CMD_READ_DATA, params, verbose=verbose)

        for frame in frames:
            if frame["type"] != TYPE_RESPONSE:
                continue
            if frame["command"] == CMD_READ_DATA:
                parsed = parse_read_response(frame)
                if parsed and parsed["data"]:
                    log(f"  [OK] 读到 {len(parsed['data'])} 字节")
                    return {
                        "ok": True,
                        "data": parsed["data"],
                        "error_code": None,
                        "message": "",
                    }
                return {
                    "ok": False,
                    "data": None,
                    "error_code": None,
                    "message": "读响应中没有数据",
                }
            if frame["command"] == CMD_ERROR:
                code = frame["payload"][0] if frame["payload"] else 0
                message = ERROR_CODES.get(code, f"未知错误 0x{code:02X}")
                log(f"  [X] 模块返回错误: {message}")
                return {
                    "ok": False,
                    "data": None,
                    "error_code": code,
                    "message": message,
                }

        return {
            "ok": False,
            "data": None,
            "error_code": None,
            "message": "模块未返回读响应",
        }

    def read_tag(self, epc: bytes, mem_bank: int = MEMBANK_USER,
                 start_addr: int = 0, word_count: int = 4,
                 verbose: bool = False) -> bytes | None:
        """
        读取指定标签的存储区数据

        完整流程（参考 E720 协议文档 V4.3.3）:
          Step 1: 清除 Select 筛选         → BB 00 0C 00 07 23 00 00 00 00 60 00 96 7E
          Step 2: 设置 Select 按 EPC 筛选  → BB 00 0C 00 13 23 00 00 00 00 60 00 {EPC} xx 7E
          Step 3: 发送读指令               → BB 00 39 00 09 {Pwd4} {Bank} {SA2} {DL2} xx 7E
          Step 4: 解析返回数据

        参数:
            epc:        目标标签 EPC（12 字节）
            mem_bank:   存储区 (0=RFU, 1=EPC, 2=TID, 3=User)
            start_addr: 起始地址偏移（Word 单位，0 表示从第 0 个 Word 开始）
            word_count: 读取长度（Word 单位，1 Word = 2 字节）
            verbose:    是否打印详细调试信息
        """
        log = print if verbose else lambda *a, **kw: None

        # ── Step 1: 清除 Select ──
        log("  [1/3] 清除 Select...")
        if not self.select_clear():
            log("  [X] Select 清除失败")
            return None
        time.sleep(0.08)

        # ── Step 2: 按 EPC 筛选 ──
        log(f"  [2/3] 按 EPC 筛选: {format_hex(epc)}")
        if not self.select_epc(epc):
            log("  [X] EPC Select 失败")
            return None
        time.sleep(0.08)

        # ── Step 3: 发送读指令 ──
        result = self.read_selected_tag_result(
            mem_bank, start_addr, word_count, verbose=verbose
        )
        return result["data"] if result["ok"] else None

    def write_tag(self, epc: bytes, data: bytes,
                  mem_bank: int = MEMBANK_USER,
                  start_addr: int = 0,
                  verbose: bool = False) -> bool:
        """
        写入数据到指定标签的存储区

        完整流程（参考 E720 协议文档 V4.3.3）:
          Step 1: 清除 Select 筛选         → BB 00 0C 00 07 23 00 00 00 00 60 00 96 7E
          Step 2: 设置 Select 按 EPC 筛选  → BB 00 0C 00 13 23 00 00 00 00 60 00 {EPC} xx 7E
          Step 3: 发送写指令               → BB 00 49 00 xx {Pwd4} {Bank} {SA2} {DL2} {Data} xx 7E
          Step 4: 验证写入结果

        标签编辑规则（重要！）:
          - 数据必须是偶数长度（1 Word = 2 字节）
          - 起始地址也以 Word 为单位
          - 默认写入 User 存储区 (MemBank=0x03)
          - 写入前模块会用 Select 锁定指定 EPC 的标签
          - 写入完成后可自动回读验证

        参数:
            epc:        目标标签 EPC（12 字节）
            data:       要写入的数据（字节串）
            mem_bank:   存储区 (默认 0x03 = User 区)
            start_addr: 起始地址偏移（Word 单位）
            verbose:    是否打印详细调试信息（排查问题时建议开启）
        """
        log = print if verbose else lambda *a, **kw: None

        # 数据校验
        if len(data) % 2 != 0:
            log("  [X] 数据长度必须是偶数（2 字节 = 1 Word）")
            return False
        if len(epc) != 12:
            log("  [X] EPC 长度必须是 12 字节")
            return False

        # ── Step 1: 清除 Select ──
        log("  [1/3] 清除 Select...")
        if not self.select_clear():
            log("  [X] Select 清除失败 — 模块可能未响应")
            return False
        time.sleep(0.08)

        # ── Step 2: 按 EPC 选择要写入的标签 ──
        log(f"  [2/3] 按 EPC 选择标签: {format_hex(epc)}")
        if not self.select_epc(epc):
            log("  [X] EPC Select 失败 — 标签可能不在范围内")
            log("      请确认: 1)标签在读取范围 2)EPC 卡号正确")
            return False
        time.sleep(0.08)

        # ── Step 3: 发送写指令 ──
        wc = len(data) // 2  # 数据长度转换为 Word 单位
        pwd = bytes(4)       # 默认访问密码 00 00 00 00
        params = (pwd + bytes([mem_bank,
                               (start_addr >> 8) & 0xFF, start_addr & 0xFF,
                               (wc >> 8) & 0xFF, wc & 0xFF]) + data)
        log(f"  [3/3] 发送写指令: Bank={mem_bank} Addr={start_addr}W Len={wc}W({len(data)}B) Data={format_hex(data)}")
        frames = self.send_and_parse(CMD_WRITE_DATA, params, verbose=verbose)

        # ── Step 4: 解析结果 ──
        # 正常的响应顺序: 先收到 Select 的响应(BB 01 0C ...)，再收到 Write 的响应(BB 01 49 ...)
        # 写失败时: 收到 BB 01 FF ... (命令码 0xFF，不是 0x49!)
        for f in frames:
            if f["type"] != TYPE_RESPONSE:
                continue
            if f["command"] == CMD_WRITE_DATA:
                # 写成功: BB 01 49 00 10 ...
                # payload 最后一个字节是结果码 (0x00 = 成功)
                if f["payload"] and f["payload"][-1] == 0x00:
                    log("  [OK] 写入成功！")
                    return True
                else:
                    err = f["payload"][-1] if f["payload"] else 0xFF
                    log(f"  [X] 写入返回错误码: 0x{err:02X}")
            elif f["command"] == CMD_ERROR:
                # 写失败: BB 01 FF 00 10 {ErrorCode} ...
                code = f["payload"][0] if f["payload"] else 0
                msg = ERROR_CODES.get(code, f"未知错误 0x{code:02X}")
                log(f"  [X] 写入失败: {msg}")
                if code == 0x10:
                    log("      标签不在范围内，或 Select 未正确锁定目标标签")
                elif code == 0x16:
                    log("      标签设置了访问密码，当前使用默认密码 00 00 00 00")
                elif code == 0xB3:
                    log("      写入地址/长度超出标签存储区范围")

        # 没有收到任何相关响应
        if not frames:
            log("  [X] 未收到模块响应 — 请检查串口连接")
        return False

    # --- 设置 ---

    def set_power(self, dbm: int) -> bool:
        """设置发射功率 (15~26 dBm)"""
        if dbm not in POWER_TABLE:
            print(f"  [!] 功率须在 {sorted(POWER_TABLE.keys())} 中")
            return False
        v = POWER_TABLE[dbm]
        for f in self.send_and_parse(CMD_SET_POWER, bytes([(v >> 8) & 0xFF, v & 0xFF])):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_SET_POWER:
                return f["payload"] and f["payload"][0] == 0x00
        return False

    def get_power(self) -> int | None:
        """获取当前功率 (dBm)"""
        for f in self.send_and_parse(CMD_GET_POWER):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_GET_POWER:
                if len(f["payload"]) >= 2:
                    return ((f["payload"][0] << 8) | f["payload"][1]) // 100
        return None

    def set_region(self, code: int) -> bool:
        """设置工作地区"""
        for f in self.send_and_parse(CMD_SET_REGION, bytes([code])):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_SET_REGION:
                return f["payload"] and f["payload"][0] == 0x00
        return False

    def get_region(self) -> int | None:
        """获取当前工作地区代码"""
        for f in self.send_and_parse(CMD_GET_REGION):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_GET_REGION:
                if f["payload"]:
                    return f["payload"][0]
        return None

    def set_channel(self, ch: int) -> bool:
        """设置工作信道"""
        for f in self.send_and_parse(CMD_SET_CHANNEL, bytes([ch & 0xFF])):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_SET_CHANNEL:
                return f["payload"] and f["payload"][0] == 0x00
        return False

    def get_channel(self) -> int | None:
        """获取当前工作信道"""
        for f in self.send_and_parse(CMD_GET_CHANNEL):
            if f["type"] == TYPE_RESPONSE and f["command"] == CMD_GET_CHANNEL:
                if f["payload"]:
                    return f["payload"][0]
        return None


# ============================================================================
#  第四部分：用户界面
# ============================================================================

def clear_screen():
    """清屏"""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def hr(char: str = "─", width: int = 60) -> str:
    """分隔线"""
    return char * width


def show_banner():
    """显示程序标题横幅"""
    print(f"""
╔══════════════════════════════════════════════════════╗
║     E720 超高频 RFID 读写模块 — Python 控制程序     ║
║                     V1.1                             ║
╚══════════════════════════════════════════════════════╝""")


def show_status(module: E720Module):
    """显示模块当前状态（单行概览）"""
    if not module.connected:
        print(f"  [状态] 未连接")
        return
    info = module.get_module_summary()
    region_name = REGION_NAMES.get(info["region"] or 0, "?")
    power = f"{info['power']}dBm" if info["power"] else "?"
    channel = f"{info['channel']}" if info["channel"] is not None else "?"
    print(f"  [状态] {module.port} | HW:{info['hw']} SW:{info['sw']} | "
          f"地区:{region_name} 功率:{power} 信道:{channel}")


def show_tag_list(tags: list[dict]):
    """简洁打印标签列表"""
    if not tags:
        print("  (无标签)")
        return
    for i, t in enumerate(tags):
        print(f"  [{i+1}] {t['epc_hex']}  RSSI:{t['rssi']:d}dBm")


def pick_tag(module: E720Module, prompt: str = "选择标签") -> bytes | None:
    """先扫标签再让用户选择，返回选中的 EPC 字节"""
    print(f"\n  {prompt} — 正在扫描...")
    tags = module.single_inventory()
    if not tags:
        print("  (未发现标签)")
        return None
    show_tag_list(tags)
    print(f"  [M] 手动输入EPC  [B] 返回")
    c = input("  > ").strip().upper()
    if c == "B":
        return None
    if c == "M":
        h = input("  输入EPC (24位十六进制): ").strip()
        try:
            b = hex_to_bytes(h)
            if len(b) == 12:
                return b
            print("  [!] 需要12字节")
        except ValueError:
            print("  [!] 无效十六进制")
        return None
    try:
        i = int(c) - 1
        if 0 <= i < len(tags):
            return tags[i]["epc"]
    except ValueError:
        pass
    print("  [!] 无效选择")
    return None


def auto_connect(module: E720Module) -> bool:
    """自动检测串口并连接；多个串口时让用户选择"""
    # 如果已经连接，先断开
    if module.connected:
        print(f"  已连接 {module.port}，将先断开...")
        module.disconnect()

    port_list = [p.device for p in serial.tools.list_ports.comports()]
    if not port_list:
        print("  [X] 未检测到串口！请检查 USB 转 TTL 是否插入。")
        return False

    if len(port_list) == 1:
        # 只有一个串口，直接连接
        port = port_list[0]
    else:
        print(f"\n  检测到 {len(port_list)} 个串口：")
        for i, p in enumerate(port_list):
            print(f"    [{i+1}] {p}")
        print(f"    [M] 手动输入  [Q] 跳过")
        c = input("  > ").strip().upper()
        if c == "Q":
            return False
        if c == "M":
            port = input("  串口号 (如 COM3 或 /dev/ttyUSB0): ").strip()
        else:
            try:
                port = port_list[int(c) - 1]
            except (ValueError, IndexError):
                print("  [!] 无效选择")
                return False

    baud_str = input(f"  波特率 (默认 {DEFAULT_BAUDRATE}): ").strip()
    baud = int(baud_str) if baud_str else DEFAULT_BAUDRATE

    ok = module.connect(port, baud)
    if ok:
        print(f"  [OK] 已连接 {port} @ {baud}")
    else:
        print(f"  [X] 连接 {port} 失败")
    return ok


def cmd_scan(module: E720Module):
    """扫描标签"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    dur = input("  扫描时长(秒, 默认3): ").strip()
    try:
        d = float(dur) if dur else 3.0
    except ValueError:
        d = 3.0
    print(f"  扫描中 ({d}s)... (按 Ctrl+C 中断)")
    try:
        tags = module.scan(d)
        print(f"\n  ── 共 {len(tags)} 张标签 ──")
        show_tag_list(tags)
    except KeyboardInterrupt:
        module.send_and_parse(CMD_STOP_INVENTORY)
        print("\n  (已中断)")


def cmd_read(module: E720Module):
    """读取标签数据 — 带详细步骤显示"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    epc = pick_tag(module, "读取标签数据")
    if not epc:
        return

    print("\n  ═══ 读取参数 ═══")
    print("  存储区说明:")
    print("    [1] User 用户区 — 用户可自由读写的数据区（推荐）")
    print("    [2] EPC  卡号区 — 标签的唯一 ID（通常只读）")
    print("    [3] TID  唯一ID — 厂商烧录的 ID（只读）")
    bank_choice = input("  选择存储区 (默认1): ").strip()
    bank_map = {"1": MEMBANK_USER, "2": MEMBANK_EPC, "3": MEMBANK_TID}
    bank = bank_map.get(bank_choice, MEMBANK_USER)

    a = input("  起始地址 (Word单位, 默认0): ").strip()
    w = input("  读取长度 (Word单位, 默认4=8字节): ").strip()
    try:
        addr = int(a) if a else 0
        wc = int(w) if w else 4
    except ValueError:
        print("  [!] 请输入数字")
        return

    print(f"\n  ═══ 开始读取 ═══")
    data = module.read_tag(epc, bank, addr, wc, verbose=True)
    if data:
        print(f"  ┌──────────────────────────")
        print(f"  │ 读取成功! 数据内容:")
        print(f"  │ HEX  : {format_hex(data)}")
        print(f"  │ 原始 : {data.hex().upper()}")
        # 尝试 ASCII 显示
        try:
            text = data.decode("ascii")
            if all(32 <= ord(c) < 127 for c in text):
                print(f"  │ ASCII: {text}")
        except (UnicodeDecodeError, ValueError):
            pass
        print(f"  └──────────────────────────")
    else:
        print(f"  ═══ 读取失败! ═══")
        print(f"  可能原因:")
        print(f"    1) 标签不在天线范围内（按 S 扫描确认）")
        print(f"    2) 存储区为空（User 区初始值全为 0x00）")
        print(f"    3) EPC 卡号不正确")


def cmd_write(module: E720Module):
    """写入标签数据 — 支持编辑 EPC 卡号和 User 区数据"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return

    # ── 选择编辑目标 ──
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║              标签编辑（写入）操作指南                    ║
  ╠══════════════════════════════════════════════════════════╣
  ║                                                        ║
  ║  【编辑模式】                                           ║
  ║    [1] 编辑 EPC 卡号 — 修改标签的唯一 ID               ║
  ║    [2] 编辑 User 数据 — 修改用户存储区数据              ║
  ║                                                        ║
  ║  【编辑 EPC 卡号 规则】                                 ║
  ║    - EPC 固定 12 字节 (24 位十六进制)                   ║
  ║    - 示例: E2 00 10 71 00 00 52 9B 09 40 B4 02       ║
  ║    - 修改后标签 ID 永久改变！                           ║
  ║                                                        ║
  ║  【编辑 User 数据 规则】                                ║
  ║    - 数据偶数长度即可（2 字节 = 1 Word）                ║
  ║    - 示例: 01 02 03 04  或  00 01  或  FF FF          ║
  ║    - User 区初始值全为 00                               ║
  ║                                                        ║
  ║  【工作流程】                                           ║
  ║    ① 扫描标签 → ② 选择目标标签                         ║
  ║    ③ 显示当前数据 → ④ 输入新数据                       ║
  ║    ⑤ 确认 → ⑥ 写入 → ⑦ 自动回读验证                   ║
  ║                                                        ║
  ╚══════════════════════════════════════════════════════════╝
""")
    mode = input("  选择编辑模式 [1]=改EPC卡号 [2]=改User数据: ").strip()

    if mode == "1":
        # ─────────── 编辑 EPC 卡号 ───────────
        print(f"\n  ═══ 编辑 EPC 卡号 ═══")
        print(f"  说明: 将修改标签的唯一识别号(EPC)")
        print(f"        需要输入 12 字节 (24 位十六进制) 的新卡号")
        print(f"        修改后原卡号将无法识别，请谨慎操作！")

        epc_old = pick_tag(module, "先扫描 — 选择要修改的标签")
        if epc_old is None:
            return

        print(f"\n  当前 EPC: {format_hex(epc_old)}")
        print(f"  请输入新的 EPC 卡号（12 字节 = 24 位十六进制）")
        print(f"  示例: E2 00 10 71 00 00 52 9B 09 40 B4 02")
        new_hex = input("  新 EPC: ").strip()
        if not new_hex:
            print("  (未输入，已取消)")
            return
        try:
            new_epc = hex_to_bytes(new_hex)
        except ValueError:
            print("  [X] 无效的十六进制！只支持 0-9 A-F a-f")
            return
        if len(new_epc) != 12:
            print(f"  [X] EPC 必须是 12 字节！你输入了 {len(new_epc)} 字节 ({len(new_hex)} 个字符)")
            print(f"      12 字节 = 24 位十六进制字符（不含空格）")
            return

        # EPC 数据存储在 MemBank=0x01, 起始 Word 偏移 = 2
        # Word 0-1: CRC-16 (模块自动计算), Word 2-7: EPC 数据 (6 Word = 12 字节)
        bank = MEMBANK_EPC          # MemBank = 0x01
        addr = 2                    # 跳过 CRC-16(1Word) + PC(1Word)
        data = new_epc

        print(f"\n  ═══ 确认信息 ═══")
        print(f"  操作类型   : 修改 EPC 卡号")
        print(f"  存储区     : EPC 区 (MemBank=1)")
        print(f"  写入地址   : Word 偏移 2 (跳过 CRC-16 和 PC)")
        print(f"  旧 EPC     : {format_hex(epc_old)}")
        print(f"  新 EPC     : {format_hex(data)}")
        print(f"  ⚠ 警告: 写入后标签将使用新卡号，旧卡号失效！")

    elif mode == "2":
        # ─────────── 编辑 User 数据 ───────────
        print(f"\n  ═══ 编辑 User 区数据 ═══")
        epc_old = pick_tag(module, "写入 User 区 — 先扫描")
        if epc_old is None:
            return

        print(f"  目标标签: {format_hex(epc_old)}")
        print(f"  正在读取 User 区当前数据...")
        cur = module.read_tag(epc_old, MEMBANK_USER, 0, 4, verbose=True)
        if cur:
            print(f"  当前数据: {format_hex(cur)}")
        else:
            print(f"  (User 区为空或无法读取)")

        print(f"\n  请输入要写入的十六进制数据（偶数个字符）")
        print(f"  示例: 01 02 03 04 05 06 07 08 (8字节)")
        print(f"  示例: 00 01 (2字节)")
        dh = input("  新数据: ").strip()
        if not dh:
            return
        try:
            data = hex_to_bytes(dh)
        except ValueError:
            print("  [X] 无效的十六进制！")
            return
        if len(data) % 2 != 0:
            print(f"  [X] 数据长度必须是偶数！你输入了 {len(data)} 字节")
            return

        a = input("  起始地址 (Word单位, 默认0): ").strip()
        try:
            addr = int(a) if a else 0
        except ValueError:
            addr = 0

        bank = MEMBANK_USER
        print(f"\n  ═══ 确认信息 ═══")
        print(f"  操作类型   : 修改 User 区数据")
        print(f"  存储区     : User 区 (MemBank=3)")
        print(f"  起始地址   : Word {addr}")
        print(f"  写入数据   : {format_hex(data)} ({len(data)} 字节)")
    else:
        print("  [!] 无效选择，请输入 1 或 2")
        return

    confirm = input("\n  确认写入? 输入 Y 继续: ").strip().upper()
    if confirm != "Y":
        print("  已取消。")
        return

    # ── 执行写入 ──
    print(f"\n  ═══ 开始写入 ═══")
    success = module.write_tag(epc_old, data, bank, addr, verbose=True)

    if success:
        print(f"\n  [OK] ★★★ 写入成功! ★★★")
        time.sleep(0.3)

        if mode == "1":
            # EPC 已变，用新 EPC 验证 — 重新扫描
            print(f"  EPC 已更新，正在重新扫描验证...")
            new_tags = module.single_inventory()
            found_new = any(t["epc_hex"] == data.hex().upper() for t in new_tags)
            found_old = any(t["epc_hex"] == epc_old.hex().upper() for t in new_tags)
            if found_new:
                print(f"  [OK] 验证通过! 新 EPC {data.hex().upper()} 已生效 ✓")
            elif found_old:
                print(f"  [!] 旧 EPC 仍存在，新 EPC 未生效 — EPC 修改失败")
            else:
                print(f"  [OK] 旧 EPC 已消失 — 请按 S 扫描确认新卡号是否出现")
            print(f"\n  旧 EPC: {format_hex(epc_old)}")
            print(f"  新 EPC: {format_hex(data)}")
        else:
            # 回读验证 User 区
            print(f"  正在回读验证...")
            v = module.read_tag(epc_old, MEMBANK_USER, addr, len(data) // 2, verbose=True)
            if v:
                if v == data:
                    print(f"  [OK] 验证通过! 数据一致 ✓")
                else:
                    print(f"  [!] 数据不一致! 期望:{format_hex(data)} 实际:{format_hex(v)}")
            else:
                print(f"  [!] 无法回读")
    else:
        print(f"\n  ═══ 写入失败! 排查建议 ═══")
        print(f"  1. 按 S 扫描确认标签在范围内且信号良好")
        print(f"  2. 确认标签支持写入（部分标签 Preserved/EPC 区锁定）")
        print(f"  3. 用功率 [P] 调高发射功率（建议 20~26 dBm）")
        print(f"  4. 确认标签未设访问密码（默认密码 00 00 00 00）")
        print(f"  5. 尝试用 [X] 调试模式手动发送指令排查")
        print(f"     写EPC: 00 49 00 0D 00 00 00 00 01 00 02 00 06 + 12字节新EPC")
        print(f"     写User: 00 49 00 0D 00 00 00 00 03 00 00 00 04 + 8字节数据")
        print(f"     (第一条指令前需先发送 Select clear + Select mask)")


def cmd_power(module: E720Module):
    """设置功率"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    cur = module.get_power()
    if cur is not None:
        print(f"  当前功率: {cur} dBm")
    pwrs = sorted(POWER_TABLE.keys())
    print(f"  可选: {pwrs}")
    p = input("  功率(dBm): ").strip()
    try:
        module.set_power(int(p))
    except ValueError:
        print("  [!] 请输入数字")


def cmd_region(module: E720Module):
    """设置工作地区"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    cur = module.get_region()
    if cur is not None:
        print(f"  当前: {REGION_NAMES.get(cur, '?')}")
    print("  [1]China2(推荐) [2]China1 [3]US [4]Europe [5]Korea")
    c = input("  > ").strip()
    m = {"1": 0x01, "2": 0x04, "3": 0x02, "4": 0x03, "5": 0x06}
    if c in m:
        module.set_region(m[c])
    else:
        print("  [!] 无效选择")


def cmd_channel(module: E720Module):
    """设置工作信道"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    cur = module.get_channel()
    if cur is not None:
        print(f"  当前信道: {cur}")
    ch = input("  信道号: ").strip()
    try:
        module.set_channel(int(ch))
    except ValueError:
        print("  [!] 请输入数字")


def cmd_debug(module: E720Module):
    """调试模式：手动发送十六进制指令"""
    if not module.connected:
        print("  [X] 请先连接模块 (按 C)")
        return
    print("  输入十六进制指令 (不含帧头BB和帧尾7E)")
    print("  例: 00 22 00 00 = 单次寻卡")
    raw = input("  > ").strip()
    if not raw:
        return
    try:
        data = hex_to_bytes(raw)
    except ValueError:
        print("  [!] 无效十六进制")
        return
    if len(data) < 2:
        print("  [!] 指令太短")
        return

    frame = bytes([FRAME_HEADER]) + data
    frame += bytes([calc_checksum(data), FRAME_END])
    if not module.ser or not module.ser.is_open:
        return
    module.ser.reset_input_buffer()
    module.ser.write(frame)
    module.ser.flush()
    print(f"  [TX] {format_hex(frame)}")
    time.sleep(0.3)
    resp = module._read_raw()
    if resp:
        print(f"  [RX] {format_hex(resp)}")
        offset = 0
        while offset < len(resp):
            pos = resp.find(FRAME_HEADER_BYTES, offset)
            if pos < 0:
                break
            f = parse_frame(resp[pos:])
            if f:
                offset = pos + f["total_len"]
                tag_name = {0: "CMD", 1: "RSP", 2: "NTF"}.get(f["type"], "?")
                print(f"       {tag_name} cmd=0x{f['command']:02X} "
                      f"pl={format_hex(f['payload'])} "
                      f"csum={'OK' if f['checksum_ok'] else 'ERR'}")
                if f["type"] == TYPE_NOTIFICATION:
                    t = parse_notification_frame(f)
                    if t:
                        print(f"       → EPC:{t['epc_hex']} RSSI:{t['rssi']}dBm")
            else:
                offset = pos + 1
    else:
        print("  (无响应)")


# --- 主界面 ---

HELP_TEXT = """
  命令列表:
    [C] 连接模块      [D] 断开连接      [I] 模块信息
    [S] 扫描标签      [1] 单次寻卡
    [R] 读取标签数据  [W] 写入/编辑标签数据
    [P] 设置功率      [F] 设置地区      [H] 设置信道
    [X] 调试模式      [?] 帮助    [Q] 退出

  写标签快速示例:
    W → 选标签 → 输入 01 02 03 04 05 06 07 08 → 回车 → Y 确认
"""


def main():
    """主程序"""
    module = E720Module()
    show_banner()

    # 启动时尝试自动连接
    print(f"  检测串口...")
    port_list = [p.device for p in serial.tools.list_ports.comports()]
    if port_list:
        print(f"  可用: {', '.join(port_list)}")
        if len(port_list) == 1:
            print(f"  自动连接 {port_list[0]}...")
            if module.connect(port_list[0]):
                print(f"  [OK] 已连接")
        else:
            ans = input("  输入串口号或序号(回车跳过): ").strip()
            if ans:
                try:
                    idx = int(ans) - 1
                    port = port_list[idx] if 0 <= idx < len(port_list) else ans
                except ValueError:
                    port = ans
                module.connect(port)
    else:
        print("  (未检测到串口)")

    print(HELP_TEXT)

    while True:
        try:
            # 紧凑状态行
            if module.connected:
                info = module.get_module_summary()
                region = REGION_NAMES.get(info["region"] or 0, "?")
                pw = f"{info['power']}dBm" if info["power"] else "?"
                ch = f"{info['channel']}" if info["channel"] is not None else "?"
                prompt = f"\n[{module.port}] {region} {pw} ch{ch} > "
            else:
                prompt = "\n[未连接] > "

            choice = input(prompt).strip().upper()

            if choice == "Q":
                module.disconnect()
                print("  已退出。")
                break
            elif choice == "C":
                auto_connect(module)
            elif choice == "D":
                module.disconnect()
                print("  已断开。")
            elif choice == "I":
                if module.connected:
                    show_status(module)
                else:
                    print("  [X] 请先连接")
            elif choice == "S":
                cmd_scan(module)
            elif choice == "1":
                if module.connected:
                    tags = module.single_inventory()
                    show_tag_list(tags) if tags else print("  (未发现标签)")
                else:
                    print("  [X] 请先连接")
            elif choice == "R":
                cmd_read(module)
            elif choice == "W":
                cmd_write(module)
            elif choice == "P":
                cmd_power(module)
            elif choice == "F":
                cmd_region(module)
            elif choice == "H":
                cmd_channel(module)
            elif choice == "X":
                cmd_debug(module)
            elif choice == "?":
                print(HELP_TEXT)
            elif choice:
                print(f"  未知命令 '{choice}'，输入 ? 查看帮助")

        except KeyboardInterrupt:
            print("\n  按 Q 退出，其他键继续...")
            try:
                if input().strip().upper() == "Q":
                    break
            except KeyboardInterrupt:
                print("\n  已退出。")
                break

    module.disconnect()


if __name__ == "__main__":
    main()
