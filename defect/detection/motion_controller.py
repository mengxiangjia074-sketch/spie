"""
运动控制器通讯程序
基于 Modbus RTU 协议，通过串口控制运动控制器（X轴/Y轴）

依赖安装: pip install minimalmodbus pyserial
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Callable

import minimalmodbus
import serial.tools.list_ports


# ==================== 寄存器地址定义 ====================
REG_PULSE_PER_REV   = 0x0001   # 指令脉冲数/转
REG_BAUD_RATE       = 0x01BD   # 485波特率
REG_TRIGGER         = 0x6002   # 触发命令
REG_MOTION_MODE     = 0x6208   # 运动模式
REG_JOYSTICK_SPEED  = 0x6027   # 摇杆速度
REG_AUTO_SPEED      = 0x620B   # 自动速度
REG_ACCEL_TIME      = 0x620C   # 加速时间
REG_DECEL_TIME      = 0x620D   # 减速时间
REG_MOTOR_POS_H     = 0x602C   # 电机位置高16位（只读）
REG_MOTOR_POS_L     = 0x602D   # 电机位置低16位（只读）
REG_CMD_POS_H       = 0x6209   # 命令位置高16位
REG_CMD_POS_L       = 0x620A   # 命令位置低16位
REG_JOG_CONTROL     = 0x1801   # 点动控制
REG_STATUS          = 0x1003   # 状态监控（只读）
REG_ALARM           = 0x2203   # 报警读取（只读）
REG_ALARM_CLEAR     = 0x1801   # 报警清除

# ==================== 常量定义 ====================
# 触发命令
TRIGGER_POSITION    = 0x011    # 定位运行
TRIGGER_HOME        = 0x020    # 回零运行
TRIGGER_SET_ZERO    = 0x021    # 当前位置设为零
TRIGGER_STOP        = 0x040    # 停止

# 运动模式
MODE_ABSOLUTE       = 0x01     # 绝对位置模式
MODE_RELATIVE       = 0x41     # 相对位置模式

# 点动
JOG_FORWARD         = 0x4001   # 正向点动
JOG_REVERSE         = 0x4002   # 反向点动

# 报警清除
ALARM_CLEAR_CURRENT = 0x1111   # 复位当前报警
ALARM_CLEAR_HISTORY = 0x1122   # 复位历史报警

# 设备地址
ADDR_X_AXIS = 1
ADDR_Y_AXIS = 2

# 状态位
STATUS_FAULT         = 0x01    # Bit0: 故障
STATUS_ENABLED       = 0x02    # Bit1: 使能
STATUS_RUNNING       = 0x04    # Bit2: 运行
STATUS_INVALID       = 0x08    # Bit3: 无效
STATUS_CMD_DONE      = 0x10    # Bit4: 指令完成
STATUS_PATH_DONE     = 0x20    # Bit5: 路径完成
STATUS_HOME_DONE     = 0x40    # Bit6: 回零完成


class MotionController:
    """单轴运动控制器。"""

    def __init__(self, port: str, slave_address: int = ADDR_X_AXIS,
                 baudrate: int = 38400, lead_mm_per_rev: float = 1.0,
                 log_callback: Callable[[str], None] | None = None):
        """
        初始化运动控制器

        Args:
            port: 串口号，如 'COM3' 或 '/dev/ttyUSB0'
            slave_address: 从站地址，X轴=1, Y轴=2
            baudrate: 波特率，默认38400
            lead_mm_per_rev: 丝杠导程，单位 mm/转
            log_callback: 可选日志回调；未提供时保持命令行 print 输出
        """
        self.instrument = minimalmodbus.Instrument(port, slave_address)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = 1.0
        self.instrument.mode = minimalmodbus.MODE_RTU
        self.slave_address = slave_address
        self.lead_mm_per_rev = lead_mm_per_rev
        self._last_trigger = None
        self._log_callback = log_callback

    def _log(self, message: str) -> None:
        """输出控制器信息；GUI 可通过回调接管，避免写入启动终端。"""
        if self._log_callback is None:
            print(message)
        else:
            self._log_callback(message)

    # -------------------- 基础读写 --------------------

    def read_register(self, register_address: int) -> int:
        """读取单个寄存器"""
        return self.instrument.read_register(register_address,
                                              number_of_decimals=0,
                                              functioncode=0x03)

    def write_register(self, register_address: int, value: int):
        """写入单个寄存器"""
        self.instrument.write_register(register_address, value,
                                        number_of_decimals=0,
                                        functioncode=0x06)

    def read_registers(self, start_address: int, count: int) -> list:
        """读取多个寄存器"""
        return self.instrument.read_registers(start_address, count,
                                               functioncode=0x03)

    def write_registers(self, start_address: int, values: list):
        """写入多个寄存器"""
        self.instrument.write_registers(start_address, values)

    def read_pulses_per_rev(self) -> int:
        """读取协议寄存器 0x0001：指令脉冲数/转。"""
        pulses_per_rev = int(self.read_register(REG_PULSE_PER_REV))
        if pulses_per_rev <= 0:
            raise ValueError(
                f"[轴{self.slave_address}] 指令脉冲数/转无效: {pulses_per_rev}"
            )
        return pulses_per_rev

    @staticmethod
    def _to_decimal(value) -> Decimal:
        return Decimal(str(value))

    @staticmethod
    def _signed32_from_words(high_word: int, low_word: int) -> int:
        value = ((high_word & 0xFFFF) << 16) | (low_word & 0xFFFF)
        if value >= 0x80000000:
            value -= 0x100000000
        return value

    @staticmethod
    def _words_from_signed32(value: int) -> list:
        if value < -0x80000000 or value > 0x7FFFFFFF:
            raise ValueError(f"32位有符号位置超出范围: {value}")
        raw = value & 0xFFFFFFFF
        return [(raw >> 16) & 0xFFFF, raw & 0xFFFF]

    def pulses_to_mm(self, pulses: int, lead_mm_per_rev: float | None = None) -> float:
        """把脉冲位置换算成 mm。每次换算都读取当前指令脉冲数/转。"""
        lead = self._to_decimal(
            self.lead_mm_per_rev if lead_mm_per_rev is None else lead_mm_per_rev
        )
        pulses_per_rev = self._to_decimal(self.read_pulses_per_rev())
        return float(self._to_decimal(pulses) * lead / pulses_per_rev)

    def mm_to_pulses(self, position_mm: float, lead_mm_per_rev: float | None = None) -> int:
        """把 mm 位置换算成最接近的整数脉冲。"""
        lead = self._to_decimal(
            self.lead_mm_per_rev if lead_mm_per_rev is None else lead_mm_per_rev
        )
        if lead <= 0:
            raise ValueError(f"[轴{self.slave_address}] 丝杠导程必须大于0: {lead}")

        pulses_per_rev = self._to_decimal(self.read_pulses_per_rev())
        pulses = self._to_decimal(position_mm) * pulses_per_rev / lead
        return int(pulses.to_integral_value(rounding=ROUND_HALF_UP))

    # -------------------- 运动控制 --------------------

    def set_motion_mode(self, mode: int):
        """设置运动模式：MODE_ABSOLUTE 或 MODE_RELATIVE"""
        self.write_register(REG_MOTION_MODE, mode)
        mode_name = "绝对" if mode == MODE_ABSOLUTE else "相对"
        self._log(f"[轴{self.slave_address}] 设置运动模式: {mode_name}")

    def set_speed(self, speed_rpm: int):
        """设置运行速度（单位：RPM）"""
        self.write_register(REG_AUTO_SPEED, speed_rpm)
        self._log(f"[轴{self.slave_address}] 设置速度: {speed_rpm} RPM")

    def set_acceleration(self, accel_time: int):
        """设置加速时间（单位：ms/1000rpm，默认50）"""
        self.write_register(REG_ACCEL_TIME, accel_time)
        self._log(f"[轴{self.slave_address}] 设置加速时间: {accel_time}")

    def set_deceleration(self, decel_time: int):
        """设置减速时间（单位：ms/1000rpm，默认50）"""
        self.write_register(REG_DECEL_TIME, decel_time)
        self._log(f"[轴{self.slave_address}] 设置减速时间: {decel_time}")

    def move_to_position(self, position_mm: float, speed_rpm: int = 600,
                         mode: int = MODE_ABSOLUTE,
                         lead_mm_per_rev: float | None = None) -> int:
        """
        运动到指定位置（单位：mm）。

        Args:
            position_mm: 目标位置或相对位移，单位 mm
            speed_rpm: 运行速度（RPM）
            mode: 运动模式，MODE_ABSOLUTE 或 MODE_RELATIVE
            lead_mm_per_rev: 丝杠导程，默认使用初始化时的导程

        Returns:
            实际写入控制器的目标脉冲数。
        """
        position_pulses = self.mm_to_pulses(position_mm, lead_mm_per_rev)
        # 设置运动模式
        self.set_motion_mode(mode)
        # 设置速度
        self.set_speed(speed_rpm)
        # 设置加速度和减速度
        self.set_acceleration(50)
        self.set_deceleration(50)
        # 设置目标位置（高16位 + 低16位）
        pos_h, pos_l = self._words_from_signed32(position_pulses)
        self.write_registers(REG_CMD_POS_H, [pos_h, pos_l])
        self._log(
            f"[轴{self.slave_address}] 设置位置: {position_mm} mm "
            f"({position_pulses} 脉冲, H=0x{pos_h:04X}, L=0x{pos_l:04X})"
        )
        # 触发运行
        self.write_register(REG_TRIGGER, TRIGGER_POSITION)
        self._last_trigger = TRIGGER_POSITION
        self._log(f"[轴{self.slave_address}] 已触发定位运行")
        return position_pulses

    def move_distance_mm(self, distance_mm: float, lead: float | None = None,
                         speed_rpm: int = 300, mode: int = MODE_ABSOLUTE):
        """
        移动指定距离（单位：mm）

        Args:
            distance_mm: 距离（mm）
            lead: 丝杠导程（mm/转），默认使用初始化时的导程
            speed_rpm: 电机转速（RPM）
            mode: 运动模式
        """
        return self.move_to_position(
            distance_mm,
            speed_rpm=speed_rpm,
            mode=mode,
            lead_mm_per_rev=lead,
        )

    def home(self, wait: bool = True, timeout: float = 30.0):
        """回零运行"""
        self.write_register(REG_TRIGGER, TRIGGER_HOME)
        self._last_trigger = TRIGGER_HOME
        self._log(f"[轴{self.slave_address}] 触发回零运行")
        if wait:
            self.wait_for_completion(timeout=timeout)

    def set_current_position_zero(self):
        """将当前位置设为零点"""
        self.write_register(REG_TRIGGER, TRIGGER_SET_ZERO)
        self._last_trigger = TRIGGER_SET_ZERO
        self._log(f"[轴{self.slave_address}] 当前位置已设为零点")

    def stop(self):
        """急停"""
        self.write_register(REG_TRIGGER, TRIGGER_STOP)
        self._last_trigger = TRIGGER_STOP
        self._log(f"[轴{self.slave_address}] 已停止")

    def jog_forward(self):
        """正向点动"""
        self.write_register(REG_JOG_CONTROL, JOG_FORWARD)
        self._log(f"[轴{self.slave_address}] 正向点动")

    def jog_reverse(self):
        """反向点动"""
        self.write_register(REG_JOG_CONTROL, JOG_REVERSE)
        self._log(f"[轴{self.slave_address}] 反向点动")

    # -------------------- 状态读取 --------------------

    def read_motor_position_pulses(self) -> int:
        """读取电机当前位置原始值（32位有符号脉冲数）。"""
        values = self.read_registers(REG_MOTOR_POS_H, 2)
        position = self._signed32_from_words(values[0], values[1])
        return position

    def read_motor_position(self, lead_mm_per_rev: float | None = None) -> float:
        """读取电机当前位置，返回单位为 mm 的位置。"""
        position_pulses = self.read_motor_position_pulses()
        position_mm = self.pulses_to_mm(position_pulses, lead_mm_per_rev)
        self._log(
            f"[轴{self.slave_address}] 当前位置: {position_mm:.6f} mm "
            f"({position_pulses} 脉冲)"
        )
        return position_mm

    def read_status(self) -> dict:
        """读取状态监控"""
        status = self.read_register(REG_STATUS)
        result = {
            'fault':       bool(status & STATUS_FAULT),
            'enabled':     bool(status & STATUS_ENABLED),
            'running':     bool(status & STATUS_RUNNING),
            'invalid':     bool(status & STATUS_INVALID),
            'cmd_done':    bool(status & STATUS_CMD_DONE),
            'path_done':   bool(status & STATUS_PATH_DONE),
            'home_done':   bool(status & STATUS_HOME_DONE),
            'raw':         status,
        }
        return result

    def read_trigger_status(self) -> int:
        """读取触发状态（0x001=定位完成, 0x101=路径运行中）"""
        return self.read_register(REG_TRIGGER)

    def read_alarm(self) -> int:
        """读取报警信息"""
        alarm = self.read_register(REG_ALARM)
        if alarm != 0:
            alarm_names = {
                0x01: "过流",
                0x02: "过压",
                0x40: "电流采样回路故障",
                0x80: "锁轴（缺相）故障",
                0x200: "EEPROM故障",
                0x100: "参数自整定故障",
                0x020: "超差报警",
                0x008: "编码器断线报警",
                0x009: "输入IO重复配置",
                0x00A: "过温报警",
            }
            alarms = [name for code, name in alarm_names.items() if alarm & code]
            self._log(
                f"[轴{self.slave_address}] 报警: {', '.join(alarms)} "
                f"(码: 0x{alarm:04X})"
            )
        else:
            self._log(f"[轴{self.slave_address}] 无报警")
        return alarm

    def clear_alarm(self, clear_history: bool = False):
        """清除报警"""
        value = ALARM_CLEAR_HISTORY if clear_history else ALARM_CLEAR_CURRENT
        self.write_register(REG_ALARM_CLEAR, value)
        label = "历史" if clear_history else "当前"
        self._log(f"[轴{self.slave_address}] 已清除{label}报警")

    def print_status(self):
        """打印当前状态"""
        status = self.read_status()
        self._log(
            f"[轴{self.slave_address}] 状态: "
            f"故障={status['fault']}, 使能={status['enabled']}, "
            f"运行={status['running']}, 指令完成={status['cmd_done']}, "
            f"回零完成={status['home_done']}"
        )

    def wait_for_completion(self, timeout: float = 30.0):
        """等待最近一次回零或定位命令完成。"""
        import time
        start = time.time()
        saw_running = False
        while time.time() - start < timeout:
            status = self.read_status()
            alarm = self.read_register(REG_ALARM)

            if status["fault"] or alarm:
                raise RuntimeError(
                    f"[轴{self.slave_address}] 运动异常: "
                    f"status=0x{status['raw']:04X}, alarm=0x{alarm:04X}"
                )

            if status["running"]:
                saw_running = True

            trigger_status = self.read_trigger_status()
            if self._is_completion_status(status, trigger_status, saw_running, start):
                label = "回零" if self._last_trigger == TRIGGER_HOME else "定位"
                self._log(f"[轴{self.slave_address}] {label}完成")
                return True
            time.sleep(0.1)
        self._log(f"[轴{self.slave_address}] 等待超时")
        return False

    def _is_completion_status(self, status: dict, trigger_status: int,
                              saw_running: bool, start_time: float) -> bool:
        """根据当前命令类型判断控制器是否完成。"""
        import time

        if status["running"]:
            return False

        # 允许短动作没有被轮询到 running 状态，但避免刚触发时读到旧完成位。
        command_had_time_to_start = saw_running or (time.time() - start_time) >= 0.2
        if not command_had_time_to_start:
            return False

        if self._last_trigger == TRIGGER_HOME:
            return status["home_done"] or status["path_done"]

        if self._last_trigger == TRIGGER_POSITION:
            return (
                trigger_status == 0x001
                or status["cmd_done"]
                or status["path_done"]
            )

        return (
            trigger_status == 0x001
            or status["cmd_done"]
            or status["path_done"]
            or status["home_done"]
        )

    def close(self):
        """关闭串口连接"""
        self.instrument.serial.close()
        self._log(f"[轴{self.slave_address}] 连接已关闭")


class XYMotionController:
    """X/Y 双轴位移台控制器，坐标单位统一为 mm。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 38400,
        x_lead_mm_per_rev: float = 1.0,
        y_lead_mm_per_rev: float = 1.0,
        x_slave_address: int = ADDR_X_AXIS,
        y_slave_address: int = ADDR_Y_AXIS,
        log_callback: Callable[[str], None] | None = None,
    ):
        self.x = MotionController(
            port,
            slave_address=x_slave_address,
            baudrate=baudrate,
            lead_mm_per_rev=x_lead_mm_per_rev,
            log_callback=log_callback,
        )
        self.y = MotionController(
            port,
            slave_address=y_slave_address,
            baudrate=baudrate,
            lead_mm_per_rev=y_lead_mm_per_rev,
            log_callback=log_callback,
        )

    def read_motor_position(self) -> dict:
        """读取当前 X/Y 坐标，返回 {'x': x_mm, 'y': y_mm}。"""
        return {
            "x": self.x.read_motor_position(),
            "y": self.y.read_motor_position(),
        }

    def move_to_position(
        self,
        x_mm: float | None = None,
        y_mm: float | None = None,
        speed_rpm: int = 600,
        mode: int = MODE_ABSOLUTE,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> dict:
        """
        移动到指定 X/Y 坐标，单位 mm。

        x_mm 或 y_mm 为 None 时，对应轴不移动。
        """
        written_pulses = {}
        if x_mm is not None:
            written_pulses["x"] = self.x.move_to_position(
                x_mm, speed_rpm=speed_rpm, mode=mode
            )
        if y_mm is not None:
            written_pulses["y"] = self.y.move_to_position(
                y_mm, speed_rpm=speed_rpm, mode=mode
            )

        if wait:
            if x_mm is not None:
                self.x.wait_for_completion(timeout=timeout)
            if y_mm is not None:
                self.y.wait_for_completion(timeout=timeout)

        return written_pulses

    def close(self):
        """关闭 X/Y 两个轴的串口连接。"""
        self.x.close()
        self.y.close()


def list_serial_ports():
    """列出可用的串口"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未发现可用串口")
        return []
    print("可用串口:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    return [p.device for p in ports]
