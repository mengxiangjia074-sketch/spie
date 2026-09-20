"""
test/test_motion_controller.py

detection/motion_controller.py 中 main() 的测试程序, 主体与其保持一致。

从 test/ 目录引入 detection/motion_controller.py 并执行示例操作:
列出串口 -> 读取状态/报警/位置 -> 回零 -> 绝对定位 -> 关闭连接。

依赖安装: pip install minimalmodbus pyserial
"""

import sys
from pathlib import Path

# test/ 位于项目根目录, motion_controller.py 位于 detection/ 下。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))


def main():
    """示例主程序"""
    try:
        from motion_controller import (
            ADDR_X_AXIS,
            ADDR_Y_AXIS,
            MODE_ABSOLUTE,
            XYMotionController,
            list_serial_ports,
        )
    except ModuleNotFoundError as error:
        print(f"无法导入 detection/motion_controller.py: {error}")
        print("依赖安装: pip install minimalmodbus pyserial")
        return

    print("=" * 50)
    print("  运动控制器通讯程序 (Modbus RTU)")
    print("=" * 50)

    # 1. 列出可用串口
    list_serial_ports()

    # ---- 请根据实际情况修改以下参数 ----
    PORT = 'COM3'           # 修改为你的串口号
    BAUDRATE = 38400        # 波特率
    X_LEAD = 1.0            # X轴丝杠导程 (mm/转)
    Y_LEAD = 1.0            # Y轴丝杠导程 (mm/转)

    try:
        # 2. 创建 X/Y 位移台控制器实例
        stage = XYMotionController(
            PORT,
            baudrate=BAUDRATE,
            x_lead_mm_per_rev=X_LEAD,
            y_lead_mm_per_rev=Y_LEAD,
        )
        print(
            f"\nX/Y 位移台已连接 "
            f"(X地址={ADDR_X_AXIS}, Y地址={ADDR_Y_AXIS}, 串口={PORT})\n"
        )

        # 3. 示例操作

        # 读取状态
        print("--- 读取状态 ---")
        stage.x.print_status()
        stage.y.print_status()
        stage.x.read_alarm()
        stage.y.read_alarm()
        print(stage.read_motor_position())

        # stage.move_to_position(
        #                     x_mm=10,
        #                     y_mm=10,
        #                     speed_rpm=300,
        #                     mode=MODE_ABSOLUTE,
        #                     wait=True,
        #                 )

        # 回零
        print("\n--- 回零操作 ---")
        # stage.x.home()
        # stage.y.home()
        stage.move_to_position(
                    x_mm=0,
                    y_mm=0,
                    speed_rpm=300,
                    mode=MODE_ABSOLUTE,
                    wait=True,
                )
        # stage.x.wait_for_completion(timeout=60)
        # stage.y.wait_for_completion(timeout=60)

        print("\n--- 读取位置 ---")
        print(stage.read_motor_position())

        # 绝对定位：移动到 X=10mm, Y=5mm
        print("\n--- 绝对定位 X=10mm, Y=5mm ---")
        # stage.move_to_position(
        #     x_mm=0,
        #     y_mm=15.0,
        #     speed_rpm=300,
        #     mode=MODE_ABSOLUTE,
        #     wait=True,
        # )

        # 相对定位：X 正向移动 5mm，Y 负向移动 2mm
        # print("\n--- 相对定位 X +5mm, Y -2mm ---")
        # stage.move_to_position(
        #     x_mm=5.0,
        #     y_mm=-2.0,
        #     speed_rpm=300,
        #     mode=MODE_RELATIVE,
        #     wait=True,
        # )

        # # 读取当前位置
        # print("\n--- 读取当前位置 ---")
        # print(stage.read_motor_position())

        # 停止（演示急停）
        # stage.x.stop()
        # stage.y.stop()

        # 关闭连接
        stage.close()
        print("\n程序结束")

    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查:")
        print("  1. 串口号是否正确")
        print("  2. 设备是否已连接")
        print("  3. 波特率是否匹配")


if __name__ == '__main__':
    main()
