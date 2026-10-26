"""
       里德堡原子超外差电场灵敏度测量程序
控制流程:
    1. 连接 LO 源 / SIG 源 / 频谱仪 (PyVISA) + 旋转镜架 (串口), LO 保留外参考配置，SIG 使用已验证的内部参考
    2. 镜架先转到扫描起始角 ANGLE_START(此时源尚未开启输出)
    3. 源参数设定(频率/功率直接可配)并开启输出
    4. 阶段一 角度扫描: 镜架从 ANGLE_START 起按 ANGLE_STEP 逐度转动,
       每点频谱仪单次扫描读 IF Marker 峰值(dBm), 存角度 CSV
    5. 阶段二 功率扫描(镜架停在阶段一终点): SIG 功率向下扫描(双速率),
       高于 FINE_FROM_DBM 用粗步进, 以下改细步进, 存功率 CSV
    6. 安全退出: finally 中停镜架、强制关闭两台微波源 RF 输出, 释放资源

依赖: pip install pyvisa pyserial numpy (LAN 通信需 Keysight IO Libraries / NI-VISA)
"""

import sys
import time

import numpy as np

from 旋转镜架 import ElliptecRotator

# ==================== 配置区 ====================

# ---- 本振微波源 LO ----
LO_VISA = "TCPIP0::192.168.1.101::inst0::INSTR"  # LO 源 (R&S SMB100A) VISA 地址
LO_FREQ_HZ = 18.97498917e9       # LO 源输出频率 (Hz, 倍频前基频): 填多少源就输出多少
LO_REFERENCE = "EXT"  # 保留原配置；须确认外参考连接及锁定
LO_POWER_DBM = 10        # LO 输出功率 (dBm): 最佳本振功率(镜架寻优结果)

# ---- 信号微波源 SIG ----
SIG_VISA = "TCPIP0::192.168.1.103::inst0::INSTR"  # SIG 源 (Keysight M9484C) VISA 地址
SIG_FREQ_HZ = 12.6500066689e9    # SIG 源输出频率 (Hz, 倍频前基频): 填多少源就输出多少
SIG_REFERENCE = "INT"  # 与已验证有输出的独立程序一致
# 正式共参考测量时，确认两源的外参考连接及锁定后再改为 EXT。
# 功率扫描(向下): 高于 FINE_FROM_DBM 用粗步进, 降到该值后改细步进
SIG_POWER_START = 8.0      # 扫描起始功率 (dBm)
SIG_POWER_STOP = -5     # 扫描终止功率 (dBm) —— 默认值, 按需修改!
SIG_POWER_STEP = 1.0        # 粗步进 (dB), 用于功率 > FINE_FROM_DBM 段
SIG_POWER_STEP_FINE = 0.1  # 细步进 (dB), 用于功率 <= FINE_FROM_DBM 段
FINE_FROM_DBM = 0        # 粗/细步进切换点 (dBm)

# ---- 安全限制 ----
MAX_POWER_DBM = 10.0        # [硬限制] 微波源输出功率上限 (dBm), 任何指令不得超过

# ---- 旋转镜架 (Thorlabs ELL21/M) + 角度扫描 ----
ELL_PORT = "COM17"          # 镜架串口号 (实测: ELL21/M SN 12100076 挂在 COM19)
# 角度范围支持跨零环绕: 如 316° → 36° 表示经 0°/360° 正向扫 80°;
# 步长取负则为反向扫描. 角度自动按 360° 取模, 镜架走最短路径(已实测).
ANGLE_START = 240           # 起始角度 (deg)
ANGLE_STOP = 300            # 终止角度 (deg)
ANGLE_STEP = 1              # 步长 (deg)

SETTLE_TIME = 1.0           # 镜架到位后机械稳定等待 (s)
MOVE_TIMEOUT_S = 30          # 单次运动的超时 (s)
POWER_SETTLE_S = 0.1        # 功率设置后的信号稳定等待 (s)

# ---- 频谱仪 ----
SA_VISA = "TCPIP0::192.168.1.107::inst0::INSTR"  # 频谱仪 VISA 地址
IF_CENTER_HZ = 250e3        # 中心频率 = 差频 (Hz)
IF_SPAN_HZ = 1e3           # 扫宽 (Hz): 只覆盖 IF 线宽附近的窄窗口
RBW_HZ = 1              # 分辨率带宽 (Hz): 噪声功率 ∝ RBW, 小 RBW 压低噪声底

# ---- 输出 ----
CSV_ANGLE_FILE = "heterodyne_angle_scan.csv"   # 阶段一: 角度扫描数据
CSV_POWER_FILE = "heterodyne_power_scan.csv"   # 阶段二: 功率扫描数据


# ==================== 微波源控制 ====================

class MicrowaveSource:
    """微波源封装 (PyVISA + SCPI), 兼容 R&S / Keysight 信号源"""

    def __init__(self, address, name="MW"):
        import pyvisa  # 延迟导入, 便于纯数据分析时不依赖 pyvisa
        self.name = name
        self.rm = pyvisa.ResourceManager()
        self.instr = self.rm.open_resource(address)
        self.instr.timeout = 10000                # VISA 超时 10 s
        self.instr.write_termination = "\n"
        self.instr.read_termination = "\n"

    @property
    def idn(self):
        return self.instr.query("*IDN?").strip()

    def _check_power(self, power_dbm):
        """安全限制: 任何情况下源输出不得超过 MAX_POWER_DBM"""
        if power_dbm > MAX_POWER_DBM:
            raise ValueError(
                f"[{self.name}] 拒绝设置功率 {power_dbm:.1f} dBm: "
                f"超过安全上限 {MAX_POWER_DBM:.1f} dBm")

    def setup(self, freq_hz, power_dbm, reference="INT"):
        """按已验证顺序设置，并检查 RF 状态与错误队列。"""
        self._check_power(power_dbm)
        reference = reference.strip().upper()
        if reference not in ("INT", "EXT"):
            raise ValueError(f"[{self.name}] 参考源必须为 INT 或 EXT")

        print(f"[{self.name}] 开始设置，参考源={reference}", flush=True)
        self.instr.write("*CLS")
        self.instr.write(f"ROSC:SOUR {reference}")
        time.sleep(0.2)
        self.instr.write(f"SOUR:FREQ {freq_hz}")
        self.instr.write(f"SOUR:POW {power_dbm}")
        self.instr.write("OUTP ON")
        time.sleep(1.0)

        output = self.instr.query("OUTP?").strip()
        freq = float(self.instr.query("SOUR:FREQ?"))
        power = float(self.instr.query("SOUR:POW?"))
        error = self.instr.query("SYST:ERR?").strip()
        print(f"[{self.name}] RF={output}，参考配置={reference}", flush=True)
        print(f"[{self.name}] 频率={freq / 1e9:.12f} GHz，功率={power:.2f} dBm")
        print(f"[{self.name}] 仪器错误={error}", flush=True)
        if output.upper() not in ("1", "ON"):
            raise RuntimeError(f"[{self.name}] RF 未成功开启")
        if int(error.split(",", 1)[0].strip()) != 0:
            raise RuntimeError(f"[{self.name}] 仪器返回错误：{error}")
        # 参考源配置不等于参考锁定；EXT 的锁定状态须另行确认。

    def set_power(self, power_dbm):
        """仅改功率(如需在扫描中调整源输出)"""
        self._check_power(power_dbm)               # 安全上限检查
        self.instr.write(f"SOUR:POW {power_dbm:.6g}")

    def rf_off(self):
        """尝试关闭 RF，并明确报告是否确认关闭。"""
        try:
            self.instr.write("OUTP OFF")
            output = self.instr.query("OUTP?").strip()
            if output.upper() in ("0", "OFF"):
                print(f"[{self.name}] RF 输出已关闭")
            else:
                print(f"[{self.name}] 关闭后 RF 状态为 {output}，请检查仪器")
        except Exception as exc:
            print(f"[{self.name}] 无法确认 RF 已关闭，请检查仪器：{exc}")

    def close(self):
        try:
            self.instr.close()
        finally:
            self.rm.close()


# ==================== 频谱仪控制 ====================

class SpectrumAnalyzer:
    """频谱仪封装 (PyVISA + SCPI): 读中频处 Marker 峰值功率"""

    def __init__(self, address, center_hz, span_hz, rbw_hz):
        import pyvisa
        self.rm = pyvisa.ResourceManager()
        self.instr = self.rm.open_resource(address)
        self.instr.timeout = 120000              # RBW=1 Hz 时单次扫描很慢, 120 s 超时
        self.instr.write_termination = "\n"
        self.instr.read_termination = "\n"

        # 复位并清状态, 保证程序可重复
        self.instr.write("*RST;*CLS")
        # 单次扫描模式: 由程序主动触发, 读数与功率点一一对应
        self.instr.write("INIT:CONT OFF")

        # 中心频率对准差频 delta_s, IF 信号落在屏幕中央
        self.instr.write(f"SENS:FREQ:CENT {center_hz:.15g}")
        # 极窄扫宽: 只看 IF 线宽附近, 缩短单次扫描时间
        self.instr.write(f"SENS:FREQ:SPAN {span_hz:.15g}")
        # RBW=100 Hz: 噪声功率 ∝ 带宽, 小 RBW 把噪声底压低 ~20 dB(RBW 从 10k 到 100)
        self.instr.write(f"SENS:BAND:RES {rbw_hz:.15g}")
        # 打开 Marker 1
        self.instr.write("CALC:MARK1 ON")

    @property
    def idn(self):
        return self.instr.query("*IDN?").strip()

    def read_if_marker(self):
        """触发单次扫描, 返回 (IF Marker 峰值功率 dBm, Marker 频率 Hz)"""
        # INIT:IMM: 触发一次新的单次扫描(旧轨迹作废)
        self.instr.write("INIT:IMM")
        # *OPC?: 阻塞等待本次扫描真正完成, 再读 Marker ——
        #   否则可能读到上一功率点的旧轨迹, 造成曲线"错位一个点"
        self.instr.query("*OPC?")
        # 峰值搜索: Marker1 自动跳到轨迹最高点(追踪 IF 峰)
        self.instr.write("CALC:MARK1:MAX")
        power_dbm = float(self.instr.query("CALC:MARK1:Y?"))   # 幅值 (dBm)
        freq_hz = float(self.instr.query("CALC:MARK1:X?"))     # 频率, 核对峰在 IF 处
        return power_dbm, freq_hz

    def close(self):
        try:
            self.instr.close()
        finally:
            self.rm.close()


# ==================== 旋转镜架控制 ====================

class ElliptecRotationStage:
    """Thorlabs ELL21/M 旋转镜架 (Elliptec 串口协议)

    与"最佳本征场测量"程序同款模块, 驱动见同目录 旋转镜架.py
    (已在 ELL21/M 实测: 360° 行程, 65536 步/圈). move_to 为阻塞式,
    返回时运动已完成, 出错(超行程/机械超时等)抛异常由 main 统一捕获.
    """

    def __init__(self, port=ELL_PORT):
        self.rot = ElliptecRotator(port=port)
        self.rot.get_info()
        print(f"[镜架] {self.rot.model}{'/M' if self.rot.metric else ''} "
              f"SN {self.rot.serial_number}, 行程 {self.rot.travel}°, "
              f"每圈 {self.rot.pulses_per_rev} 脉冲")

    def move_to(self, angle_deg):
        """绝对运动到指定角度(阻塞), 到位后再做机械稳定等待"""
        self.rot.move_absolute(float(angle_deg), max_wait=MOVE_TIMEOUT_S)
        time.sleep(SETTLE_TIME)       # 消除残余机械振动再开始测量

    def stop(self):
        # 'st' 急停(实测 ELL21 固件支持); 电机静止时发送亦无害
        try:
            self.rot.stop()
        except Exception:
            pass

    def close(self):
        self.rot.close()


# ==================== 扫描 ====================

def build_angle_list(start, stop, step):
    """生成扫描角度序列(按 360° 取模), 支持跨 0°/360° 环绕扫描

    规则(ELL21/M 固件对绝对运动自动走最短路径, 已实测):
      - step > 0 且 stop >= start : 正向普通扫描, 如 240° → 300°
      - step > 0 且 stop <  start : 正向跨零环绕, 如 316° → 36° (经 0°)
      - step < 0 且 stop <= start : 反向普通扫描, 如 90° → 0°
      - step < 0 且 stop >  start : 反向跨零环绕, 如 36° → 316° (经 0°)
    """
    if step == 0:
        raise ValueError("角度配置无效: ANGLE_STEP 不能为 0")
    if (step > 0 and stop >= start) or (step < 0 and stop <= start):
        span = abs(stop - start)          # 普通扫描: 直接跨度
    else:
        span = 360.0 - abs(stop - start)  # 跨零环绕: 补一圈的跨度
    n = int(round(span / abs(step)))
    return [round((start + i * step) % 360.0, 6) for i in range(n + 1)]


def build_power_list(start, stop, step, step_fine, fine_from):
    """生成 SIG 功率序列(从 start 向下扫到 stop, 双速率步进)

    - 功率 >  fine_from : 用 step 粗步进 (如 1 dB)
    - 功率 <= fine_from : 用 step_fine 细步进 (如 0.1 dB)
    末点精确落在 stop; 步进方向固定向下(高功率 -> 低功率).
    """
    if not stop < start:
        raise ValueError("功率扫描配置无效: 需 SIG_POWER_STOP < SIG_POWER_START")
    if step <= 0 or step_fine <= 0:
        raise ValueError("功率扫描配置无效: 步进须为正数(方向为向下)")
    powers = []
    p = float(start)
    while p > stop + 1e-9:
        powers.append(round(p, 4))
        p -= step if p > fine_from + 1e-9 else step_fine
    powers.append(round(float(stop), 4))
    return powers


def run_angle_scan(stage, sa, angles):
    """阶段一: 角度扫描主循环, 返回 (records, aborted)

    records 每行为 (镜架角度 deg, IF Marker 功率 dBm, Marker 频率 Hz)
    """
    records = []
    aborted = False
    n = len(angles)
    print(f"\n========== 阶段一 角度扫描: {angles[0]:.1f}° → {angles[-1]:.1f}°, "
          f"共 {n} 点 ==========")
    try:
        for i, ang in enumerate(angles, 1):
            stage.move_to(ang)              # 阻塞运动 + 机械稳定等待
            power, freq = sa.read_if_marker()
            records.append((float(ang), power, freq))
            print(f"[{i:>3}/{n}] θ = {ang:7.2f}°   P_IF = {power:8.2f} dBm "
                  f"@ {freq/1e3:.1f} kHz")
    except KeyboardInterrupt:
        aborted = True
        print("\n[中断] Ctrl+C, 保存已测数据并准备安全退出 ...")
    except Exception as exc:
        aborted = True
        print(f"\n[错误] 角度扫描中止({exc}), 保存已测数据 ...")
    return records, aborted


def run_power_scan(sig, sa, powers):
    """阶段二: SIG 功率扫描主循环, 返回 (records, aborted)

    records 每行为 (SIG 设定功率 dBm, IF Marker 功率 dBm, Marker 频率 Hz)
    """
    records = []
    aborted = False
    n = len(powers)
    print(f"\n========== 阶段二 功率扫描: {powers[0]:.1f} → {powers[-1]:.1f} dBm, "
          f"共 {n} 点 ==========")
    try:
        for i, p in enumerate(powers, 1):
            sig.set_power(p)              # 步进改变信号源功率
            time.sleep(POWER_SETTLE_S)   # 等待源输出稳定
            power, freq = sa.read_if_marker()
            records.append((float(p), power, freq))
            print(f"[{i:>3}/{n}] SIG = {p:7.1f} dBm   "
                  f"P_IF = {power:8.2f} dBm @ {freq/1e3:.1f} kHz")
    except KeyboardInterrupt:
        aborted = True
        print("\n[中断] Ctrl+C, 保存已测数据并准备安全退出 ...")
    except Exception as exc:
        aborted = True
        print(f"\n[错误] 功率扫描中止({exc}), 保存已测数据 ...")
    return records, aborted


# ==================== 数据保存 ====================

def save_records(records, filename, header):
    """原始数据存 CSV, 每行三列, 列名由 header 指定"""
    data = np.array(records)
    np.savetxt(filename, data, delimiter=",", header=header, comments="")
    print(f"\n数据文件: {filename}  (共 {len(data)} 点)")


# ==================== 主流程 ====================

def main():
    # ---- 配置校验与扫描计划 ----
    try:
        angles = build_angle_list(ANGLE_START, ANGLE_STOP, ANGLE_STEP)
    except ValueError as exc:
        sys.exit(f"[错误] {exc}")
    try:
        powers = build_power_list(SIG_POWER_START, SIG_POWER_STOP,
                                  SIG_POWER_STEP, SIG_POWER_STEP_FINE,
                                  FINE_FROM_DBM)
    except ValueError as exc:
        sys.exit(f"[错误] {exc}")
    if LO_POWER_DBM > MAX_POWER_DBM or SIG_POWER_START > MAX_POWER_DBM:
        sys.exit(f"[错误] 功率配置超过安全上限 {MAX_POWER_DBM:.1f} dBm: "
                 f"LO {LO_POWER_DBM} dBm, SIG 起始 {SIG_POWER_START} dBm")
    print(f"[计划] 阶段一: 角度 {ANGLE_START}° → {ANGLE_STOP}°, "
          f"步长 {ANGLE_STEP}°, 共 {len(angles)} 点")
    print(f"[计划] 阶段二: SIG 功率 {SIG_POWER_START} → {SIG_POWER_STOP} dBm "
          f"(> {FINE_FROM_DBM} dB 段步进 {SIG_POWER_STEP} dB, "
          f"以下 {SIG_POWER_STEP_FINE} dB), 共 {len(powers)} 点")
    print(f"[计划] LO {LO_FREQ_HZ/1e9:.7f} GHz @ {LO_POWER_DBM} dBm; "
          f"SIG {SIG_FREQ_HZ/1e9:.7f} GHz")

    lo = sig = sa = stage = None
    rec_angle, rec_power = [], []
    try:
        # ---- 1. 初始化: 三台仪器 + 旋转镜架 ----
        print("[初始化] 正在连接 LO ...", flush=True)
        lo = MicrowaveSource(LO_VISA, "LO")
        print("[LO ]", lo.idn)
        print("[初始化] 正在连接 SIG ...", flush=True)
        sig = MicrowaveSource(SIG_VISA, "SIG")
        print("[SIG]", sig.idn)
        print("[初始化] 正在配置频谱仪，尚未执行源输出设置 ...", flush=True)
        sa = SpectrumAnalyzer(SA_VISA, IF_CENTER_HZ, IF_SPAN_HZ, RBW_HZ)
        print("[SA ]", sa.idn)
        print("[初始化] 正在连接镜架 ...", flush=True)
        stage = ElliptecRotationStage()

        # ---- 2. 先旋转镜架到扫描起始角(此时源未开输出) ----
        print(f"[镜架] 正在移动到 {angles[0]:.1f}° ...", flush=True)
        stage.move_to(angles[0])
        print(f"[镜架] 已到达起始角 {angles[0]:.1f}°")

        # ---- 3. 源参数设定(频率/功率直接可配)并开启输出 ----
        lo.setup(LO_FREQ_HZ, LO_POWER_DBM, reference=LO_REFERENCE)
        sig.setup(SIG_FREQ_HZ, SIG_POWER_START, reference=SIG_REFERENCE)
        print("[参数] LO/SIG 设置及 RF 状态检查完成")

        # ---- 4. 阶段一: 角度扫描(镜架逐度转动, 每点读 IF) ----
        rec_angle, aborted = run_angle_scan(stage, sa, angles)
        if rec_angle:
            save_records(rec_angle, CSV_ANGLE_FILE,
                         "angle_deg,if_power_dbm,marker_freq_hz")

        # ---- 5. 阶段二: 功率扫描(镜架停在阶段一终点角度) ----
        if aborted:
            print("[提示] 角度扫描未完整结束, 跳过功率扫描")
        else:
            rec_power, _ = run_power_scan(sig, sa, powers)
            if rec_power:
                save_records(rec_power, CSV_POWER_FILE,
                              "sig_power_set_dbm,if_power_dbm,marker_freq_hz")

        if not rec_angle and not rec_power:
            sys.exit("[错误] 未采集到任何数据")

    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，准备关闭输出")
    except Exception as exc:
        print(f"\n[错误] {exc}")
    finally:
        # ---- 安全退出: 无论何种原因中断, 先停镜架, 再关 RF, 最后释放资源 ----
        if stage is not None:
            stage.stop()               # 确保马达静止
        for src, tag in ((sig, "SIG"), (lo, "LO")):
            if src is not None:
                src.rf_off()          # OUTP OFF, 防止持续辐射
        for dev in (lo, sig, sa):
            if dev is not None:
                try:
                    dev.close()
                except Exception as exc:
                    print(f"[清理] 释放 VISA 资源失败：{exc}")
        if stage is not None:
            try:
                stage.close()
            except Exception as exc:
                print(f"[清理] 释放镜架串口失败：{exc}")
        print("清理流程完成；请以上面的 RF 查询结果为准")


if __name__ == "__main__":
    main()
