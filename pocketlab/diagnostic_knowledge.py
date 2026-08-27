from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pocketlab.sensor_models import SensorAnalysis, SensorKind


@dataclass(frozen=True)
class DiagnosticAnalyzerGuide:
    """Source-derived interpretation contract exposed to the diagnostic reasoner.

    The contract mirrors the metrics and quality checks implemented by the analyzers.
    It deliberately exposes physical meaning and limits instead of Python source code,
    keeping prompts small and preventing implementation text from becoming authority.
    """

    sensor: SensorKind
    primary_metric_key: str
    signal_meaning: str
    metric_interpretations: dict[str, str]
    interpretation_rules: tuple[str, ...]
    confounders: tuple[str, ...]
    claim_limits: tuple[str, ...]
    user_checks: tuple[str, ...]


_GUIDES: dict[SensorKind, DiagnosticAnalyzerGuide] = {
    "accelerometer": DiagnosticAnalyzerGuide(
        sensor="accelerometer",
        primary_metric_key="selected_axis_rms_m_s2",
        signal_meaning="手机在测点处感受到的三轴加速度变化，可表征振动、冲击和周期运动。",
        metric_interpretations={
            "selected_axis_rms_m_s2": "主导轴振动的总体有效强度；同协议对照下降通常表示测点振动减弱。",
            "selected_axis_peak_to_peak_m_s2": "记录中最大正负摆幅范围，对偶发冲击和大幅摆动更敏感。",
            "dominant_frequency_hz": "频谱中最突出的重复频率，可用于关联转速、周期碰撞或结构选择性响应。",
            "spectral_snr_db": "主频相对频谱背景的突出程度；过低时不应强调单一主频。",
        },
        interpretation_rules=(
            "RMS 明显改变而主频近似保持，优先考虑激励强度或传递路径改变。",
            "主频随设备档位或转速同步改变，支持源驱动；主频固定但幅值在位置间改变，需考虑结构耦合。",
            "峰峰值很高但 RMS 变化有限时，检查偶发冲击、手机松动或短时碰撞。",
        ),
        confounders=("手机方向或测点改变", "手机未固定或外壳碰撞", "采样率不足导致混叠", "不同工况或记录时长"),
        claim_limits=("不能仅凭手机振动确定内部零件故障", "不能替代结构安全或设备维修鉴定"),
        user_checks=("让手机在两次测量中保持同一位置与方向", "一次只改变负载、支撑、接触或档位中的一项"),
    ),
    "gyroscope": DiagnosticAnalyzerGuide(
        sensor="gyroscope",
        primary_metric_key="mean_angular_speed_rad_s",
        signal_meaning="手机绕三个轴转动的角速度，可表征摆动、旋转和姿态不稳定。",
        metric_interpretations={
            "mean_angular_speed_rad_s": "整段记录的平均转动强度，适合比较两个受控状态的整体摆动。",
            "angular_speed_std_rad_s": "角速度波动程度；升高说明转动更不稳定或更间歇。",
            "peak_angular_speed_rad_s": "最强瞬时转动，适合发现短时摆动或冲击带来的姿态变化。",
        },
        interpretation_rules=(
            "平均值与波动同时下降，支持整体摆动或转动源减弱。",
            "只有峰值升高而平均值稳定时，优先检查偶发触碰、松动或一次性动作。",
            "陀螺仪描述手机转动，不等同于被测设备内部转速。",
        ),
        confounders=("手持抖动", "手机放置角度变化", "测点表面滑动", "记录起止时的人为拿取"),
        claim_limits=("不能从角速度单独确定内部旋转部件", "不能替代设备转速表或姿态标定系统"),
        user_checks=("开始前固定手机并在结束后再拿起", "对照时保持安装方向和测点一致"),
    ),
    "magnetometer": DiagnosticAnalyzerGuide(
        sensor="magnetometer",
        primary_metric_key="mean_field_magnitude_ut",
        signal_meaning="手机测得的三轴磁场强度，可表征附近磁性材料、电流相关设备或姿态变化造成的相对扰动。",
        metric_interpretations={
            "mean_field_magnitude_ut": "记录期间磁场模长平均水平，适合在同一手机与姿态下做相对比较。",
            "field_magnitude_std_ut": "磁场模长波动，升高说明环境扰动或运动影响增强。",
            "field_peak_to_peak_ut": "整段磁场变化范围，对短时靠近磁体或通电切换敏感。",
            "max_field_deviation_ut": "相对中位水平的最大偏离，用于定位突发磁扰动。",
        },
        interpretation_rules=(
            "设备状态切换后磁场变化可重复，支持存在状态相关磁扰动，但不直接证明某个内部部件故障。",
            "改变手机方向也会改变三轴分量；优先使用模长并固定姿态。",
            "高波动与高峰峰值更适合描述扰动，不应被解释为绝对磁场异常。",
        ),
        confounders=("手机姿态变化", "磁性保护壳或支架", "附近扬声器、电机和电源线", "缺少 accuracy 状态"),
        claim_limits=("不能用于市电安全判断", "不能从磁场变化推断具体内部电气故障"),
        user_checks=("移除磁性手机壳并标记手机方向", "先远离可见磁体和大电流设备建立背景记录"),
    ),
    "light": DiagnosticAnalyzerGuide(
        sensor="light",
        primary_metric_key="median_illuminance_lx",
        signal_meaning="手机感光位置接收到的照度，用于比较遮挡、朝向、距离或光源状态造成的相对变化。",
        metric_interpretations={
            "median_illuminance_lx": "对短时尖峰较稳健的典型照度水平，是受控条件比较的主指标。",
            "illuminance_iqr_lx": "中间一半读数的离散范围，较大表示照明或遮挡状态不稳定。",
            "coefficient_of_variation_ratio": "相对波动比例，可比较不同亮度水平下的稳定性。",
            "upper_plateau_fraction": "读数停留在最高平台的比例，过高可能提示量程饱和或平台截断。",
        },
        interpretation_rules=(
            "固定手机位置与朝向后，移除遮挡导致照度可重复上升，支持光路遮挡解释。",
            "中位数变化同时伴随高 IQR 或高变异系数时，应先排查闪烁、移动阴影或自动亮度波动。",
            "上平台比例过高时，不宜比较强光条件之间的真实幅度差。",
        ),
        confounders=("感光器位置或朝向改变", "人影和移动遮挡", "环境光随时间变化", "强光量程饱和"),
        claim_limits=("手机照度不等同于校准照度计", "不能据此做职业照明或法规合规认证"),
        user_checks=("标记手机感光器位置并固定朝向", "在相近时间完成对照并避免身体遮光"),
    ),
    "pressure": DiagnosticAnalyzerGuide(
        sensor="pressure",
        primary_metric_key="pressure_change_hpa",
        signal_meaning="手机气压计记录的相对气压变化，可辅助判断楼层、门窗压差或缓慢环境趋势。",
        metric_interpretations={
            "pressure_change_hpa": "记录起止的相对气压差，适合比较短时状态切换。",
            "relative_height_change_m": "由气压差换算的近似相对高度变化，仅用于短时相对比较。",
            "pressure_trend_hpa_per_min": "气压随时间的趋势，可区分稳定平台和持续漂移。",
            "pressure_mad_hpa": "气压围绕中位数的稳健离散程度，反映环境或采样稳定性。",
        },
        interpretation_rules=(
            "短时间内可重复的气压阶跃可支持楼层或封闭状态变化，缓慢趋势需考虑天气与传感器漂移。",
            "相对高度只在相近环境和短时间对照下解释，不作为绝对海拔。",
            "变化小于记录波动时，应报告未分辨到差异而不是无变化。",
        ),
        confounders=("天气系统变化", "空调或通风气流", "手机温度漂移", "门窗开合和高度同时变化"),
        claim_limits=("不能提供校准海拔", "不能用于建筑气密性或暖通合规认证"),
        user_checks=("缩短基线与对照之间的时间间隔", "记录楼层、门窗和空调状态并一次只改一项"),
    ),
    "proximity": DiagnosticAnalyzerGuide(
        sensor="proximity",
        primary_metric_key="transition_count",
        signal_meaning="手机接近传感器的离散或连续输出，用于判断遮挡物是否触发近/远状态切换。",
        metric_interpretations={
            "observed_level_count": "记录中出现的不同输出层级数量，用于判断单状态、二态或近似连续输出。",
            "signal_mode_code": "分析器识别的输出模式编码，只描述二态/连续模式，不代表真实距离。",
            "transition_count": "近远状态切换次数，适合检查遮挡、保护膜或触发不稳定。",
        },
        interpretation_rules=(
            "稳定的近/远切换支持传感器可响应，但二态输出不能换算为厘米距离。",
            "无切换且只见一个层级时，需先确认实验、遮挡动作和传感器位置。",
            "频繁切换可能来自边界抖动、遮挡不稳或保护膜影响。",
        ),
        confounders=("保护膜或手机壳遮挡", "不同机型为二态或连续输出", "传感器区域不明确", "手的角度和材质"),
        claim_limits=("二态接近输出不能证明具体距离", "不能用于安全防撞或人体检测认证"),
        user_checks=("查明听筒附近的传感器区域并清洁表面", "用同一物体按固定路径完成远—近—远动作"),
    ),
    "microphone": DiagnosticAnalyzerGuide(
        sensor="microphone",
        primary_metric_key="mean_relative_level_db",
        signal_meaning="由手机音频幅值派生的相对声级特征，可比较同一手机设置下不同工况的声音强弱。",
        metric_interpretations={
            "mean_relative_level_db": "整段记录的平均相对声音水平，适合受控工况比较。",
            "peak_relative_level_db": "最强瞬时相对声音水平，对敲击、启停或短时异响敏感。",
            "relative_level_span_db": "峰值与较低水平之间的跨度，用于描述声音是否间歇或突发。",
        },
        interpretation_rules=(
            "同一手机、增益、距离和朝向下的相对声级差可支持声源或传播路径变化。",
            "峰值升高但平均值稳定时，优先考虑短时异响而非持续噪声增强。",
            "相对 dB 未校准，不能与法规限值直接比较。",
        ),
        confounders=("自动增益或不同音频实验", "手机距离和朝向", "房间反射与背景声", "缓冲区错位或记录截断"),
        claim_limits=("不是校准声级计", "不能用于职业噪声、听力风险或产品合规判定"),
        user_checks=("保持手机距离、朝向和音频实验一致", "关闭无关声源并分别记录设备开与关"),
    ),
    "location": DiagnosticAnalyzerGuide(
        sensor="location",
        primary_metric_key="trajectory_distance_m",
        signal_meaning="手机位置点形成的轨迹、端点位移和速度，可描述室外移动路线与停留。",
        metric_interpretations={
            "trajectory_distance_m": "相邻有效位置段累积的路径长度。",
            "displacement_m": "起点到终点的直线位移，用于区分往返路线与净移动。",
            "average_path_speed_m_s": "总路径长度除以有效时间，描述整段平均移动速度。",
            "path_efficiency_ratio": "端点位移与总路径之比；低值通常表示绕行、往返或定位漂移。",
        },
        interpretation_rules=(
            "轨迹距离明显大于位移可表示绕行或往返，但须先排除定位精度造成的漂移。",
            "端点位移未超过水平不确定性时，不应宣称发生净移动。",
            "室内 GPS 往往不稳定，适合转向其他传感器或仅保留能力边界。",
        ),
        confounders=("室内或高楼遮挡", "水平 accuracy 缺失", "定位状态中断", "隐私裁剪或采样过稀"),
        claim_limits=("不能用于人身追踪或身份推断", "不能替代测绘、导航安全或执法级位置证据"),
        user_checks=("优先在安全开阔的室外路线测量", "开始前确认定位权限并记录 accuracy 状态"),
    ),
}


def diagnostic_analyzer_guides() -> dict[SensorKind, DiagnosticAnalyzerGuide]:
    return dict(_GUIDES)


def diagnostic_analyzer_guide(sensor: SensorKind) -> DiagnosticAnalyzerGuide:
    try:
        return _GUIDES[sensor]
    except KeyError as exc:
        raise ValueError(f"{sensor} has no numeric diagnostic interpretation contract") from exc


def analyzer_prompt_context(
    sensor: SensorKind,
    analysis: SensorAnalysis,
) -> dict[str, Any]:
    """Build a bounded, JSON-safe context directly from analyzer output and its guide."""

    guide = diagnostic_analyzer_guide(sensor)
    return {
        "sensor": sensor,
        "analyzer_id": analysis.analyzer_id,
        "analyzer_version": analysis.analyzer_version,
        "signal_meaning": guide.signal_meaning,
        "primary_metric_key": guide.primary_metric_key,
        "metric_interpretations": dict(guide.metric_interpretations),
        "interpretation_rules": list(guide.interpretation_rules),
        "confounders": list(guide.confounders),
        "claim_limits": list(guide.claim_limits),
        "user_checks": list(guide.user_checks),
        "quality": {
            "confidence": analysis.confidence,
            "sample_count": analysis.sample_count,
            "duration_s": analysis.duration_s,
            "sampling_rate_hz": analysis.sampling_rate_hz,
            "sampling_jitter_ratio": analysis.sampling_jitter_ratio,
            "max_sampling_gap_ratio": analysis.max_sampling_gap_ratio,
            "warnings": list(analysis.warnings),
        },
        "available_metrics": [item.model_dump(mode="json") for item in analysis.metrics],
    }


def analyzer_guide_as_dict(sensor: SensorKind) -> dict[str, Any]:
    return asdict(diagnostic_analyzer_guide(sensor))
