from __future__ import annotations

from dataclasses import dataclass

from pocketlab.schemas import SensorKind, TaskAnalyzerStatus


@dataclass(frozen=True)
class SensorRequirement:
    sensor: SensorKind
    label: str
    measurement_quantity: str
    recommended_phyphox_experiment: str
    analyzer_status: TaskAnalyzerStatus
    default_metric_key: str | None
    accepted_metric_keys: tuple[str, ...]


SENSOR_REQUIREMENTS: dict[SensorKind, SensorRequirement] = {
    "accelerometer": SensorRequirement(
        sensor="accelerometer",
        label="加速度计",
        measurement_quantity="三轴加速度、振动 RMS 与主频",
        recommended_phyphox_experiment="“加速度（不含重力）”或“加速度”实验",
        analyzer_status="ready",
        default_metric_key="selected_axis_rms_m_s2",
        accepted_metric_keys=(
            "selected_axis_rms_m_s2",
            "selected_axis_peak_to_peak_m_s2",
            "dominant_frequency_hz",
            "spectral_snr_db",
        ),
    ),
    "gyroscope": SensorRequirement(
        sensor="gyroscope",
        label="陀螺仪",
        measurement_quantity="三轴角速度",
        recommended_phyphox_experiment="输入为“陀螺仪 / gyroscope”的实验",
        analyzer_status="ready",
        default_metric_key="mean_angular_speed_rad_s",
        accepted_metric_keys=(
            "mean_angular_speed_rad_s",
            "angular_speed_std_rad_s",
            "peak_angular_speed_rad_s",
        ),
    ),
    "magnetometer": SensorRequirement(
        sensor="magnetometer",
        label="磁力计",
        measurement_quantity="三轴磁场强度",
        recommended_phyphox_experiment="输入为“磁力计 / magnetic field”的实验",
        analyzer_status="ready",
        default_metric_key="mean_field_magnitude_ut",
        accepted_metric_keys=(
            "mean_field_magnitude_ut",
            "field_magnitude_std_ut",
            "field_peak_to_peak_ut",
            "max_field_deviation_ut",
        ),
    ),
    "light": SensorRequirement(
        sensor="light",
        label="光线传感器",
        measurement_quantity="相对照度",
        recommended_phyphox_experiment="输入为“光线 / light”的实验",
        analyzer_status="ready",
        default_metric_key="median_illuminance_lx",
        accepted_metric_keys=(
            "median_illuminance_lx",
            "illuminance_iqr_lx",
            "coefficient_of_variation_ratio",
            "upper_plateau_fraction",
        ),
    ),
    "pressure": SensorRequirement(
        sensor="pressure",
        label="气压计",
        measurement_quantity="气压与相对高度变化",
        recommended_phyphox_experiment="输入为“气压 / pressure”的实验",
        analyzer_status="ready",
        default_metric_key="pressure_change_hpa",
        accepted_metric_keys=(
            "pressure_change_hpa",
            "relative_height_change_m",
            "pressure_trend_hpa_per_min",
            "pressure_mad_hpa",
        ),
    ),
    "proximity": SensorRequirement(
        sensor="proximity",
        label="接近传感器",
        measurement_quantity="接近状态或距离",
        recommended_phyphox_experiment="输入为“接近传感器 / proximity”的实验",
        analyzer_status="ready",
        default_metric_key="transition_count",
        accepted_metric_keys=(
            "observed_level_count",
            "signal_mode_code",
            "transition_count",
        ),
    ),
    "microphone": SensorRequirement(
        sensor="microphone",
        label="麦克风",
        measurement_quantity="相对声音幅值或频谱",
        recommended_phyphox_experiment=(
            "输入为“麦克风 / audio”的声音实验（例如“音频幅值 / Audio amplitude”）"
        ),
        analyzer_status="ready",
        default_metric_key="mean_relative_level_db",
        accepted_metric_keys=(
            "mean_relative_level_db",
            "peak_relative_level_db",
            "relative_level_span_db",
        ),
    ),
    "location": SensorRequirement(
        sensor="location",
        label="GPS / 位置",
        measurement_quantity="位置、轨迹或速度",
        recommended_phyphox_experiment="输入为“位置 / location (GPS)”的实验",
        analyzer_status="ready",
        default_metric_key="trajectory_distance_m",
        accepted_metric_keys=(
            "trajectory_distance_m",
            "displacement_m",
            "average_path_speed_m_s",
            "path_efficiency_ratio",
        ),
    ),
    "bluetooth": SensorRequirement(
        sensor="bluetooth",
        label="Bluetooth 外部传感器",
        measurement_quantity="外部设备定义的测量通道",
        recommended_phyphox_experiment="与外部 BLE 设备协议匹配的自定义实验",
        analyzer_status="detection_only",
        default_metric_key=None,
        accepted_metric_keys=(),
    ),
}

_SENSOR_KEYWORDS: list[tuple[SensorKind, tuple[str, ...]]] = [
    (
        "microphone",
        ("麦克风", "声音", "噪声", "声学", "声压", "响度", "音频", "audio", "microphone"),
    ),
    ("light", ("光线", "光照", "照度", "光强", "亮度", "灯光", "illuminance", "lux")),
    ("pressure", ("气压", "大气压", "气压计", "barometer", "pressure")),
    ("magnetometer", ("磁场", "磁力计", "磁感应", "指南针", "magnetometer")),
    ("gyroscope", ("陀螺仪", "角速度", "角位移", "gyroscope")),
    ("location", ("gps", "经纬度", "定位轨迹", "位置轨迹", "location")),
    ("proximity", ("接近传感器", "距离传感器", "近远状态", "proximity")),
    ("bluetooth", ("bluetooth", "ble", "外部传感器")),
    ("accelerometer", ("加速度", "振动", "震动", "冲击", "步频", "accelerometer")),
]


def sensor_requirement(sensor: SensorKind) -> SensorRequirement:
    return SENSOR_REQUIREMENTS[sensor]


def explicit_sensor_preference(text: str) -> SensorKind | None:
    """Return only a user-declared primary sensor, without broad keyword guessing."""

    return _match_explicit_sensor_preference(text)


def infer_task_sensor(
    declared_sensor: SensorKind,
    *,
    task_text: str,
    case_text: str = "",
) -> SensorKind:
    """Prefer explicit task language, then case context, then the declared/default sensor."""

    case_preference = _match_explicit_sensor_preference(case_text)
    if case_preference is not None:
        return case_preference
    task_match = _match_explicit_sensor_preference(task_text) or _match_sensor(task_text)
    if task_match is not None:
        return task_match
    case_match = _match_sensor(case_text)
    if case_match is not None and declared_sensor == "accelerometer":
        return case_match
    return declared_sensor


def _match_sensor(text: str) -> SensorKind | None:
    normalized = text.casefold()
    for sensor, keywords in _SENSOR_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return sensor
    return None


def _match_explicit_sensor_preference(text: str) -> SensorKind | None:
    """Respect phrases such as '优先使用气压，需要时调用麦克风'."""

    normalized = text.casefold()
    anchors = ("优先使用", "优先用", "首选", "先使用", "主要使用", "primary sensor")
    anchor_positions = [
        normalized.find(anchor) + len(anchor) for anchor in anchors if anchor in normalized
    ]
    if not anchor_positions:
        return None
    start = min(anchor_positions)
    explicit_aliases: dict[SensorKind, tuple[str, ...]] = {
        sensor: keywords for sensor, keywords in _SENSOR_KEYWORDS
    }
    # These short words are too broad for free-text fallback matching, but are
    # unambiguous inside an explicit phrase such as “优先使用位置的相对轨迹”.
    explicit_aliases["location"] += ("位置", "定位", "轨迹")
    matches: list[tuple[int, SensorKind]] = []
    for sensor, keywords in explicit_aliases.items():
        positions = [normalized.find(keyword, start) for keyword in keywords]
        positions = [position for position in positions if position >= start]
        if positions:
            matches.append((min(positions), sensor))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]
