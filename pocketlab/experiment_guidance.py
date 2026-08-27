from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pocketlab.sensor_models import SensorKind
from pocketlab.sensor_requirements import SENSOR_REQUIREMENTS

QUALITY_CORRECTION_CORE_INSTRUCTION = (
    "先暂停本轮物理对照，不改变目标设备或环境工况。用不遮挡传感器开孔的手机支架或防滑垫，"
    "把手机重新固定在原测点标记处并恢复原朝向；确认所需传感器读数连续稳定后，"
    "恢复同一问题工况，等待信号稳定，只采集一个完整记录窗口"
)
QUALITY_CORRECTION_VARIABLE = "仅修正手机固定与采集完整性，不改变目标物理条件"
QUALITY_CORRECTION_CONTROLS = (
    "目标设备或环境工况",
    "记录时长",
    "非目标物理条件",
)
STABILITY_OBSERVATION_CORE_INSTRUCTION = (
    "不新增任何物理改变，保持刚完成的已定义对照条件、手机测点、朝向和记录时长不变；"
    "待信号稳定后只采集一个独立观察窗口，用于检查读数是否仍受瞬态波动影响"
)

_DURATION_PATTERN = re.compile(
    r"(?:连续|至少|约|不少于|不超过)?\s*\d+(?:\.\d+)?\s*(?:秒|分钟|s\b|min\b)",
    re.IGNORECASE,
)
_DURATION_VALUE_PATTERN = re.compile(
    r"(?P<prefix>连续|至少|约|不少于|不超过)?\s*"
    r"(?:记录|采集|测量)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>秒|分钟|s\b|min\b)",
    re.IGNORECASE,
)
_VAGUE_OPERATION_MARKERS = (
    "用户原问题中描述的",
    "一个安全可逆变化",
    "某个安全可逆变化",
    "一个尚未检验的安全因素",
    "某个尚未检验的安全因素",
    "确认后的单一因素",
    "目标条件",
    "适当改变",
    "按提示操作",
    "一个安全设置",
    "某个安全设置",
    "安全设置内容",
    "一个安全因素",
    "某个安全因素",
)
_ACTION_MARKERS = (
    "固定",
    "放置",
    "打开",
    "关闭",
    "启动",
    "停止",
    "进入",
    "等待",
    "记录",
    "遮挡",
    "移除",
    "增加",
    "减少",
    "移动",
    "旋转",
    "切换",
    "恢复",
    "重排",
    "靠近",
    "远离",
    "播放",
    "保持",
    "改变",
    "测量",
    "选择",
    "点击",
    "运行",
)

_GUIDE_SECTIONS = (
    "准备：",
    "操作：",
    "记录：",
    "保持不变：",
    "停止条件：",
)

_SENSOR_PLACEMENT_GUIDANCE: dict[SensorKind, str] = {
    "accelerometer": "用防滑垫或可移除胶带固定手机，防止滑动，并标记朝向",
    "gyroscope": "用防滑垫或可移除胶带固定手机，防止滑动，并标记朝向",
    "magnetometer": "取下带磁吸件的磁性手机壳，固定手机并标记朝向",
    "light": "确认光线传感器开孔没有被手、支架或手机壳遮挡，并固定朝向",
    "pressure": "确认手机气压开孔没有被手或胶带堵住，放稳后等待读数稳定",
    "proximity": "先找到屏幕顶部的接近传感器位置，确认手和支架不会意外遮挡",
    "microphone": "保持麦克风孔无遮挡，固定手机与声源的距离和朝向",
    "location": "选择空旷且安全、无需边走边操作手机的位置，等待 GPS 稳定",
    "bluetooth": "确认外部设备协议与实验匹配；Bluetooth 只做能力识别，不提交数值证据",
}


@dataclass(frozen=True)
class ExperimentGuidanceAudit:
    """Deterministic receipt proving that one UI task is executable as written."""

    executable: bool
    blocker_codes: tuple[str, ...]


class ExperimentGuidanceError(ValueError):
    """Raised before a vague or structurally impossible task enters a state machine."""

    def __init__(self, blocker_codes: tuple[str, ...]) -> None:
        super().__init__("experiment guidance failed: " + ", ".join(blocker_codes))
        self.blocker_codes = blocker_codes


def operation_text_is_specific(value: str) -> bool:
    """Accept an executable action while rejecting known placeholder prose."""

    normalized = " ".join(value.strip().split())
    return (
        len(normalized) >= 8
        and not any(marker in normalized for marker in _VAGUE_OPERATION_MARKERS)
        and any(marker in normalized for marker in _ACTION_MARKERS)
    )


_MULTI_RECORD_PATTERN = re.compile(
    r"(?:再|随后|然后|立即)\s*(?:重复|复测|记录|采集)\s*(?:一次|两次|2\s*次)"
    r"|(?:重复|复测)\s*(?:一次|两次|2\s*次)"
    r"|(?:两次|2\s*次)\s*(?:记录|采集|测量)"
    r"|分别\s*(?:记录|采集|测量)"
    r"|比较\s*(?:前后|两次|2\s*次)\s*(?:记录|采集|测量)?",
    re.IGNORECASE,
)


def operation_text_is_single_record(value: str) -> bool:
    """A diagnostic Task can accept exactly one new recording submission."""

    normalized = " ".join(value.strip().split())
    return bool(normalized) and _MULTI_RECORD_PATTERN.search(normalized) is None


def audit_experiment_operation_guide(
    value: str,
    *,
    sensors: Iterable[SensorKind],
    execution_mode: Literal["physical", "simulation"] = "physical",
) -> ExperimentGuidanceAudit:
    """Check the rendered user contract without trusting model wording."""

    normalized = " ".join(value.strip().split())
    sensor_tuple = tuple(dict.fromkeys(sensors))
    blockers: list[str] = []
    if not normalized:
        blockers.append("empty-guidance")
    if not sensor_tuple:
        blockers.append("missing-sensor")
    if any(normalized.count(section) != 1 for section in _GUIDE_SECTIONS):
        blockers.append("missing-or-duplicate-section")
    operation_section = normalized
    if "操作：" in normalized:
        operation_section = normalized.split("操作：", 1)[1].split("记录：", 1)[0]
    variable_section = ""
    for label in ("单一变量：", "本次复现：", "本次校正：", "本次观察：", "基线状态："):
        if label in normalized:
            variable_section = normalized.split(label, 1)[1].split("保持不变：", 1)[0]
            break
    actionable_text = f"{operation_section} {variable_section}"
    if any(marker in actionable_text for marker in _VAGUE_OPERATION_MARKERS):
        blockers.append("placeholder-operation")
    if not operation_text_is_specific(operation_section):
        blockers.append("operation-not-actionable")
    if not operation_text_is_single_record(operation_section):
        blockers.append("multiple-records-in-one-task")
    if _DURATION_PATTERN.search(normalized) is None:
        blockers.append("missing-record-duration")
    if execution_mode == "physical" and "phyphox" not in normalized.casefold():
        blockers.append("missing-phyphox-preparation")
    if execution_mode == "simulation" and not (
        "pocketlab" in normalized.casefold() and "模拟" in normalized
    ):
        blockers.append("missing-simulation-preparation")
    for sensor in sensor_tuple:
        requirement = SENSOR_REQUIREMENTS[sensor]
        if requirement.label not in normalized:
            blockers.append(f"missing-sensor-label:{sensor}")
    blocker_codes = tuple(dict.fromkeys(blockers))
    return ExperimentGuidanceAudit(
        executable=not blocker_codes,
        blocker_codes=blocker_codes,
    )


def assert_experiment_operation_guide(
    value: str,
    *,
    sensors: Iterable[SensorKind],
    execution_mode: Literal["physical", "simulation"] = "physical",
) -> ExperimentGuidanceAudit:
    """Return an audit receipt or reject the task with stable machine-readable codes."""

    audit = audit_experiment_operation_guide(
        value,
        sensors=sensors,
        execution_mode=execution_mode,
    )
    if not audit.executable:
        raise ExperimentGuidanceError(audit.blocker_codes)
    return audit


def concise_operation_label(value: str, *, fallback: str) -> str:
    """Turn an executable method into a compact UI label without inventing an action."""

    normalized = " ".join(value.strip().split())
    first_clause = re.split(r"[；;。]", normalized, maxsplit=1)[0].strip(" ，,:：")
    if not operation_text_is_specific(first_clause):
        return fallback
    return first_clause[:180]


def _clip_text(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split()).rstrip("。；;")
    if len(normalized) <= limit:
        return normalized
    window = normalized[:limit]
    sentence_end = max(window.rfind(marker) for marker in "。；;！？!?．")
    if sentence_end >= max(12, limit // 3):
        return window[: sentence_end + 1].rstrip("。；; ")
    clause_end = max(window.rfind(marker) for marker in "，、,:：")
    if clause_end >= max(12, limit // 2):
        return window[:clause_end].rstrip("，、,:： ") + "…"
    clipped = window[: max(1, limit - 1)].rstrip("，、；;：: ")
    clipped = re.sub(r"[A-Za-z][A-Za-z0-9_-]*$", "", clipped).rstrip()
    return clipped + "…"


def _bound_single_record_duration(
    value: str,
    *,
    sensors: tuple[SensorKind, ...],
) -> str:
    """Keep one household task short enough to execute and review safely."""

    maximum_seconds = 120 if "microphone" in sensors else 300

    def replace(match: re.Match[str]) -> str:
        numeric = float(match.group("value"))
        unit = match.group("unit").casefold()
        seconds = numeric * 60 if unit in {"分钟", "min"} else numeric
        if seconds <= maximum_seconds:
            return match.group(0)
        return f"连续 {maximum_seconds} 秒"

    return _DURATION_VALUE_PATTERN.sub(replace, value)


def _sensor_preparation(
    sensors: tuple[SensorKind, ...],
    *,
    core_instruction: str,
    task_kind: str,
    execution_mode: Literal["physical", "simulation"],
) -> str:
    requirements = [SENSOR_REQUIREMENTS[sensor] for sensor in sensors]
    labels = "、".join(item.label for item in requirements)
    if execution_mode == "simulation":
        return (
            f"在 PocketLab 打开模拟数据区域，确认本轮模拟会生成{labels}证据；"
            "无需连接 phyphox，也不要把结果当作真机测量"
        )
    if len(requirements) == 1:
        experiment_step = f"打开{requirements[0].recommended_phyphox_experiment}"
    else:
        ordered = "；".join(
            f"{index}.{item.recommended_phyphox_experiment}"
            for index, item in enumerate(requirements, start=1)
        )
        sequence = "→".join(str(index) for index in range(1, len(requirements) + 1))
        experiment_step = (
            f"优先打开能同时输出{labels}的复合实验；若没有，按 {sequence} 的顺序依次打开："
            f"{ordered}"
        )
    if task_kind == "correction":
        placement = "备好可移除测点标记和不遮挡传感器开孔的固定方式，暂不改变目标物理工况"
    else:
        placement = "；".join(
            dict.fromkeys(_SENSOR_PLACEMENT_GUIDANCE[sensor] for sensor in sensors)
        )
    return f"在 phyphox {experiment_step}，确认页面识别到{labels}；{placement}"


def _safety_stop(sensors: tuple[SensorKind, ...], notes: tuple[str, ...]) -> str:
    if "location" in sensors:
        default = "进入车行道、需要边走边操作手机或周围环境不安全时立即停止"
    elif "microphone" in sensors:
        default = "不要录制私人谈话；声音过大、设备异常或需要靠近危险声源时立即停止"
    else:
        default = "出现剧烈位移、异味、漏液、过热、裸露带电部件或其他危险迹象时立即停止"
    safe_notes = [" ".join(item.strip().split()) for item in notes if item.strip()]
    return "；".join((default, *safe_notes[:1]))


def build_experiment_operation_guide(
    *,
    core_instruction: str,
    sensors: Iterable[SensorKind],
    variable_to_change: str,
    controlled_variables: Iterable[str],
    default_duration_s: int,
    safety_notes: Iterable[str] = (),
    repeat_index: int | None = None,
    task_kind: str = "control",
    execution_mode: Literal["physical", "simulation"] = "physical",
) -> str:
    """Render a bounded task as a concrete, user-facing execution contract."""

    sensor_tuple = tuple(dict.fromkeys(sensors))
    if not sensor_tuple:
        raise ValueError("an experiment operation guide requires at least one sensor")
    core = _clip_text(
        _bound_single_record_duration(core_instruction, sensors=sensor_tuple),
        175,
    )
    variable = _clip_text(variable_to_change, 80)
    controls = [
        _clip_text(item, 40)
        for item in controlled_variables
        if item.strip()
    ]
    controls = list(dict.fromkeys(controls))[:4]
    if task_kind == "correction" and any(
        marker in variable_to_change for marker in ("固定", "测点", "朝向", "位置", "姿态")
    ):
        controls = [
            item
            for item in controls
            if not any(marker in item for marker in ("手机位置", "手机姿态", "测点与朝向"))
        ]
    notes = tuple(safety_notes)
    if execution_mode == "simulation" and len(sensor_tuple) > 1:
        record_window = (
            "本 Task 只运行一次模拟；为每个传感器各提交一条同窗模拟证据，"
            "运行期间不修改场景参数或切换条件"
        )
    elif execution_mode == "simulation":
        record_window = (
            "本 Task 只运行一次模拟并提交一个新记录；运行期间不修改场景参数或切换条件"
        )
    elif len(sensor_tuple) > 1:
        duration = (
            "按操作段时长"
            if _DURATION_PATTERN.search(core)
            else f"每次连续 {default_duration_s} 秒"
        )
        record_window = (
            f"每个传感器各提交一条证据；有复合实验则同窗采集，否则{duration}逐个采集，"
            "切换实验时不改变物理条件"
        )
    else:
        record_window = (
            "本 Task 只提交一个新记录；按操作段给出的时长采集，期间不要移动手机或切换条件"
            if _DURATION_PATTERN.search(core)
            else f"本 Task 只提交一个新记录；待目标工况稳定后连续采集 {default_duration_s} 秒，期间不要移动手机或切换条件"
        )
    repeat_note = (
        f"；这是该条件第 {repeat_index} 次独立重复，开始前先恢复同一初始布置"
        if repeat_index is not None
        else ""
    )
    operation_label = {
        "baseline": "基线状态",
        "control": "单一变量",
        "replication": "本次复现",
        "correction": "本次校正",
        "exploration": "本次观察",
    }.get(task_kind, "本次操作")
    default_controls = (
        "目标设备或环境工况、记录时长和非目标物理条件"
        if task_kind == "correction"
        else "手机测点、朝向、记录时长和非目标工况"
    )
    return (
        f"准备：{_clip_text(_sensor_preparation(sensor_tuple, core_instruction=core, task_kind=task_kind, execution_mode=execution_mode), 210)}。"
        f"操作：{core}。"
        f"记录：{_clip_text(record_window + repeat_note, 95)}。"
        f"{operation_label}：{variable}。"
        f"保持不变：{_clip_text('、'.join(controls), 90) if controls else default_controls}。"
        f"停止条件：{_clip_text(_safety_stop(sensor_tuple, notes), 75)}。"
    )
