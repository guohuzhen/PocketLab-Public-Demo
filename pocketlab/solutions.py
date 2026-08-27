from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, get_args

from pocketlab.schemas import (
    DiagnosticActionId,
    DiagnosticCase,
    DiagnosticFinalReport,
    DiagnosticOptionalRetest,
    DiagnosticSolutionAction,
    DiagnosticSolutionPlan,
)

_STOP_AND_ESCALATE = [
    "出现焦糊味、冒烟、漏电迹象或异常发热时，停止设备并在安全前提下断开电源。",
    "出现燃气异味、漏水、剧烈位移、金属撞击声或可见结构损伤时，不继续自行测试。",
    "需要拆机、接触市电、燃气、制冷剂或承重结构时，交由合格专业人员处理。",
]


@dataclass(frozen=True)
class _SafeAction:
    title: str
    rationale: str
    steps: tuple[str, ...]
    expected_result: str
    risk_level: str = "low"
    safety_notes: tuple[str, ...] = ()
    action_role: str = "resolve"


@dataclass(frozen=True)
class _ExecutionGuide:
    preparation: tuple[str, ...]
    verification: str
    if_not_improved: str
    estimated_time: str
    tools_needed: tuple[str, ...]
    do_not_do: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticFinalizationResult:
    report: DiagnosticFinalReport


_SAFE_ACTIONS: dict[DiagnosticActionId, _SafeAction] = {
    "preserve-and-observe": _SafeAction(
        "保留现场并记录触发条件",
        "先保留可复现条件，避免未经验证的改动破坏证据。",
        ("记录发生时间、工况和位置。", "只做外部目视检查，不拆机。"),
        "形成可复现的触发条件，并识别是否需要立即停止。",
        action_role="verify",
    ),
    "repeat-controlled-measurement": _SafeAction(
        "按同一条件重复一次测量",
        "重复可区分稳定机制与偶然波动。",
        ("手机回到同一测点和方向。", "保持工况与时长不变，只重复记录。"),
        "关键指标方向可重复，或明确暴露当前结论的不稳定性。",
        action_role="verify",
    ),
    "redistribute-balanced-load": _SafeAction(
        "停机后重新均匀分布衣物",
        "当前证据支持衣物偏载：衣物集中在滚筒一侧会增大高速旋转时的不平衡力。",
        (
            "通过洗衣机正常控制停止程序，等待门锁正常解除并确认滚筒完全停止。",
            "打开机门，把缠成一团或集中在一侧的衣物抖散，并在滚筒内均匀分布。",
            "若只有一件吸水后的大件，按制造商说明加入可搭配衣物或改用适合大件的程序；不要超载。",
            "关好机门，使用原来的脱水程序重新运行；发现剧烈位移或金属撞击立即停止。",
        ),
        "若偏载是主因，重新均匀分布后高速脱水的机身摆动、地面振动和碰撞声应明显减弱。",
        "caution",
        (
            "不要绕过门锁或安全联锁，不要在滚筒转动时伸手或移动洗衣机。",
            "不要为了平衡而超过制造商允许负载，也不要带电拆机。",
        ),
    ),
    "remove-external-contact": _SafeAction(
        "移除意外外部接触",
        "家具、线缆或杂物的硬接触可能传递振动、碰撞或遮挡信号。",
        ("停机后检查外部接触物。", "移开松散杂物或意外搭接，不拆外壳。"),
        "由外部接触造成的响应应减弱。",
        "caution",
        ("设备运行时不移动重物，不接触内部运动件。",),
    ),
    "stabilize-external-support": _SafeAction(
        "检查外部支撑与稳定性",
        "不稳定支撑会放大设备或家具的整体运动。",
        ("停机后确认可见接触点稳定。", "仅依据制造商说明调整用户可调支撑。"),
        "同一工况下的摆动或振动应降低。",
        "caution",
        ("不独自倾倒重型设备，不使用易滑或承载未知的临时垫材。",),
    ),
    "reduce-user-adjustable-source": _SafeAction(
        "降低用户可调的源强",
        "降低音量、速度或负载等正常设置可检验源强是否主导问题。",
        ("只调整产品正常提供的一个用户设置。", "保持测点与其他条件不变。"),
        "若该源主导问题，相关传感器指标应按预测方向减弱。",
        "caution",
        ("不绕过安全联锁，不超出制造商允许范围。",),
    ),
    "reposition-within-safe-use": _SafeAction(
        "在安全使用范围内调整位置或朝向",
        "位置和朝向会改变耦合、遮挡、反射或接收路径。",
        ("只做小幅、可逆的位置或朝向调整。", "不遮挡通风、通道和安全装置。"),
        "若路径机制成立，关键指标会随位置或朝向系统变化。",
        "caution",
    ),
    "improve-light-path": _SafeAction(
        "改善照明路径",
        "遮挡、朝向和反射面会改变到达感光面的光通量。",
        ("清除安全可移除的遮挡。", "调整灯具正常可调方向或工作面位置。"),
        "目标区域照度提高且重复测量方向一致。",
        "low",
        ("不拆灯具，不直视强光，不登上不稳家具。",),
    ),
    "reduce-acoustic-exposure": _SafeAction(
        "降低声源暴露",
        "降低源强、增加距离或改变朝向可减少到达手机和人的相对声学水平。",
        ("先降低用户可调音量或缩短暴露时间。", "在不影响通风时增加距离或改变朝向。"),
        "同一手机设置下的相对声音指标降低。",
        "low",
        ("手机派生声学量不是校准声级计，不用于职业噪声合规判断。",),
    ),
    "reduce-magnetic-interference": _SafeAction(
        "移开可控磁干扰并固定手机姿态",
        "磁性手机壳、扬声器、电机和大电流线缆会改变手机附近的相对磁场。",
        ("移除磁性手机壳或磁吸支架。", "在不移动被测设备的前提下移开附近可见磁体或无关电器。"),
        "同一姿态下的磁场波动或状态差异减小。",
        "caution",
        ("不打开电器外壳，不接触裸露线路，不把磁体靠近医疗植入设备。",),
    ),
    "clear-sensor-path": _SafeAction(
        "清理传感器外部通道",
        "保护膜、手机壳、灰尘或可移除遮挡可能妨碍光线或接近传感器正常响应。",
        ("先确认手机上对应传感器的大致位置。", "用干燥软布清洁表面并移开可安全移除的遮挡。"),
        "传感器输出恢复稳定，受控动作能产生可重复的状态变化。",
        "low",
        ("不使用腐蚀性清洁剂，不拆开手机或被测设备。",),
    ),
    "verify-environmental-context": _SafeAction(
        "核对环境条件后再判断",
        "气压、位置和光照会受到天气、楼层、门窗、遮挡与定位环境影响。",
        ("记录测量时间、地点、楼层和门窗或空调状态。", "在短时间内用相同设置重复参考条件。"),
        "环境变量被固定后，关键差异仍可重复，或原先差异被识别为环境波动。",
        "low",
    ),
    "isolate-operating-source": _SafeAction(
        "逐一隔离可控运行源",
        "多个设备同时运行时，单次测量无法确定哪个声源、振动源或磁扰动源占主导。",
        ("列出现场正在运行且可正常关闭的设备。", "每次只通过正常控制关闭一个设备并保持其他条件不变。"),
        "关闭某一来源时关键指标可重复下降，从而定位主要贡献源。",
        "caution",
        ("不拔除不明线路，不关闭生命安全、网络基础或关键制冷设备。",),
    ),
    "check-manufacturer-guidance": _SafeAction(
        "核对制造商的用户维护指引",
        "设备专用限制、调平和清洁步骤应以正式说明为准。",
        ("按型号查阅官方说明。", "只执行标为用户可维护的步骤。"),
        "避免因通用建议引入与具体设备不兼容的操作。",
        "low",
        action_role="escalate",
    ),
    "request-professional-inspection": _SafeAction(
        "安排专业检查",
        "当现象涉及内部部件、电气、燃气、结构或持续异常时，手机证据只能辅助描述。",
        ("保存报告和发生工况。", "向合格人员说明已观察到的条件与传感器变化。"),
        "由具备资质的人员确认内部原因和维修范围。",
        "professional",
        action_role="escalate",
    ),
}


_EXECUTION_GUIDES: dict[DiagnosticActionId, _ExecutionGuide] = {
    "preserve-and-observe": _ExecutionGuide(
        ("先判断是否触发下方任何停止条件。", "准备手机备忘录，记录时间、工况和持续时长。"),
        "至少记录两次相同触发条件；若现象、工况和指标方向一致，可视为可复现。",
        "若无法复现，保留记录并等待下一次自然出现，不要为了制造现象而超范围操作设备。",
        "5–10 分钟",
        ("手机备忘录或纸笔",),
        ("不要拆机寻找原因", "不要忽略焦糊味、漏电、燃气或结构异常"),
    ),
    "repeat-controlled-measurement": _ExecutionGuide(
        ("确认使用同一个 phyphox 实验和通道。", "用胶带或照片标记手机测点与朝向。"),
        "重复值应保持相同变化方向，且差异大于分析器报告的波动或质量告警。",
        "若方向不一致，先检查手机位置、记录时长、工况和采样告警，再把结论降为不确定。",
        "8–15 分钟",
        ("手机", "phyphox", "用于标记位置的可移除胶带"),
        ("不要在两次记录之间同时改变多个条件",),
    ),
    "redistribute-balanced-load": _ExecutionGuide(
        (
            "先通过正常控制停机，等待滚筒完全停止且门锁正常解除。",
            "保留本次衣物总量和程序设置；若含单件吸水大件，先查看制造商的大件洗涤说明。",
        ),
        "在同一程序下观察高速脱水是否恢复平稳；如要量化，可选用原测点做一次处理后复测，但不要求继续诊断才能执行本建议。",
        "若均匀重排后仍剧烈振动，停止反复高速脱水，下一步检查设备是否水平、四个支脚是否按说明稳定着地；仍异常则联系售后。",
        "5–15 分钟",
        ("无需额外工具", "可选：手机相机记录处理前后现象"),
        (
            "不要在运行中开门、伸手或移动洗衣机",
            "不要绕过门锁、安全联锁或超出额定负载",
        ),
    ),
    "remove-external-contact": _ExecutionGuide(
        ("通过设备正常控制停机并等待运动部件完全停止。", "拍照记录原有接触位置，便于恢复。"),
        "恢复相同工况后，比较处理前后的主指标和异常声音或碰撞是否同时减弱。",
        "若无改善，恢复原状态并转向支撑、源强或专业检查，不继续扩大拆动范围。",
        "10–20 分钟",
        ("手电筒", "手机相机"),
        ("不要接触内部运动件", "不要带电移动重型设备"),
    ),
    "stabilize-external-support": _ExecutionGuide(
        ("停机并确认设备不会倾倒或滑动。", "查阅该型号官方调平或支撑说明。"),
        "同一负载和档位下，振动或摆动指标下降且设备没有新增位移。",
        "若支撑已符合说明但异常持续，停止继续垫高或倾斜，保存报告并联系售后。",
        "15–30 分钟",
        ("制造商说明", "适合时使用水平尺"),
        ("不要使用承载未知的软垫或临时物品", "不要独自倾倒重型设备"),
    ),
    "reduce-user-adjustable-source": _ExecutionGuide(
        ("确认要调整的是产品正常提供的档位、音量、速度或负载。", "记录调整前设置。"),
        "只改变一个档位后，相关主指标按报告预测方向变化且可重复。",
        "若指标不随设置变化，恢复原设置并检查传递路径、其他运行源或竞争机制。",
        "5–15 分钟",
        ("设备正常控制面板", "手机"),
        ("不要绕过联锁", "不要超出制造商给定负载、速度或温度范围"),
    ),
    "reposition-within-safe-use": _ExecutionGuide(
        ("确认移动不会遮挡散热、通道或安全装置。", "标记原位置和朝向。"),
        "小幅调整后关键指标呈稳定、可重复的方向变化，并且使用条件仍符合说明。",
        "若变化随机或引入新问题，恢复原位置；不要连续试探到安全边界之外。",
        "10–20 分钟",
        ("可移除位置标记", "手机"),
        ("不要堵塞通风", "不要在不稳家具、高处或潮湿区域操作"),
    ),
    "improve-light-path": _ExecutionGuide(
        ("识别手机感光器与需要照亮的实际工作面。", "记录当前灯具、窗帘和遮挡物状态。"),
        "在相同时间和手机朝向下，照度中位数提高且 IQR 或变异系数没有明显恶化。",
        "若照度没有提高，检查感光器是否被手或手机壳遮挡；仍无改善时核对灯具说明或请专业人员检查。",
        "5–15 分钟",
        ("手机", "干燥软布", "可选的位置标记"),
        ("不要拆灯具", "不要直视强光", "不要登上不稳家具"),
    ),
    "reduce-acoustic-exposure": _ExecutionGuide(
        ("保持同一音频幅值实验、手机距离和朝向。", "先关闭与问题无关的电视、风扇或音乐。"),
        "平均相对声级或峰值在相同工况下下降，并且主观异响同步减弱。",
        "若无改善，逐一隔离可控声源；出现摩擦、金属撞击或异常升温时停止设备并联系专业人员。",
        "5–15 分钟",
        ("手机", "phyphox 音频幅值实验"),
        ("不要把相对 dB 与法规限值比较", "不要为测试而长时间暴露在高声级下"),
    ),
    "reduce-magnetic-interference": _ExecutionGuide(
        ("确认没有佩戴或使用对磁场敏感的医疗设备。", "标记手机位置与方向并记录设备开关状态。"),
        "移除无关磁源后，磁场模长波动或开关状态差异下降并可重复。",
        "若差异仍与设备状态同步，只保留相对扰动结论，不拆机追查内部电气原因。",
        "10–15 分钟",
        ("手机", "非磁性位置标记"),
        ("不要靠近裸露线路", "不要用强磁体主动刺激设备"),
    ),
    "clear-sensor-path": _ExecutionGuide(
        ("查看手机说明或用已知动作确认传感器位置。", "锁定手机位置和屏幕朝向。"),
        "完成远—近—远或遮挡—无遮挡动作时，输出层级或照度变化稳定重复。",
        "若仍无响应，确认 phyphox 实验通道是否正确；不同机型可能不提供连续距离值。",
        "5–10 分钟",
        ("干燥软布", "手机说明"),
        ("不要用液体直接擦拭开孔", "不要把二态输出解释成厘米距离"),
    ),
    "verify-environmental-context": _ExecutionGuide(
        ("记录测量时间、地点以及可能影响信号的环境状态。", "尽量在 10 分钟内完成参考与对照。"),
        "控制环境后差异仍超过记录波动，并在重复时保持相同方向。",
        "若差异随天气、定位精度或背景光变化而消失，将其归为环境混杂并改用更直接的传感器。",
        "10–20 分钟",
        ("手机", "时间与环境记录"),
        ("不要把短时相对气压当绝对海拔", "不要在危险路线中为定位测试分心操作手机"),
    ),
    "isolate-operating-source": _ExecutionGuide(
        ("列出所有可通过正常控制开关的候选来源。", "先保持手机位置、房门和背景设备不变。"),
        "某一来源关闭时指标稳定下降、重新开启时恢复，才把它排在主要来源前列。",
        "若没有单一来源满足开—关复现，考虑叠加来源或传播路径，并停止无信息增益的重复。",
        "15–30 分钟",
        ("手机", "候选设备清单"),
        ("不要关闭安全关键设备", "不要直接插拔不明电源或线路"),
    ),
    "check-manufacturer-guidance": _ExecutionGuide(
        ("记录品牌、完整型号和当前故障现象。", "优先打开制造商官网或随附说明。"),
        "确认建议动作属于用户维护范围，并核对现象是否落入官方给出的正常或异常条件。",
        "若说明没有覆盖、要求拆机或现象持续，停止通用尝试并联系官方售后。",
        "10–20 分钟",
        ("型号铭牌照片", "制造商官方说明"),
        ("不要使用来源不明的拆机教程替代官方安全要求",),
    ),
    "request-professional-inspection": _ExecutionGuide(
        ("停止触发异常的工况并保存 PocketLab 报告。", "整理型号、发生条件、时间和已尝试的安全动作。"),
        "由有资质人员确认故障范围、维修方式和恢复使用条件。",
        "若无法立即获得服务且存在安全迹象，保持停用并按制造商或当地应急指引处理。",
        "预约约 10 分钟；现场时间由问题决定",
        ("诊断报告", "型号与购买信息", "现场照片（安全时）"),
        ("不要在等待期间反复触发异常", "不要自行接触市电、燃气、制冷剂或承重结构"),
    ),
}


_SENSOR_RELEVANT_ACTIONS: dict[str, tuple[DiagnosticActionId, ...]] = {
    "accelerometer": (
        "remove-external-contact",
        "stabilize-external-support",
        "reduce-user-adjustable-source",
    ),
    "gyroscope": ("stabilize-external-support", "reposition-within-safe-use"),
    "magnetometer": ("reduce-magnetic-interference", "isolate-operating-source"),
    "light": ("improve-light-path", "clear-sensor-path"),
    "pressure": ("verify-environmental-context",),
    "proximity": ("clear-sensor-path",),
    "microphone": ("reduce-acoustic-exposure", "isolate-operating-source"),
    "location": ("verify-environmental-context", "repeat-controlled-measurement"),
}

_INCONCLUSIVE_ACTIONS: set[DiagnosticActionId] = {
    "preserve-and-observe",
    "repeat-controlled-measurement",
    "check-manufacturer-guidance",
    "request-professional-inspection",
}

_GENERIC_ACTIONS: set[DiagnosticActionId] = {
    "preserve-and-observe",
    "repeat-controlled-measurement",
    "check-manufacturer-guidance",
    "request-professional-inspection",
}

_PRIMARY_GENERIC_ACTIONS: set[DiagnosticActionId] = {
    "preserve-and-observe",
    "repeat-controlled-measurement",
}


def _direct_resolution_actions(
    case: DiagnosticCase,
    sensor: str | None,
) -> list[DiagnosticActionId]:
    """Map an evidence-supported leading mechanism to safe user remediation.

    Provider timeouts must not erase an obvious, server-verifiable resolution.
    These rules intentionally require leader-specific wording; the full case
    text only supplies appliance context.
    """

    leader = next(
        (
            item
            for item in case.hypotheses
            if item.hypothesis_id == case.termination_vector.leading_hypothesis_id
        ),
        None,
    )
    if leader is None:
        return []
    # Critical predictions often mention a competing factor in a negated
    # contrast (for example, "external contact does not change the result").
    # Remediation must therefore follow the supported mechanism statement and
    # rationale, not keyword hits inside the prediction text.
    leader_text = f"{leader.statement} {leader.rationale}".casefold()
    case_text = f"{case.title} {case.problem_statement} {case.context}".casefold()

    appliance_is_washer = any(
        token in case_text for token in ("洗衣机", "洗衣", "脱水", "甩干", "滚筒")
    )
    leader_is_load_imbalance = any(
        token in leader_text
        for token in ("偏载", "衣物不均", "分布不均", "不平衡负载", "负载不平衡", "偏心")
    )
    if appliance_is_washer and leader_is_load_imbalance and sensor in {None, "accelerometer"}:
        return ["redistribute-balanced-load"]

    mechanism_rules: tuple[
        tuple[tuple[str, ...], tuple[str, ...], DiagnosticActionId], ...
    ] = (
        (("accelerometer", "gyroscope"), ("支脚", "支撑不稳", "未调平", "不水平"), "stabilize-external-support"),
        (("accelerometer", "gyroscope", "microphone"), ("外部接触", "硬接触", "杂物碰撞", "线缆碰撞"), "remove-external-contact"),
        (("light",), ("遮挡", "入射光", "光路"), "improve-light-path"),
        (("magnetometer",), ("磁干扰", "磁性手机壳", "磁吸"), "reduce-magnetic-interference"),
        (("light", "proximity"), ("保护膜", "传感器被遮", "感光器被遮", "传感器表面"), "clear-sensor-path"),
        (("microphone",), ("多个声源", "背景声源", "其他声源"), "isolate-operating-source"),
    )
    return [
        action_id
        for sensors, tokens, action_id in mechanism_rules
        if sensor in sensors and any(token in leader_text for token in tokens)
    ][:2]


def finalization_action_candidates(
    case: DiagnosticCase,
    *,
    conclusive: bool,
) -> list[dict[str, object]]:
    """Expose only actions the model may turn into user-facing instructions."""

    facts = [fact for evidence in case.evidence for fact in evidence.facts]
    sensor = facts[-1].sensor if facts else None
    direct_actions = _direct_resolution_actions(case, sensor) if conclusive else []
    allowed_ids = set(_INCONCLUSIVE_ACTIONS)
    if conclusive:
        allowed_ids |= set(direct_actions)
    ordered_ids = [*direct_actions, *sorted(allowed_ids - set(direct_actions))]
    return [
        {
            "action_id": action_id,
            "action_role": _SAFE_ACTIONS[action_id].action_role,
            "title": _SAFE_ACTIONS[action_id].title,
            "risk_level": _SAFE_ACTIONS[action_id].risk_level,
            "purpose": _SAFE_ACTIONS[action_id].rationale,
            "mandatory_safety_notes": list(_SAFE_ACTIONS[action_id].safety_notes),
        }
        for action_id in ordered_ids
    ]


_UNSAFE_POSITIVE_ACTION_PATTERNS = (
    "带电打开",
    "带电拆",
    "拆机",
    "拆开外壳",
    "绕过门锁",
    "绕过安全",
    "短接",
    "接触裸露导线",
    "移除安全保护",
    "禁用安全保护",
    "用手固定正在转动",
    "用手按住正在转动",
)


def _assert_safe_model_action_text(action: Any) -> None:
    positive_text = " ".join(
        [
            action.title,
            action.rationale,
            *action.preparation,
            *action.steps,
            action.expected_result,
            action.how_to_verify,
            action.if_not_improved,
            *action.tools_needed,
        ]
    ).casefold()
    if any(pattern.casefold() in positive_text for pattern in _UNSAFE_POSITIVE_ACTION_PATTERNS):
        raise ValueError("unsafe model-authored action text")


def build_model_finalization(
    case: DiagnosticCase,
    proposal: Any,
    *,
    runtime: Mapping[str, object],
) -> DiagnosticFinalizationResult:
    """Validate model prose against evidence and server safety, then adopt it.

    The model owns the explanatory and execution prose on this success path.
    The server owns case identity, fact lineage, termination, action IDs, risk
    roles, mandatory prohibitions and escalation conditions.
    """

    report = case.final_report
    if report is None:
        raise ValueError("diagnostic finalization requires a finished report")
    conclusive = report.outcome == "completed_with_conclusion"
    if proposal.case_id != case.case_id:
        raise ValueError("finalization case identity mismatch")
    expected_leader = case.termination_vector.leading_hypothesis_id
    if proposal.leading_hypothesis_id != expected_leader:
        raise ValueError("finalization leading hypothesis mismatch")

    facts = [fact for evidence in case.evidence for fact in evidence.facts]
    fact_by_id = {fact.fact_id: fact for fact in facts}
    unknown_fact_ids = set(proposal.source_fact_ids) - set(fact_by_id)
    if unknown_fact_ids:
        raise ValueError("finalization references unknown evidence facts")
    selected_facts = [fact_by_id[fact_id] for fact_id in proposal.source_fact_ids]
    if not selected_facts:
        raise ValueError("finalization requires at least one evidence fact")
    latest_fact = selected_facts[-1]

    direct_actions = (
        _direct_resolution_actions(case, latest_fact.sensor)
        if conclusive
        else []
    )
    candidate_ids = {
        item["action_id"]
        for item in finalization_action_candidates(case, conclusive=conclusive)
    }
    requested_ids = [action.action_id for action in proposal.actions]
    disallowed = set(requested_ids) - candidate_ids
    if disallowed:
        raise ValueError("model action ID is not allowed for this conclusion")
    if conclusive and direct_actions and requested_ids[0] not in direct_actions:
        raise ValueError("model primary action is not allowed for the supported mechanism")
    if conclusive and not direct_actions:
        first_role = _SAFE_ACTIONS[requested_ids[0]].action_role
        if first_role == "verify":
            raise ValueError("conclusive report requires a resolution or escalation first")

    rendered_actions: list[DiagnosticSolutionAction] = []
    for action in proposal.actions:
        _assert_safe_model_action_text(action)
        safe = _SAFE_ACTIONS[action.action_id]
        guide = _EXECUTION_GUIDES[action.action_id]
        preparation = list(dict.fromkeys([*guide.preparation, *action.preparation]))
        do_not_do = list(dict.fromkeys([*action.do_not_do, *guide.do_not_do]))
        rendered_actions.append(
            DiagnosticSolutionAction(
                action_id=action.action_id,
                action_role=safe.action_role,
                title=action.title,
                rationale=action.rationale,
                preparation=preparation,
                steps=list(action.steps),
                expected_result=action.expected_result,
                how_to_verify=action.how_to_verify,
                if_not_improved=action.if_not_improved,
                estimated_time=action.estimated_time,
                tools_needed=list(action.tools_needed),
                do_not_do=do_not_do,
                risk_level=safe.risk_level,
                safety_notes=list(safe.safety_notes),
            )
        )

    existing_optional_retest = (
        report.solution_plan.optional_retest
        if report.solution_plan is not None
        else _optional_retest(case)
    )
    plan = DiagnosticSolutionPlan(
        basis="evidence_supported" if conclusive else "inconclusive_safe_next_steps",
        summary=proposal.summary,
        evidence_summary=proposal.evidence_summary,
        first_action_reason=proposal.first_action_reason,
        actions=rendered_actions,
        escalation_conditions=list(_STOP_AND_ESCALATE),
        optional_retest=existing_optional_retest,
    )
    adopted = report.model_copy(
        deep=True,
        update={
            "conclusion": (
                f"{proposal.answer_headline}。{proposal.mechanism_explanation}"
            ).strip(),
            "answer_headline": proposal.answer_headline,
            "mechanism_explanation": proposal.mechanism_explanation,
            "source_fact_ids": list(proposal.source_fact_ids),
            "user_takeaway": proposal.user_takeaway,
            "confidence_explanation": proposal.confidence_explanation,
            "solution_plan": plan,
            "finalization_source": "model_generated",
            "finalization_model": str(runtime.get("model") or "unknown-model"),
            "finalization_transport": str(
                runtime.get("transport") or "validated_json_chat"
            ),
            "finalization_model_requests": max(1, int(runtime.get("attempts") or 1)),
            "finalization_elapsed_ms": max(0, int(runtime.get("elapsed_ms") or 0)),
            "finalization_fallback_reason": None,
            "finalization_retryable": False,
        },
    )
    return DiagnosticFinalizationResult(report=adopted)


def build_fallback_finalization(
    case: DiagnosticCase,
    *,
    fallback_reason: str,
    model: str,
    model_requests: int,
    elapsed_ms: int,
) -> DiagnosticFinalReport:
    """Label the server playbook honestly when final model generation fails."""

    if case.final_report is None:
        raise ValueError("diagnostic fallback finalization requires a finished report")
    conclusive = case.final_report.outcome == "completed_with_conclusion"
    plan = build_solution_plan(case, conclusive=conclusive)
    plan.summary = (
        "模型本次没有生成可安全采纳的完整解决方案。以下内容是服务端安全兜底，"
        "用于避免丢失已有证据；它不是完整的基模诊断结果，可直接重试终局生成。 "
        + plan.summary
    )
    return case.final_report.model_copy(
        deep=True,
        update={
            "solution_plan": plan,
            "finalization_source": "deterministic_fallback",
            "finalization_model": model,
            "finalization_transport": "deterministic_fallback",
            "finalization_model_requests": max(0, model_requests),
            "finalization_elapsed_ms": max(0, elapsed_ms),
            "finalization_fallback_reason": fallback_reason[:500],
            "finalization_retryable": True,
        },
    )


def build_solution_plan(
    case: DiagnosticCase,
    *,
    conclusive: bool,
) -> DiagnosticSolutionPlan:
    """Render model-ranked choices through a server-owned reversible action allowlist."""

    leading_hypothesis_id = case.termination_vector.leading_hypothesis_id
    reasoning = next(
        (
            item.reasoning_receipt
            for item in reversed(case.evidence)
            if item.reasoning_receipt is not None
            and (
                not conclusive
                or leading_hypothesis_id is None
                or item.reasoning_receipt.ranked_hypothesis_ids[0]
                == leading_hypothesis_id
            )
        ),
        None,
    )
    requested = list(reasoning.recommended_action_ids) if reasoning else []
    if not conclusive:
        requested = [item for item in requested if item in _INCONCLUSIVE_ACTIONS]
    facts = [fact for evidence in case.evidence for fact in evidence.facts]
    referenced_fact_ids = set(reasoning.source_fact_ids) if reasoning else set()
    latest_fact = next(
        (fact for fact in reversed(facts) if fact.fact_id in referenced_fact_ids),
        facts[-1] if facts else None,
    )
    direct_actions = (
        _direct_resolution_actions(case, latest_fact.sensor if latest_fact else None)
        if conclusive
        else []
    )
    if conclusive:
        relevant = (
            _SENSOR_RELEVANT_ACTIONS.get(latest_fact.sensor, ())
            if latest_fact is not None
            else ()
        )
        allowed = _GENERIC_ACTIONS | set(relevant) | set(direct_actions)
        requested = [item for item in requested if item in allowed]
        if direct_actions:
            requested = [
                *direct_actions,
                *(
                    item
                    for item in requested
                    if item not in direct_actions and item not in _PRIMARY_GENERIC_ACTIONS
                ),
            ]
        if any(item not in _GENERIC_ACTIONS for item in requested):
            requested = [item for item in requested if item not in _PRIMARY_GENERIC_ACTIONS]
    if not requested:
        requested = ["preserve-and-observe", "check-manufacturer-guidance"]
    requested = list(dict.fromkeys(requested))[:3]
    model_rationale = (
        reasoning.solution_rationale.strip()
        if reasoning is not None
        and reasoning.transport != "deterministic_fallback"
        and (
            not direct_actions
            or any(item in direct_actions for item in reasoning.recommended_action_ids)
        )
        else ""
    )
    actions = [_render_action(action_id, latest_fact) for action_id in requested]
    leader = next(
        (
            item.statement
            for item in case.hypotheses
            if item.hypothesis_id == case.termination_vector.leading_hypothesis_id
        ),
        "当前最可能解释",
    )
    has_direct_resolution = bool(actions and actions[0].action_role == "resolve")
    if conclusive and has_direct_resolution:
        summary = (
            f"当前证据支持“{leader}”。先给出可直接处理主因的安全动作；复测只用于确认效果，"
            "不作为获得解决方案的前置条件。"
        )
    elif conclusive:
        summary = (
            f"当前证据支持“{leader}”，但服务端动作目录中没有与它匹配的安全用户处理动作；"
            "因此只保留验证、官方指引或专业升级，不能把通用复测冒充解决方案。"
        )
    else:
        summary = "当前区分度仍有限；只保留观察、受控复测、官方指引和专业升级动作。"
    if model_rationale:
        summary = f"{summary} Agent 的证据理由：{model_rationale}"
    evidence_summary = _format_fact_evidence(latest_fact)
    return DiagnosticSolutionPlan(
        basis="evidence_supported" if conclusive else "inconclusive_safe_next_steps",
        summary=summary,
        evidence_summary=evidence_summary,
        first_action_reason=(
            f"先执行“{actions[0].title}”，因为它是在安全边界内直接处理当前证据支持主因的动作。"
            if has_direct_resolution
            else f"先执行“{actions[0].title}”，因为当前尚无可安全签发的直接处理动作。"
            if actions
            else ""
        ),
        actions=actions,
        escalation_conditions=_STOP_AND_ESCALATE,
        optional_retest=_optional_retest(case),
    )


def _render_action(
    action_id: DiagnosticActionId,
    latest_fact,
) -> DiagnosticSolutionAction:
    item = _SAFE_ACTIONS[action_id]
    execution = _EXECUTION_GUIDES[action_id]
    steps = list(item.steps)
    if item.action_role != "escalate":
        steps.extend(
            (
                "完成处理或验证后，把手机恢复到报告中的同一测点和方向。",
                "如需确认效果，保持设备工况与采样时长不变，把结果与处理前记录并排比较。",
            )
        )
    return DiagnosticSolutionAction(
        action_id=action_id,
        action_role=item.action_role,
        title=item.title,
        rationale=item.rationale,
        preparation=list(execution.preparation),
        steps=steps,
        expected_result=item.expected_result,
        how_to_verify=execution.verification,
        if_not_improved=execution.if_not_improved,
        estimated_time=execution.estimated_time,
        tools_needed=list(execution.tools_needed),
        do_not_do=list(execution.do_not_do),
        risk_level=item.risk_level,
        safety_notes=list(item.safety_notes),
    )


def _format_fact_evidence(latest_fact) -> str:
    if latest_fact is None:
        return "当前没有可显示的数值事实；行动仅作为安全的下一步。"
    current = f"{latest_fact.metric_label}为 {latest_fact.value:.4g} {latest_fact.metric_unit}".strip()
    if latest_fact.baseline_value is None:
        return f"本轮依据：{current}，证据质量为{latest_fact.quality}。"
    direction = {
        "increase": "上升",
        "decrease": "下降",
        "within_repeatability": "未分辨到稳定变化",
    }.get(latest_fact.relation, "变化")
    return (
        f"本轮依据：{latest_fact.metric_label}从 {latest_fact.baseline_value:.4g} "
        f"{latest_fact.metric_unit} 到 {latest_fact.value:.4g} {latest_fact.metric_unit}，"
        f"判定为{direction}，证据质量为{latest_fact.quality}。"
    )


def audit_action_catalog() -> tuple[bool, list[str]]:
    """Return whether every allowlisted action has complete user execution guidance."""

    missing: list[str] = []
    schema_actions = set(get_args(DiagnosticActionId))
    catalog_actions = set(_SAFE_ACTIONS)
    for action_id in sorted(schema_actions - catalog_actions):
        missing.append(f"{action_id}:missing safe action")
    for action_id in sorted(catalog_actions - schema_actions):
        missing.append(f"{action_id}:not in DiagnosticActionId")
    for action_id, action in _SAFE_ACTIONS.items():
        guide = _EXECUTION_GUIDES.get(action_id)
        if guide is None:
            missing.append(f"{action_id}:missing execution guide")
            continue
        if len(guide.preparation) < 2 or not guide.verification or not guide.if_not_improved:
            missing.append(f"{action_id}:incomplete execution guide")
        if not guide.tools_needed or not guide.do_not_do:
            missing.append(f"{action_id}:missing tools or do-not-do boundary")
        if action.action_role not in {"resolve", "verify", "escalate"}:
            missing.append(f"{action_id}:invalid action role")
    for action_id in sorted(set(_EXECUTION_GUIDES) - catalog_actions):
        missing.append(f"{action_id}:guide has no safe action")
    return not missing, missing


def _optional_retest(case: DiagnosticCase) -> DiagnosticOptionalRetest:
    latest_fact = next(
        (fact for evidence in reversed(case.evidence) for fact in reversed(evidence.facts)),
        None,
    )
    metric = (
        f"{latest_fact.metric_label}（{latest_fact.metric_unit or '无量纲'}）"
        if latest_fact
        else "本案例的关键指标"
    )
    return DiagnosticOptionalRetest(
        title="可选：处理后做一次同条件复测",
        purpose="验证所选动作是否真正改变问题；复测不阻止当前案例结束。",
        instruction=(
            "每次只执行一个建议，把手机放回原测点并保持工况、方向、采样时长和"
            "phyphox 实验一致，然后重新记录。"
        ),
        controlled_variables=["手机与测点", "设备工况", "采样时长", "传感器与通道"],
        success_criteria=[
            f"{metric}沿报告预测方向改变。",
            "变化可重复，且没有触发新的安全停止条件。",
        ],
        result_use="达到标准可记为动作得到支持；未达到时原报告保持结束，并重新评估竞争机制。",
    )
