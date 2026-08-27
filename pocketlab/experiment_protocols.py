from __future__ import annotations

from pocketlab.investigation_models import (
    ExperimentParameterDefinition,
    ExperimentProtocol,
    ExperimentToolDefinition,
)

_LIGHT_DISTANCE_LAW_V1 = ExperimentProtocol(
    protocol_id="light-distance-law.v1",
    protocol_version="1.0.0",
    mode="explore",
    title="灯光距离衰减",
    primary_sensor="light",
    required_analyzer_id="pocketlab.light.v2",
    measurement_metric_key="median_illuminance_lx",
    measurement_metric_unit="lx",
    allowed_tools=[
        ExperimentToolDefinition(
            tool_id="sensor_analysis.light.v2",
            version="2.0.0",
            allowed_sensors=["light"],
            input_metric_keys=["illuminance"],
            output_metric_keys=[
                "median_illuminance_lx",
                "minimum_illuminance_lx",
                "maximum_illuminance_lx",
                "illuminance_iqr_lx",
                "coefficient_of_variation_ratio",
                "minimum_quantization_step_lx",
                "upper_plateau_fraction",
            ],
            deterministic=True,
            read_only=True,
            maturity="analysis_ready",
        ),
        ExperimentToolDefinition(
            tool_id="aggregate_light_conditions",
            version="1.0.0",
            allowed_sensors=["light"],
            input_metric_keys=[
                "background_lx",
                "distance_m",
                "median_illuminance_lx",
            ],
            output_metric_keys=[
                "distance_m",
                "median_net_illuminance_lx",
                "mad_net_illuminance_lx",
                "repeat_count",
            ],
            deterministic=True,
            read_only=True,
            maturity="analysis_ready",
        ),
        ExperimentToolDefinition(
            tool_id="fit_light_distance_decay",
            version="1.0.0",
            allowed_sensors=["light"],
            input_metric_keys=[
                "distance_m",
                "median_net_illuminance_lx",
                "mad_net_illuminance_lx",
                "repeat_count",
            ],
            output_metric_keys=[
                "condition_count",
                "minimum_distance_m",
                "maximum_distance_m",
                "distance_span_ratio",
                "free_exponent",
                "free_exponent_ci95_low",
                "free_exponent_ci95_high",
                "free_model_scale",
                "free_model_r_squared",
                "inverse_square_k_lx_m2",
                "inverse_square_relative_rmse",
            ],
            deterministic=True,
            read_only=True,
            maturity="analysis_ready",
        ),
        ExperimentToolDefinition(
            tool_id="select_next_design_point",
            version="1.0.0",
            allowed_sensors=["light"],
            input_metric_keys=[
                "distance_m",
                "median_net_illuminance_lx",
                "upper_plateau_fraction",
                "distance_span_ratio",
                "repeat_count",
            ],
            output_metric_keys=[
                "next_distance_m",
                "design_reason_code",
            ],
            deterministic=True,
            read_only=True,
            maturity="analysis_ready",
        ),
        ExperimentToolDefinition(
            tool_id="sample_light_fit_series",
            version="1.0.0",
            allowed_sensors=["light"],
            input_metric_keys=[
                "minimum_distance_m",
                "maximum_distance_m",
                "free_exponent",
                "free_model_scale",
                "inverse_square_k_lx_m2",
            ],
            output_metric_keys=[
                "distance_m",
                "free_model_net_illuminance_lx",
                "inverse_square_net_illuminance_lx",
            ],
            deterministic=True,
            read_only=True,
            maturity="analysis_ready",
        ),
    ],
    max_measurements=14,
    max_corrections=3,
    parameters=[
        ExperimentParameterDefinition(
            key="distance_m",
            value_type="number",
            unit="m",
            minimum=0.1,
            maximum=4.0,
            recommended=0.5,
            description="光源参考点到手机光线传感器受光面的测量距离。",
        )
    ],
    controls=[
        "保持同一光源和光源输出设置。",
        "保持光源与手机光线传感器的相对朝向。",
        "保持同一手机、同一 phyphox light 实验和同一测量通道。",
        "尽量保持环境光、遮挡和反射条件稳定。",
        "每次等待读数稳定后再采集，并按同一方式记录距离。",
    ],
    safety_notes=[
        "不要直视或近距离测量太阳、激光及高温强光源。",
        "固定手机和光源，避免移动线缆或测距过程造成绊倒、跌落和烫伤。",
        "phyphox 远程访问仅在可信局域网中短时开启，完成后及时关闭。",
    ],
    claim_boundaries=[
        "没有可追溯校准链时只作同一手机内的相对照度比较，不宣称绝对光度准确度。",
        "结论只适用于已测试的光源、几何、手机、环境和距离范围。",
        "与平方反比近似一致不等于普适证明，偏离也不等于否定物理定律。",
        "有限尺寸光源、近场、入射角、反射、量化和饱和都可能改变观测关系。",
        "数学 oracle 和合成安全样本不得冒充真实手机数据或市场验证。",
    ],
    market_validated=False,
)


_PROTOCOLS = (_LIGHT_DISTANCE_LAW_V1,)
_PROTOCOL_INDEX = {
    (protocol.protocol_id, protocol.protocol_version): protocol for protocol in _PROTOCOLS
}

if len(_PROTOCOL_INDEX) != len(_PROTOCOLS):
    raise RuntimeError("experiment protocol id/version pairs must be unique")


def list_experiment_protocols() -> list[ExperimentProtocol]:
    """Return isolated protocol copies in stable catalog order."""

    return [protocol.model_copy(deep=True) for protocol in _PROTOCOLS]


def get_experiment_protocol(protocol_id: str, protocol_version: str) -> ExperimentProtocol:
    """Return one exact protocol version without fuzzy or free-text matching."""

    try:
        protocol = _PROTOCOL_INDEX[(protocol_id, protocol_version)]
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment protocol: id={protocol_id!r}, version={protocol_version!r}"
        ) from exc
    return protocol.model_copy(deep=True)
