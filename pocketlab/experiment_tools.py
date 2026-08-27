from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_MEASUREMENTS = 10_000
_MAX_CONDITIONS = 1_000
_MAX_SERIES_POINTS = 200
_MAX_DISTANCE_M = 1_000_000.0
_MAX_ILLUMINANCE_LX = 1_000_000_000_000.0
_MAX_WITHIN_CONDITION_DISTANCE_SPAN_RATIO = 0.05
_MIN_DISTANCE_SPAN_RATIO = 3.0

# These are deliberately product policy thresholds, not physical constants. They must be
# recalibrated from real-phone replay data before the light experiment can become Agent-ready.
_MIN_FREE_MODEL_R_SQUARED = 0.90
_MAX_INVERSE_SQUARE_RELATIVE_RMSE = 0.15
_MAX_DECISIVE_EXPONENT_CI_WIDTH = 0.75

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LightMeasurement(_StrictFrozenModel):
    evidence_id: str = Field(min_length=1, max_length=80)
    condition_id: str = Field(min_length=1, max_length=80)
    distance_m: float = Field(gt=0.0, le=_MAX_DISTANCE_M, allow_inf_nan=False)
    observed_illuminance_lx: float = Field(
        ge=0.0,
        le=_MAX_ILLUMINANCE_LX,
        allow_inf_nan=False,
    )

    @field_validator("evidence_id", "condition_id")
    @classmethod
    def identifiers_are_machine_safe(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError(
                "identifiers must start with an ASCII alphanumeric character and contain "
                "only ASCII letters, digits, dot, underscore, colon or hyphen"
            )
        return value


class LightConditionAggregate(_StrictFrozenModel):
    condition_id: str = Field(min_length=1, max_length=80)
    distance_m: float = Field(gt=0.0, le=_MAX_DISTANCE_M, allow_inf_nan=False)
    median_net_illuminance_lx: float = Field(
        gt=0.0,
        le=_MAX_ILLUMINANCE_LX,
        allow_inf_nan=False,
    )
    mad_net_illuminance_lx: float = Field(
        ge=0.0,
        le=_MAX_ILLUMINANCE_LX,
        allow_inf_nan=False,
    )
    repeat_count: int = Field(ge=1, le=_MAX_MEASUREMENTS)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=_MAX_MEASUREMENTS)

    @field_validator("condition_id")
    @classmethod
    def condition_id_is_machine_safe(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("condition_id is not a valid machine identifier")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_valid_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _ID_PATTERN.fullmatch(value) for value in values):
            raise ValueError("evidence_ids contain an invalid machine identifier")
        if len(set(values)) != len(values):
            raise ValueError("evidence_ids must be unique")
        return values

    @model_validator(mode="after")
    def repeat_count_matches_evidence(self) -> LightConditionAggregate:
        if self.repeat_count != len(self.evidence_ids):
            raise ValueError("repeat_count must equal the number of evidence_ids")
        return self


LightFitClassification = Literal[
    "consistent_with_inverse_square",
    "not_supported_in_tested_range",
    "inconclusive",
]


class LightDecayFit(_StrictFrozenModel):
    condition_count: int = Field(ge=4, le=_MAX_CONDITIONS)
    minimum_distance_m: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_distance_m: float = Field(gt=0.0, allow_inf_nan=False)
    distance_span_ratio: float = Field(ge=_MIN_DISTANCE_SPAN_RATIO, allow_inf_nan=False)
    free_exponent: float = Field(allow_inf_nan=False)
    free_exponent_ci95_low: float = Field(allow_inf_nan=False)
    free_exponent_ci95_high: float = Field(allow_inf_nan=False)
    free_model_scale: float = Field(gt=0.0, allow_inf_nan=False)
    free_model_r_squared: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    inverse_square_k_lx_m2: float = Field(gt=0.0, allow_inf_nan=False)
    inverse_square_relative_rmse: float = Field(ge=0.0, allow_inf_nan=False)
    classification: LightFitClassification
    reasons: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def interval_and_distance_bounds_are_ordered(self) -> LightDecayFit:
        if self.maximum_distance_m <= self.minimum_distance_m:
            raise ValueError("maximum_distance_m must exceed minimum_distance_m")
        if self.free_exponent_ci95_high < self.free_exponent_ci95_low:
            raise ValueError("exponent confidence interval is reversed")
        return self


def aggregate_light_conditions(
    background_lx: float,
    measurements: Sequence[LightMeasurement],
    min_repeats: int = 2,
) -> list[LightConditionAggregate]:
    """Aggregate repeated light measurements after a caller-supplied background control.

    Distance consistency is evaluated as ``(max - min) / median <= 5%`` within each
    condition. Every individual background-subtracted value must be positive; silently
    discarding an invalid repeat would overstate the quality of the condition.
    """

    background = _finite_bounded_number(
        background_lx,
        name="background_lx",
        minimum=0.0,
        maximum=_MAX_ILLUMINANCE_LX,
    )
    if isinstance(min_repeats, bool) or not isinstance(min_repeats, int):
        raise TypeError("min_repeats must be an integer")
    if not 1 <= min_repeats <= _MAX_MEASUREMENTS:
        raise ValueError(f"min_repeats must be between 1 and {_MAX_MEASUREMENTS}")
    if len(measurements) > _MAX_MEASUREMENTS:
        raise ValueError(f"at most {_MAX_MEASUREMENTS} light measurements are accepted")
    if not measurements:
        raise ValueError("at least one light measurement is required")
    if any(not isinstance(item, LightMeasurement) for item in measurements):
        raise TypeError("measurements must contain only LightMeasurement objects")

    evidence_ids = [item.evidence_id for item in measurements]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence_id values must be unique across measurements")

    grouped: dict[str, list[LightMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.condition_id].append(measurement)
    if len(grouped) > _MAX_CONDITIONS:
        raise ValueError(f"at most {_MAX_CONDITIONS} light conditions are accepted")

    aggregates: list[LightConditionAggregate] = []
    for condition_id, members in grouped.items():
        if len(members) < min_repeats:
            raise ValueError(
                f"condition {condition_id!r} requires at least {min_repeats} repeats; "
                f"received {len(members)}"
            )
        distances = np.asarray([item.distance_m for item in members], dtype=np.float64)
        representative_distance = float(np.median(distances))
        distance_span_ratio = float(np.ptp(distances) / representative_distance)
        if distance_span_ratio > _MAX_WITHIN_CONDITION_DISTANCE_SPAN_RATIO + 1e-12:
            raise ValueError(
                f"condition {condition_id!r} has distances differing by more than 5%"
            )

        net_values = np.asarray(
            [item.observed_illuminance_lx - background for item in members],
            dtype=np.float64,
        )
        if not np.isfinite(net_values).all() or np.any(net_values <= 0.0):
            raise ValueError(
                f"condition {condition_id!r} must have positive background-subtracted "
                "illuminance for every repeat"
            )
        median_net = float(np.median(net_values))
        mad_net = float(np.median(np.abs(net_values - median_net)))
        aggregates.append(
            LightConditionAggregate(
                condition_id=condition_id,
                distance_m=representative_distance,
                median_net_illuminance_lx=median_net,
                mad_net_illuminance_lx=mad_net,
                repeat_count=len(members),
                evidence_ids=tuple(sorted(item.evidence_id for item in members)),
            )
        )

    return sorted(aggregates, key=lambda item: (item.distance_m, item.condition_id))


def fit_light_distance_decay(
    aggregates: Sequence[LightConditionAggregate],
) -> LightDecayFit:
    """Fit ``net illuminance = K * distance ** (-n)`` and an n=2 reference model.

    Classification uses explicit first-version engineering thresholds:

    * both decisive classifications require free-model R-squared >= 0.90;
    * inverse-square consistency additionally requires its 95% exponent interval to contain
      2 and fixed-n=2 RMS relative error <= 15%;
    * non-support requires the interval to exclude 2 and have width <= 0.75;
    * every other valid fit is inconclusive.

    These thresholds are product policy for the deterministic offline harness, not universal
    physics constants or evidence of real-phone readiness.
    """

    if len(aggregates) > _MAX_CONDITIONS:
        raise ValueError(f"at most {_MAX_CONDITIONS} light conditions are accepted")
    if len(aggregates) < 4:
        raise ValueError("at least four light conditions are required")
    if any(not isinstance(item, LightConditionAggregate) for item in aggregates):
        raise TypeError("aggregates must contain only LightConditionAggregate objects")

    condition_ids = [item.condition_id for item in aggregates]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition_id values must be unique for fitting")
    all_evidence_ids = [evidence_id for item in aggregates for evidence_id in item.evidence_ids]
    if len(set(all_evidence_ids)) != len(all_evidence_ids):
        raise ValueError("evidence_id values must be unique across conditions")

    ordered = sorted(aggregates, key=lambda item: (item.distance_m, item.condition_id))
    distances = np.asarray([item.distance_m for item in ordered], dtype=np.float64)
    net_values = np.asarray(
        [item.median_net_illuminance_lx for item in ordered],
        dtype=np.float64,
    )
    if np.unique(distances).size != len(distances):
        raise ValueError("each fitted condition must use a distinct representative distance")
    distance_span_ratio = float(distances[-1] / distances[0])
    if distance_span_ratio < _MIN_DISTANCE_SPAN_RATIO:
        raise ValueError("light fitting requires a distance span ratio of at least 3")

    log_distance = np.log(distances)
    log_illuminance = np.log(net_values)
    centered_x = log_distance - float(np.mean(log_distance))
    centered_y = log_illuminance - float(np.mean(log_illuminance))
    sum_xx = float(np.dot(centered_x, centered_x))
    if sum_xx <= 0.0:
        raise ValueError("light fitting requires distinct distances")
    slope = float(np.dot(centered_x, centered_y) / sum_xx)
    intercept = float(np.mean(log_illuminance) - slope * np.mean(log_distance))
    fitted_log = intercept + slope * log_distance
    residual_log = log_illuminance - fitted_log
    residual_sum_squares = float(np.dot(residual_log, residual_log))
    total_sum_squares = float(np.dot(centered_y, centered_y))
    r_squared = (
        max(0.0, min(1.0, 1.0 - residual_sum_squares / total_sum_squares))
        if total_sum_squares > 1e-24
        else 0.0
    )

    degrees_of_freedom = len(ordered) - 2
    slope_standard_error = math.sqrt(
        max(0.0, residual_sum_squares / degrees_of_freedom / sum_xx)
    )
    t_critical = _student_t_975(degrees_of_freedom)
    exponent = -slope
    exponent_margin = t_critical * slope_standard_error
    exponent_ci_low = exponent - exponent_margin
    exponent_ci_high = exponent + exponent_margin
    free_scale = _safe_exp(intercept, "free-model scale")

    inverse_square_basis = distances**-2.0
    inverse_square_k = float(
        np.dot(inverse_square_basis, net_values)
        / np.dot(inverse_square_basis, inverse_square_basis)
    )
    inverse_square_prediction = inverse_square_k * inverse_square_basis
    inverse_square_relative_rmse = float(
        np.sqrt(np.mean(((inverse_square_prediction - net_values) / net_values) ** 2))
    )
    _require_finite_fit_values(
        exponent,
        exponent_ci_low,
        exponent_ci_high,
        free_scale,
        r_squared,
        inverse_square_k,
        inverse_square_relative_rmse,
    )

    contains_two = exponent_ci_low <= 2.0 <= exponent_ci_high
    ci_width = exponent_ci_high - exponent_ci_low
    if (
        r_squared >= _MIN_FREE_MODEL_R_SQUARED
        and contains_two
        and inverse_square_relative_rmse <= _MAX_INVERSE_SQUARE_RELATIVE_RMSE
    ):
        classification: LightFitClassification = "consistent_with_inverse_square"
        reasons = (
            "自由幂律拟合 R² 达到首版工程门槛。",
            "幂指数 95% 区间包含 2。",
            "固定 n=2 模型的 RMS 相对误差不超过 15%。",
        )
    elif (
        r_squared >= _MIN_FREE_MODEL_R_SQUARED
        and not contains_two
        and ci_width <= _MAX_DECISIVE_EXPONENT_CI_WIDTH
    ):
        classification = "not_supported_in_tested_range"
        reasons = (
            "自由幂律拟合 R² 达到首版工程门槛。",
            "幂指数 95% 区间排除 2。",
            "幂指数区间宽度不超过 0.75。",
        )
    else:
        classification = "inconclusive"
        reasons = (
            "拟合没有同时满足首版工程门槛；不能据此确认或否定平方反比近似。",
        )

    return LightDecayFit(
        condition_count=len(ordered),
        minimum_distance_m=float(distances[0]),
        maximum_distance_m=float(distances[-1]),
        distance_span_ratio=distance_span_ratio,
        free_exponent=exponent,
        free_exponent_ci95_low=exponent_ci_low,
        free_exponent_ci95_high=exponent_ci_high,
        free_model_scale=free_scale,
        free_model_r_squared=r_squared,
        inverse_square_k_lx_m2=inverse_square_k,
        inverse_square_relative_rmse=inverse_square_relative_rmse,
        classification=classification,
        reasons=reasons,
    )


def sample_light_fit_series(
    fit: LightDecayFit,
    point_count: int = 48,
) -> list[dict[str, float]]:
    """Return bounded, finite numeric points for a controlled visualization renderer."""

    if not isinstance(fit, LightDecayFit):
        raise TypeError("fit must be a LightDecayFit")
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise TypeError("point_count must be an integer")
    if not 2 <= point_count <= _MAX_SERIES_POINTS:
        raise ValueError(f"point_count must be between 2 and {_MAX_SERIES_POINTS}")

    log_distances = np.linspace(
        math.log(fit.minimum_distance_m),
        math.log(fit.maximum_distance_m),
        num=point_count,
        dtype=np.float64,
    )
    points: list[dict[str, float]] = []
    for log_distance in log_distances:
        distance = math.exp(float(log_distance))
        free_value = _safe_exp(
            math.log(fit.free_model_scale) - fit.free_exponent * float(log_distance),
            "sampled free-model illuminance",
        )
        inverse_square_value = fit.inverse_square_k_lx_m2 / (distance * distance)
        _require_finite_fit_values(distance, free_value, inverse_square_value)
        points.append(
            {
                "distance_m": distance,
                "free_model_net_illuminance_lx": free_value,
                "inverse_square_net_illuminance_lx": inverse_square_value,
            }
        )
    return points


def _finite_bounded_number(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _safe_exp(value: float, name: str) -> float:
    try:
        result = math.exp(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the supported numeric range") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} is outside the supported numeric range")
    return result


def _require_finite_fit_values(*values: float) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("light fit produced a non-finite result")


def _student_t_975(degrees_of_freedom: int) -> float:
    # Two-sided 95% Student-t critical values. A normal approximation is sufficient for
    # df > 30 at the precision exposed by this first-version experiment tool.
    values = (
        0.0,
        12.706204736,
        4.30265273,
        3.182446305,
        2.776445105,
        2.570581836,
        2.446911851,
        2.364624252,
        2.306004135,
        2.262157163,
        2.228138852,
        2.20098516,
        2.17881283,
        2.160368656,
        2.144786688,
        2.131449546,
        2.119905299,
        2.109815578,
        2.10092204,
        2.093024054,
        2.085963447,
        2.079613845,
        2.073873068,
        2.06865761,
        2.063898562,
        2.059538553,
        2.055529439,
        2.051830516,
        2.048407142,
        2.045229642,
        2.042272456,
    )
    if degrees_of_freedom < 1:
        raise ValueError("at least three observations are required for a confidence interval")
    if degrees_of_freedom < len(values):
        return values[degrees_of_freedom]
    return 1.959963985
