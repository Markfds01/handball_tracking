from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from enum import Enum
from typing import Dict, List, Tuple


CENTIMETERS_PER_FOOT = Decimal("30.48")


class League(Enum):
    IHF = "ihf"  # International Handball Federation


# presets stored in centimeters
PRESETS_CENTIMETERS: Dict[League, Dict[str, int]] = {
    League.IHF: dict(
        court_width=2000,
        court_length=4000,
        goal_width=300,
        goal_depth=100,  # Standard depth, though rarely tracked on 2D planes
        goal_area_radius=600,       # 6m line
        free_throw_radius=900,      # 9m line
        penalty_mark_distance=700,  # 7m mark
        goal_keeper_line_distance=400, # 4m mark
        substitution_line_distance=450, # 4.5m from center line
        goal_post_offset=150, # 3m goal width / 2
    ),
}


@dataclass
class HandballCourtConfiguration:
    """Configure handball court dimensions for IHF standards.
    Provides court measurements in centimeters or feet with proper unit
    conversion and vertex/edge data for court visualization.

    Args:
        league: The handball league standard to use (default: `League.IHF`).
        measurement_unit: Output unit for measurements
            (`MeasurementUnit.CENTIMETERS` or `MeasurementUnit.FEET`).
    """
    league: League = League.IHF
    measurement_unit: MeasurementUnit = MeasurementUnit.CENTIMETERS

    # internal values in centimeters
    _court_width_in_centimeters: int = field(init=False)
    _court_length_in_centimeters: int = field(init=False)
    _goal_width_in_centimeters: int = field(init=False)
    _goal_area_radius_in_centimeters: int = field(init=False)
    _free_throw_radius_in_centimeters: int = field(init=False)
    _penalty_mark_distance_in_centimeters: int = field(init=False)
    _goal_keeper_line_distance_in_centimeters: int = field(init=False)
    _substitution_line_distance_in_centimeters: int = field(init=False)
    _goal_post_offset_in_centimeters: int = field(init=False)

    def __post_init__(self) -> None:
        preset = PRESETS_CENTIMETERS[self.league]
        self._court_width_in_centimeters = preset["court_width"]
        self._court_length_in_centimeters = preset["court_length"]
        self._goal_width_in_centimeters = preset["goal_width"]
        self._goal_area_radius_in_centimeters = preset["goal_area_radius"]
        self._free_throw_radius_in_centimeters = preset["free_throw_radius"]
        self._penalty_mark_distance_in_centimeters = preset["penalty_mark_distance"]
        self._goal_keeper_line_distance_in_centimeters = preset["goal_keeper_line_distance"]
        self._substitution_line_distance_in_centimeters = preset["substitution_line_distance"]
        self._goal_post_offset_in_centimeters = preset["goal_post_offset"]

    # conversion helpers
    def _to_output_unit_rounded_up(self, value_in_centimeters: float) -> float:
        value = Decimal(str(value_in_centimeters))
        if self.measurement_unit == MeasurementUnit.FEET:
            value = value / CENTIMETERS_PER_FOOT
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))

    # public properties
    @property
    def court_width(self) -> float:
        return self._to_output_unit_rounded_up(self._court_width_in_centimeters)

    @property
    def court_length(self) -> float:
        return self._to_output_unit_rounded_up(self._court_length_in_centimeters)

    @property
    def goal_width(self) -> float:
        return self._to_output_unit_rounded_up(self._goal_width_in_centimeters)

    @property
    def goal_area_radius(self) -> float:
        """Radius of the 6m line."""
        return self._to_output_unit_rounded_up(self._goal_area_radius_in_centimeters)

    @property
    def free_throw_radius(self) -> float:
        """Radius of the 9m dashed line."""
        return self._to_output_unit_rounded_up(self._free_throw_radius_in_centimeters)

    @property
    def penalty_mark_distance(self) -> float:
        """Distance to the 7m penalty mark."""
        return self._to_output_unit_rounded_up(self._penalty_mark_distance_in_centimeters)

    # Geometry internals
    def _raw_vertices_centimeters(self) -> List[Tuple[int, int]]:
        w = self._court_width_in_centimeters
        l = self._court_length_in_centimeters
        mw = w // 2  # Middle width (1000)
        ml = l // 2  # Middle length (2000)
        
        # Goal post offsets
        post_off = self._goal_post_offset_in_centimeters # 150
        y_post_bot = mw - post_off # 850
        y_post_top = mw + post_off # 1150
        
        # Radii
        r6 = self._goal_area_radius_in_centimeters # 600
        r9 = self._free_throw_radius_in_centimeters # 900
        
        # 9m line intersection with sideline calc
        # (x^2) + (post_offset^2) = r9^2  -> x = sqrt(r9^2 - post_offset^2)
        # 900^2 - 850^2 = 810000 - 722500 = 87500 -> sqrt = ~296
        x_9m_sideline_sq = (r9**2) - (y_post_bot**2)
        x_9m_sideline = int(x_9m_sideline_sq**0.5) if x_9m_sideline_sq > 0 else 0

        # Substitution line offset
        sub_off = self._substitution_line_distance_in_centimeters # 450

        return [
            # 0-3: Court Corners
            (0, 0),             # 00 Bottom-Left
            (0, w),             # 01 Top-Left
            (l, w),             # 02 Top-Right
            (l, 0),             # 03 Bottom-Right

            # 4-5: Center Line
            (ml, 0),            # 04 Center Bottom
            (ml, w),            # 05 Center Top

            # --- LEFT SIDE (x=0) ---
            
            # 6-9: 6m Goal Area Line (The "D")
            # Starts at baseline, goes to straight line start, ends straight line, back to baseline
            (0, y_post_bot - r6),       # 06 6m Start on Baseline (Bottom) (0, 250)
            (r6, y_post_bot),           # 07 6m Curve End / Straight Start Bottom (600, 850)
            (r6, y_post_top),           # 08 6m Curve Start / Straight End Top (600, 1150)
            (0, y_post_top + r6),       # 09 6m End on Baseline (Top) (0, 1750)

            # 10-13: 9m Free Throw Line (The Dashed "D")
            # Note: 9m line usually hits the sideline before the baseline if the court is tight, 
            # but in standard 20m width, it hits the sideline approx 2.96m from corner.
            (x_9m_sideline, 0),         # 10 9m Start on Sideline (Bottom) (~296, 0)
            (r9, y_post_bot),           # 11 9m Curve End / Straight Start Bottom (900, 850)
            (r9, y_post_top),           # 12 9m Curve Start / Straight End Top (900, 1150)
            (x_9m_sideline, w),         # 13 9m End on Sideline (Top) (~296, 2000)

            # 14-15: Marks
            (self._penalty_mark_distance_in_centimeters, mw),      # 14 7m Penalty Mark
            (self._goal_keeper_line_distance_in_centimeters, mw),  # 15 4m Keeper Line

            # 16: Goal Center
            (0, mw),                                               # 16 Left Goal Center

            # --- RIGHT SIDE (x=4000) --- (Mirrored)
            
            # 17-20: 6m Goal Area
            (l, y_post_bot - r6),       # 17 6m Start Baseline
            (l - r6, y_post_bot),       # 18 6m Straight Start
            (l - r6, y_post_top),       # 19 6m Straight End
            (l, y_post_top + r6),       # 20 6m End Baseline

            # 21-24: 9m Free Throw
            (l - x_9m_sideline, 0),     # 21 9m Start Sideline
            (l - r9, y_post_bot),       # 22 9m Straight Start
            (l - r9, y_post_top),       # 23 9m Straight End
            (l - x_9m_sideline, w),     # 24 9m End Sideline

            # 25-26: Marks
            (l - self._penalty_mark_distance_in_centimeters, mw),      # 25 7m Penalty Mark
            (l - self._goal_keeper_line_distance_in_centimeters, mw),  # 26 4m Keeper Line
            
            # 27: Goal Center
            (l, mw),                                                   # 27 Right Goal Center

            # 28-31: Substitution Lines (Sideline markers)
            (ml - sub_off, 0),  # 28 Left Sub Line Bottom
            (ml + sub_off, 0),  # 29 Right Sub Line Bottom
            (ml - sub_off, w),  # 30 Left Sub Line Top
            (ml + sub_off, w),  # 31 Right Sub Line Top
        ]

    def _vertices_in_unit(self) -> List[Tuple[float, float]]:
        return [
            (
                self._to_output_unit_rounded_up(x),
                self._to_output_unit_rounded_up(y),
            )
            for x, y in self._raw_vertices_centimeters()
        ]

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get all court vertices in the configured measurement unit.
        
        Returns:
            List of (x, y) coordinates representing geometric keypoints:
            0-3: Corners, 4-5: Center Line, 
            6-9: Left 6m, 10-13: Left 9m, 14: Left 7m, 15: Left 4m, 16: Left Goal
            17-27: Right side equivalents
            28-31: Substitution lines
        """
        return self._vertices_in_unit()

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Default edges for drawing the court lines."""
        return [
            # Outer boundary
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Center line
            (4, 5),
            # Left 6m Line (Approximate polygon)
            (6, 7), (7, 8), (8, 9),
            # Left 9m Line
            (10, 11), (11, 12), (12, 13),
            # Right 6m Line
            (17, 18), (18, 19), (19, 20),
            # Right 9m Line
            (21, 22), (22, 23), (23, 24),
        ]

    # Helper properties for easy access to specific zones
    @property
    def left_goal_area_indexes(self) -> List[int]:
        """Indexes defining the Left 6m line geometry."""
        return [6, 7, 8, 9]

    @property
    def right_goal_area_indexes(self) -> List[int]:
        """Indexes defining the Right 6m line geometry."""
        return [17, 18, 19, 20]

    @property
    def left_free_throw_indexes(self) -> List[int]:
        """Indexes defining the Left 9m dashed line geometry."""
        return [10, 11, 12, 13]

    @property
    def right_free_throw_indexes(self) -> List[int]:
        """Indexes defining the Right 9m dashed line geometry."""
        return [21, 22, 23, 24]