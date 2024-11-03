from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

from sympy import Basic, Float, Number, S, Symbol, simplify
from sympy.core.expr import Expr
from sympy.core.numbers import pi
from sympy.external import import_module
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2, cos, sin, tan
from sympy.geometry.polygon import deg, rad
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.simplify import nsimplify, simplify

plt = import_module(
    "matplotlib.pyplot",
    import_kwargs={
        "fromlist": [
            "pyplot",
        ]
    },
)

patches = import_module(
    "matplotlib.patches",
    import_kwargs={
        "fromlist": [
            "FancyArrow",
            "Circle",
            "Rectangle",
            "Polygon",
        ]
    },
)

transforms = import_module(
    "matplotlib.transforms",
    import_kwargs={
        "fromlist": [
            "transforms",
        ]
    },
)

np = import_module(
    "numpy",
    import_kwargs={
        "fromlist": [
            "numpy",
        ]
    },
)


@dataclass
class Coordinates:
    start: Union[Any, Iterable[Any]]
    end: Union[Any, Iterable[Any]]


@dataclass
class Orientation:
    angle_deg: Any
    length: Any


@dataclass
class Properties:
    E: Any = None
    I: Any = None
    A: Any = None


@dataclass
class DefaultHintValues:
    member_length: Any = 5  # default length
    load_value: Any = 10  # default load
    angle_deg: Any = 45  # default angle


class Member:
    def __init__(
        self,
        m_id: int,
        start: Iterable[Any],
        end: Optional[Iterable[Any]] = None,
        angle_deg: Any = None,
        length: Any = None,
        E: Any = None,
        I: Any = None,
        A: Any = None,
        properties: Optional[Properties] = None,
        orientation: Optional[Orientation] = None,
        label: Optional[str] = None,
        symbol_hint: Optional[dict] = None,
    ):
        self.m_id = m_id
        self.label = label
        self.symbol_hint = symbol_hint or {}
        self.properties = Properties(E=E, I=I, A=A)

        self.start = tuple(start)

        # Handle orientation and end
        if end is not None:
            self.end = end
            self.coordinates = Coordinates(start=self.start, end=self.end)
            # Compute orientation if not provided
            if orientation is not None or (
                angle_deg is not None and length is not None
            ):
                # If orientation is provided, use it
                if orientation is not None:
                    self.orientation = orientation
                    self.angle_deg = orientation.angle_deg
                    self.length = orientation.length
                else:
                    self.angle_deg = angle_deg
                    self.length = length
                    self.orientation = Orientation(
                        angle_deg=self.angle_deg, length=self.length
                    )
                # Validate that orientation matches coordinates
                self._validate_orientation_with_coordinates()
            else:
                self._compute_orientation_from_coordinates()
        elif angle_deg is not None and length is not None:
            self.angle_deg = angle_deg
            self.length = length
            self.orientation = Orientation(angle_deg=self.angle_deg, length=self.length)
            # Compute end from orientation
            self.end = self._compute_end_from_orientation()
            self.coordinates = Coordinates(start=self.start, end=self.end)
        else:
            raise ValueError(
                "Must provide 'start' (with 'end' or 'angle_deg' and 'length')."
            )

        # Perform checks
        self._perform_checks()

        # Set default hints
        default_hints = DefaultHintValues()
        self._set_default_hints(default_hints)

    def _compute_orientation_from_coordinates(self):
        x0, y0 = self.start
        x1, y1 = self.end
        dx = x1 - x0
        dy = y1 - y0
        self.length = sqrt(dx**2 + dy**2)
        self.angle_deg = 180 * atan2(dy, dx) / pi
        self.orientation = Orientation(angle_deg=self.angle_deg, length=self.length)

    def _compute_end_from_orientation(self):
        x0, y0 = self.start
        angle_rad = rad(self.angle_deg)
        dx = self.length * cos(angle_rad)
        dy = self.length * sin(angle_rad)
        end_x = x0 + dx
        end_y = y0 + dy
        return (end_x, end_y)

    def _validate_orientation_with_coordinates(self):
        # Compute end from start, angle_deg, and length
        computed_end = self._compute_end_from_orientation()
        # Use provided end coordinate
        provided_end = (self.end[0], self.end[1])
        # Check if they are equal
        if not self._points_are_equal(provided_end, computed_end):
            raise ValueError(
                f"Provided 'end' {self.end} and 'orientation' (computed end {computed_end}) do not match."
            )

    def _points_are_equal(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        eq_x = simplify(x1 - x2) == 0
        eq_y = simplify(y1 - y2) == 0
        return eq_x and eq_y

    def _perform_checks(self):
        # Check if 'start' and 'end' are correct
        for point, name in [(self.start, "start"), (self.end, "end")]:
            if not (isinstance(point, Iterable) and len(point) == 2):
                raise ValueError(f"'{name}' must be a 2D point.")

    def _set_default_hints(self, default_hints: DefaultHintValues):
        # Collect all symbols that may need hints
        all_symbols = ["member_length", "load_value", "angle_deg"]
        for symbol in all_symbols:
            if symbol not in self.symbol_hint:
                self.symbol_hint[symbol] = getattr(default_hints, symbol)

    def rename(self, new_name):
        self.label = new_name

    def __repr__(self):
        return (
            f"Member(m_id={self.m_id}, coordinates={self.coordinates}, "
            f"orientation={self.orientation}, label={self.label}, "
            f"symbol_hint={self.symbol_hint}, properties={self.properties})"
        )


class Draw2:
    def __init__(self, members, loads, supports, connections, theme="default"):
        self.members = [Member(**member) for member in members]
        self.loads = loads
        self.supports = supports
        self.connections = connections

        color_themes = {
            "default": {
                "member_color": "black",
                "member_color_symbolic": "gray",
                "load_color": "red",
                "load_color_symbolic": "orange",
                "distributed_load_color": "purple",
                "distributed_load_color_symbolic": "pink",
                "support_color": "blue",
                "support_color_symbolic": "cyan",
                "support_color_icon": "black",
                "connection_color": "green",
            }
        }
        self.theme = color_themes[theme]

    AllowedValueTypes = Symbol | Expr | Number | Float

    def parse_value(
        value: AllowedValueTypes, rounding=None, try_nummeric=False
    ) -> tuple[Any, str]:
        # Handle standalone float values
        if isinstance(value, (Float, float)):
            draw_value = float(value)
            if rounding is not None:
                return round(float(value), rounding), "float_rounded"
            return value, draw_value, "float_exact"

        # Handle SymPy expressions
        if isinstance(value, Basic):
            has_symbols = bool(value.free_symbols)
            float_atoms = value.atoms(Float)
            numbers = value.atoms(Number)

            draw_value = value

            # Convert numbers to floats if try_nummeric is True
            if try_nummeric:
                value = value.xreplace({num: float(num) for num in numbers})

                if rounding is not None:
                    value = value.xreplace(
                        {f: round(float(f), rounding) for f in value.atoms(Float)}
                    )
                status = "numeric_with_symbols" if has_symbols else "numeric_only"
                return value, draw_value, status

            # Pure symbols case (only symbols, no numbers at all)
            if has_symbols and not float_atoms and not numbers:
                return value, "symbol_only"

            # Handle cases with both symbols and exact numbers (but no floats)
            if has_symbols and numbers and not float_atoms:
                return value, "symbol_mixed_exact"

            if float_atoms:
                if rounding is not None:
                    # Round only the Float components while preserving other parts
                    rounded_value = value.xreplace(
                        {f: round(float(f), rounding) for f in float_atoms}
                    )
                    return (
                        rounded_value,
                        "symbol_mixed_rounded" if has_symbols else "expr_mixed_rounded",
                    )
                return (
                    value,
                    "symbol_mixed_exact" if has_symbols else "expr_mixed_exact",
                )

            if has_symbols:
                return value, "symbol_free_exact"

            return value, "exact"

        # Fallback for unexpected cases
        return value, "unknown"

    # # Test the function
    # print(parse_value(Symbol('F')))
    # print(parse_value(sqrt(2)))

    # print(parse_value(Symbol('F') + cos(2),2))            # Expected: (F + 2, 'symbol_free_exact')
    # print(parse_value(sqrt(2.65454) + Symbol('F'), 3))  # Expected: (rounded value, 'symbol_mixed_rounded')
    # print(parse_value(2.65454, 3))                      # Expected: (2.655, 'float_rounded')
    # print(parse_value(2.65454))
    # print(parse_value(sin(2) + 3.6568, 2))              # Expected: (rounded expression, 'expr_mixed_rounded')
    # print(parse_value(sqrt(2) + 3.6568))                # Expected: (sqrt(2) + 3.6568, 'expr_mixed')
    # print(parse_value(Symbol('F') + cos(2), 2,try_nummeric=True))
    # print(parse_value(sin(pi) + 3.6568, 3,try_nummeric=True))
    # print(parse_value(sin(sqrt(2))+0.3232654))

    def draw2(self):
        ####################################################################################

        plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.gca()
        ax.set_aspect("equal")

        member_color = self.theme["member_color"]
        member_color_symbolic = self.theme["member_color_symbolic"]

        for member in self.members:
            start, end = member.start, member.end

            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                label=member.label or f"Member {member.m_id}",
                color=member_color,
                lw=5,
                solid_capstyle="butt",
                # solid_joinstyle='round',
            )

        # plt.scatter([3], [4], color='black', s=10, zorder=10)

        plt.legend()

        ###############################################################################################
        x_ticks = plt.xticks()[0]
        y_ticks = plt.yticks()[0]

        if len(x_ticks) > 1 and len(y_ticks) > 1:
            x_step = x_ticks[1] - x_ticks[0]
            y_step = y_ticks[1] - y_ticks[0]

            step = min(x_step, y_step)

            if step < 1:
                step = 1

            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()

            # Generate new ticks using the updated step
            new_x_ticks = [
                i * step for i in range(int(x_min // step), int(x_max // step) + 1)
            ]
            new_y_ticks = [
                i * step for i in range(int(y_min // step), int(y_max // step) + 1)
            ]

            plt.xticks(new_x_ticks)
            plt.yticks(new_y_ticks)

        ax.grid(True, zorder=10)
