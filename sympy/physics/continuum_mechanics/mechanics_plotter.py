from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union, cast

from sympy.core import Basic, Expr, Float, Integer, S, Symbol
from sympy.core.numbers import pi
from sympy.external import import_module
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2, cos, sin
from sympy.geometry.polygon import deg, rad
from sympy.simplify import simplify

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

np = import_module(
    "numpy",
    import_kwargs={
        "fromlist": [
            "numpy",
        ]
    },
)


# Define allowed value types
AllowedValueTypes = Union[int, float, Expr]
Point2D = Tuple[AllowedValueTypes, AllowedValueTypes]


@dataclass
class ParseResult:
    original_expression: AllowedValueTypes
    collapsed_expression: Expr


def separate_symbolic_and_constants(
    expr: Expr, symbols: Iterable[Symbol]
) -> Tuple[int, float]:
    """
    Separates the expression into symbolic and constant parts.

    Args:
        expr (Expr): The SymPy expression to separate.
        symbols (Iterable[Symbol]): The symbols to consider as symbolic parts.

    Returns:
        Tuple[Expr, float]: A tuple containing the symbolic part and the constant part.
    """
    symbolic_part = 0
    constant_part = 0.0

    # Check if the expression is additive (e.g., a + b + c)
    if expr.is_Add:
        terms = expr.as_ordered_terms()
    else:
        # Treat the entire expression as a single term
        terms = [expr]

    for term in terms:
        if term.has(*symbols):
            symbolic_part += term
        else:
            # Convert term to float to ensure constant_part remains a float
            constant_part += float(term)

    return symbolic_part, constant_part


def to_sympy_number(n: float, rounding: Optional[int] = None) -> Union[Integer, Float]:
    """
    Converts a numerical value to SymPy Integer or Float based on its value.

    Args:
        n (float): The numerical value to convert.
        rounding (Optional[int]): Number of decimal places to round the value.

    Returns:
        Union[Integer, Float]: SymPy Integer if n is whole after rounding, else SymPy Float.
    """
    if rounding is not None:
        n = round(n, rounding)
    if isinstance(n, float) and n.is_integer():
        return Integer(int(n))
    else:
        if rounding is not None:
            # Create Float from string to preserve exact decimal digits
            n_str = f"{n:.{rounding}f}"
            return Float(n_str)
        else:
            return Float(n)


def parse_value(
    value: AllowedValueTypes,
    rounding: Optional[int] = None,
) -> ParseResult:
    """
    Parses the input value and returns the original and collapsed expressions.

    The collapsed expression expresses symbolic parts multiplied by their constants.

    Args:
        value (AllowedValueTypes): The value to parse (int, float, or SymPy expression).
        rounding (Optional[int]): Number of decimal places to round the constant parts.

    Returns:
        ParseResult: An object containing the original and collapsed expressions.
    """
    MAX_ROUNDING = 15
    if rounding is None:
        rounding = MAX_ROUNDING

    if not isinstance(value, (int, float, Expr)):
        raise ValueError("Invalid value type")

    # Initialize the original expression
    original_expr: AllowedValueTypes = value

    # Initialize collapsed expression as Expr
    collapsed_expr: Expr

    if isinstance(original_expr, Expr):
        if original_expr.free_symbols:
            # Extract all free symbols in the expression
            free_symbols = cast(Iterable[Symbol], original_expr.free_symbols)
            symbolic, constant = separate_symbolic_and_constants(
                original_expr, free_symbols
            )

            # Initialize the collapsed expression as Expr
            collapsed_expr = S.Zero

            # Process each symbolic term
            if symbolic != 0:
                if isinstance(symbolic, Expr) and symbolic.is_Add:
                    symbolic_terms = symbolic.as_ordered_terms()
                else:
                    symbolic_terms = [symbolic]

                for term in symbolic_terms:
                    if hasattr(term, "as_coeff_Mul"):
                        coeff, sym_expr = term.as_coeff_Mul()
                    else:
                        coeff, sym_expr = 1, term
                    coeff = round(float(coeff), rounding)
                    if coeff == 1:
                        collapsed_expr += sym_expr
                    elif coeff == -1:
                        collapsed_expr -= sym_expr
                    else:
                        collapsed_expr += to_sympy_number(coeff, rounding) * sym_expr

            # Add the constant part if it exists
            if constant != 0:
                constant_sympy = to_sympy_number(constant, rounding)
                collapsed_expr += constant_sympy

        else:
            # Expression has no free symbols; it's a constant expression
            numeric = float(original_expr.evalf())
            numeric_sympy = to_sympy_number(numeric, rounding)
            collapsed_expr = numeric_sympy  # float or Integer

    else:
        # For int or float, collapsed expression is the same as original
        collapsed_expr = (
            to_sympy_number(float(original_expr), rounding)
            if rounding is not None
            else Float(float(original_expr))
        )

    if isinstance(collapsed_expr, Float):
        collapsed_expr = collapsed_expr.round(rounding)

    return ParseResult(
        original_expression=original_expr, collapsed_expression=collapsed_expr
    )


print(parse_value(Symbol("F")))
print(parse_value(sqrt(2)))

print(parse_value(Symbol("F") + cos(2), 3))
print(parse_value(sqrt(2.65454) + Symbol("F"), 3))
print(parse_value(2.65454, 3))
print(parse_value(2.65454))
print(parse_value(sin(2) + 3.6568, rounding=3))
print(parse_value(sqrt(2) + 3.6568))
print(parse_value(Symbol("F") + cos(2), 2))
print(parse_value(sin(pi) + 3.6568, 3))
print(parse_value(sin(sqrt(2)) + 0.3232654))


@dataclass
class Coordinates:
    start: Point2D  # Now explicitly a tuple of two elements
    end: Point2D


@dataclass
class Orientation:
    angle_deg: AllowedValueTypes
    length: AllowedValueTypes


@dataclass
class Properties:
    E: Optional[AllowedValueTypes] = None
    I: Optional[AllowedValueTypes] = None
    A: Optional[AllowedValueTypes] = None


@dataclass
class DefaultHintValues:
    member_length: AllowedValueTypes = 5  # default length
    load_value: AllowedValueTypes = 10  # default load
    angle_deg: AllowedValueTypes = 45  # default angle


class Member:
    def __init__(
        self,
        m_id: int,
        start: Point2D,
        end: Optional[Point2D] = None,
        angle_deg: Optional[AllowedValueTypes] = None,
        length: Optional[AllowedValueTypes] = None,
        E: Optional[AllowedValueTypes] = None,
        I: Optional[AllowedValueTypes] = None,
        A: Optional[AllowedValueTypes] = None,
        # properties: Optional[Properties] = None,
        orientation: Optional[Orientation] = None,
        label: Optional[str] = None,
        symbol_hint: Optional[Dict[Symbol, AllowedValueTypes]] = None,
    ):
        self.m_id = m_id
        self.label = label
        self.symbol_hint: Dict[Symbol, AllowedValueTypes] = symbol_hint or {}
        self.properties = Properties(E=E, I=I, A=A)

        # Ensure start is properly typed as Point2D
        self.start: Point2D = (start[0], start[1])

        # Initialize these attributes with proper types
        self.angle_deg: AllowedValueTypes
        self.length: AllowedValueTypes
        self.end: Point2D

        # Handle orientation and end
        if end is not None:
            self.end = (end[0], end[1])
            self.coordinates = Coordinates(
                start=self.start,  # Now properly typed as Point2D
                end=self.end,
            )
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
                    if angle_deg is None or length is None:
                        raise ValueError("Both angle_deg and length must be provided")
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
        self._validate_2d_points()

        # Set default hints
        default_hints = DefaultHintValues()
        self._set_default_hints(default_hints)

    def _compute_orientation_from_coordinates(self) -> None:
        x0, y0 = self.start
        x1, y1 = self.end
        dx = x1 - x0
        dy = y1 - y0
        self.length = sqrt(dx**2 + dy**2)
        self.angle_deg = deg(atan2(dy, dx))
        self.orientation = Orientation(angle_deg=self.angle_deg, length=self.length)

    def _compute_end_from_orientation(self) -> Point2D:
        x0, y0 = self.start
        angle_rad = rad(self.angle_deg)
        dx = self.length * cos(angle_rad)
        dy = self.length * sin(angle_rad)
        end_x = x0 + dx
        end_y = y0 + dy
        return (end_x, end_y)

    def _validate_orientation_with_coordinates(self) -> None:
        computed_end = self._compute_end_from_orientation()
        provided_end = (self.end[0], self.end[1])
        if not self._points_are_equal(provided_end, computed_end):
            raise ValueError(
                f"Provided 'end' {self.end} and 'orientation' (computed 'end' {computed_end}) do not match."
            )

    def _points_are_equal(self, point1: Point2D, point2: Point2D) -> bool:
        x1, y1 = point1
        x2, y2 = point2
        eq_x = simplify(x1 - x2) == 0
        eq_y = simplify(y1 - y2) == 0
        return eq_x and eq_y

    def _validate_2d_points(self) -> None:
        # Check that start and end are 2D points
        for point, name in [(self.start, "start"), (self.end, "end")]:
            if not (isinstance(point, Iterable) and len(point) == 2):
                raise ValueError(f"'{name}' must be a 2D point.")

    def _set_default_hints(self, default_hints: DefaultHintValues) -> None:
        # Handle angle_deg symbols
        if isinstance(self.angle_deg, Basic):
            for symbol in self.angle_deg.free_symbols:
                if isinstance(symbol, Symbol) and symbol not in self.symbol_hint:
                    self.symbol_hint[symbol] = default_hints.angle_deg

        # Handle coordinates and length symbols
        if isinstance(self.length, Basic):
            for symbol in self.length.free_symbols:
                if isinstance(symbol, Symbol) and symbol not in self.symbol_hint:
                    self.symbol_hint[symbol] = default_hints.member_length

        for coord in [self.start, self.end]:
            for value in coord:
                if isinstance(value, Basic):
                    for symbol in value.free_symbols:
                        if (
                            isinstance(symbol, Symbol)
                            and symbol not in self.symbol_hint
                        ):
                            self.symbol_hint[symbol] = default_hints.member_length

    def rename(self, new_name: str) -> None:
        self.label = new_name

    def __repr__(self) -> str:
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

        # AllowedValueTypes = Symbol | Expr | Number | Float

    # Define type aliases for better readability

    def draw2(self):
        ####################################################################################

        plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.gca()
        ax.set_aspect("equal")

        member_color = self.theme["member_color"]
        # member_color_symbolic = self.theme["member_color_symbolic"]

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
