from sympy import Basic, Float, Symbol, simplify, S,Number
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


class Member:
    def __init__(self, m_id, label, start, end, symbol_hint, properties):
        self.m_id = m_id
        self.label = label
        self.start = start
        self.end = end
        self.symbol_hint = symbol_hint
        self.properties = properties

    def rename(self, new_name):
        self.label = new_name


# Extend this to loads, supports, and connections if needed


class Draw2:
    def __init__(self, members, loads, supports, connections, theme='default'):
        self.members = [Member(**member) for member in members]
        self.loads = loads
        self.supports = supports
        self.connections = connections

        color_themes = {'default':{
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

        }}
        self.theme = color_themes[theme]


    DEFAULT_HINT_VALUES = {
        "member_length": 5,
        "load_value": 10,
        "bending_moment": 10,
        "angle_deg": 45,
    }


    def draw2(self):

        def parse_value(value, rounding=None, try_nummeric=False):
            # Handle standalone float values
            if isinstance(value, (Float, float)):
                if rounding is not None:
                    return round(float(value), rounding), 'float_rounded'
                return float(value), 'float_exact'

            # Handle SymPy expressions
            if isinstance(value, Basic):
                has_symbols = bool(value.free_symbols)
                float_atoms = value.atoms(Float)
                numbers = value.atoms(Number)

                # Convert numbers to floats if try_nummeric is True
                if try_nummeric:
                    value = value.xreplace({num: float(num) for num in numbers})
                    if rounding is not None:
                        value = value.xreplace({f: round(float(f), rounding) for f in value.atoms(Float)})
                    status = 'numeric_with_symbols' if has_symbols else 'numeric_only'
                    return value, status

                # Pure symbols case (only symbols, no numbers at all)
                if has_symbols and not float_atoms and not numbers:
                    return value, 'symbol_only'

                # Handle cases with both symbols and exact numbers (but no floats)
                if has_symbols and numbers and not float_atoms:
                    return value, 'symbol_mixed_exact'

                if float_atoms:
                    if rounding is not None:
                        # Round only the Float components while preserving other parts
                        rounded_value = value.xreplace({f: round(float(f), rounding) for f in float_atoms})
                        return rounded_value, 'symbol_mixed_rounded' if has_symbols else 'expr_mixed_rounded'
                    return value, 'symbol_mixed_exact' if has_symbols else 'expr_mixed_exact'

                if has_symbols:
                    return value, 'symbol_free_exact'

                return value, 'exact'

            # Fallback for unexpected cases
            return value, 'unknown'

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

####################################################################################

        plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.gca()
        ax.set_aspect("equal")




        member_color = self.theme['member_color']
        member_color_symbolic = self.theme['member_color_symbolic']


        for member in self.members:
            start, end = member.start, member.end

            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                label=member.label or f"Member {member.m_id}",
                color=member_color,
                lw=5,
                solid_capstyle='butt',
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



