
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, ClassVar

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.patches as mpatches

import matplotlib.font_manager as font_manager


Difficulty = str
Question = Tuple[str, str]
Slip = List[Question]


def _choose(range_or_list):
    """Helper to choose a random element or generate a random integer.

    If ``range_or_list`` is a tuple of (low, high) it returns an integer
    uniformly between low and high inclusive.  Otherwise it returns a
    random choice from the provided list.
    """
    if isinstance(range_or_list, tuple) and len(range_or_list) == 2:
        low, high = range_or_list
        return random.randint(low, high)
    else:
        return random.choice(range_or_list)


@dataclass
class ArithmeticSlipGenerator:
    """Generate arithmetic slips covering a range of mental‑math skills.

    Each slip contains exactly 10 problems from the following categories:
    addition, subtraction, multiplication, division, percentage, square,
    and cube. The quantity of each type is fixed for fairness — two
    additions, two subtractions, two multiplications, one division,
    one percentage, one square, and one cube — but their order on the
    slip is randomized. Difficulties control the ranges of the operands
    and the complexity of the problems.
    """

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

        # Define numeric ranges for each operation and difficulty.
        # The keys map difficulty names to tuples or lists used by
        # ``_choose``; ranges are inclusive.
        self.ranges: Dict[str, Dict[str, Tuple[int, int] | List[int]]] = {
            'easy': {
                'addend': (10, 99),      # two‑digit addition
                'subtrahend': (10, 99),   # two‑digit subtraction
                'multiplicand': (10, 99), # two‑digit multiplication (× one‑digit)
                'multiplier': (2, 9),
                'divisor': (2, 9),        # one‑digit divisors
                'quotient': (2, 20),
                'percent': [5, 10, 25, 50],
                'square': (2, 12),        # squares up to 12²
                'cube': (2, 5),           # cubes up to 5³
            },
            'medium': {
                'addend': (100, 999),     # three‑digit addition
                'subtrahend': (100, 999),
                'multiplicand': (20, 99), # two‑digit × two‑digit
                'multiplier': (10, 30),
                'divisor': (2, 20),
                'quotient': (10, 50),
                'percent': [5, 10, 12, 15, 20, 25, 30],
                'square': (13, 15),       # squares up to 15²
                'cube': (6, 7),           # cubes up to 7³
            },
            'hard': {
                'addend': (100, 999),
                'subtrahend': (100, 999),
                'multiplicand': (100, 999), # three‑digit × two‑digit
                'multiplier': (10, 50),
                'divisor': (10, 50),
                'quotient': (10, 99),
                'percent': [7, 13, 17, 22, 35],
                'square': (16, 20),       # squares up to 20²
                'cube': (8, 9),           # cubes up to 9³
            },
            'impossible': {
                'addend': (1000, 9999),
                'subtrahend': (1000, 9999),
                'multiplicand': (100, 999), # three‑digit × two‑digit (large multiplier)
                'multiplier': (20, 99),
                'divisor': (20, 99),
                'quotient': (10, 99),
                'percent': list(range(1, 100)),
                'square': (21, 25),       # squares up to 25²
                'cube': (10, 10),         # cubes of 10³ only
            },
        }

    def _generate_addition(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        a = _choose(r['addend'])
        b = _choose(r['addend'])
        question = f"{a} + {b} ="
        answer = str(a + b)
        return question, answer

    def _generate_subtraction(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        # Ensure the result is non‑negative by ordering the operands
        a = _choose(r['subtrahend'])
        b = _choose(r['subtrahend'])
        if b > a:
            a, b = b, a
        question = f"{a} − {b} ="
        answer = str(a - b)
        return question, answer

    def _generate_multiplication(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        a = _choose(r['multiplicand'])
        b = _choose(r['multiplier'])
        question = f"{a} × {b} ="
        answer = str(a * b)
        return question, answer

    def _generate_division(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        # Construct a dividend divisible by the divisor
        b = _choose(r['divisor'])
        q = _choose(r['quotient'])
        dividend = b * q
        question = f"{dividend} ÷ {b} ="
        answer = str(q)
        return question, answer

    def _generate_percentage(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        percent = _choose(r['percent'])
        # Choose a base number large enough to be interesting
        if difficulty == 'easy':
            base = random.randint(20, 200)
        elif difficulty == 'medium':
            base = random.randint(100, 999)
        elif difficulty == 'hard':
            base = random.randint(200, 2000)
        else:  # impossible
            base = random.randint(200, 5000)
        question = f"{percent}% of {base} ="
        result = base * percent / 100
        # Round to two decimal places only for non‑integer results
        answer = f"{result:.2f}" if result != int(result) else str(int(result))
        return question, answer

    def _generate_square(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        n = _choose(r['square'])
        question = f"{n}² ="
        answer = str(n * n)
        return question, answer

    def _generate_cube(self, difficulty: Difficulty) -> Question:
        r = self.ranges[difficulty]
        n = _choose(r['cube'])
        question = f"{n}³ ="
        answer = str(n * n * n)
        return question, answer

    def generate_slip(self, difficulty: Difficulty) -> Tuple[Slip, List[str]]:
        """Generate a single slip and corresponding answers for a given difficulty.

        Returns a tuple containing the list of (question, answer) pairs and
        the list of answers in order.  The answers list is kept separate to
        facilitate answer page construction.
        """
        # Fixed distribution for fairness; randomized order
        ops = (
            ['add'] * 1
            + ['subtract'] * 1
            + ['multiply'] * 2
            + ['divide']*2
            + ['percent']
            + ['square']*2
            + ['cube']
        )
        random.shuffle(ops)
        slip: Slip = []
        answers: List[str] = []
        for op in ops:
            if op == 'add':
                q, a = self._generate_addition(difficulty)
            elif op == 'subtract':
                q, a = self._generate_subtraction(difficulty)
            elif op == 'multiply':
                q, a = self._generate_multiplication(difficulty)
            elif op == 'divide':
                q, a = self._generate_division(difficulty)
            elif op == 'percent':
                q, a = self._generate_percentage(difficulty)
            elif op == 'square':
                q, a = self._generate_square(difficulty)
            elif op == 'cube':
                q, a = self._generate_cube(difficulty)
            else:
                raise ValueError(f"Unknown operation: {op}")
            slip.append((q, a))
            answers.append(a)
        return slip, answers

    def generate_slips(self, difficulty: Difficulty, slips_per_page: int = 10) -> Tuple[List[Slip], List[List[str]]]:
        """Generate multiple slips for the specified difficulty.

        :param difficulty: One of 'easy', 'medium', 'hard', 'impossible'
        :param slips_per_page: Number of slips to generate (default: 10)
        :returns: A tuple of list of slips and list of answer lists.
        """
        slips: List[Slip] = []
        answers_list: List[List[str]] = []
        for _ in range(slips_per_page):
            slip, ans = self.generate_slip(difficulty)
            slips.append(slip)
            answers_list.append(ans)
        return slips, answers_list


@dataclass
class EquationSlipGenerator:
    """Generate slips containing algebraic equations of varying complexity.

    Each slip contains ten equations, and the mix of equation types depends
    on the chosen difficulty.  The generator supports four categories of
    equations: ``one_step``, ``two_step``, ``both_sides`` and ``multi_step``.
    """

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

    # The following dictionaries are class attributes rather than dataclass
    # fields.  Defining them at the class level avoids the mutable default
    # error that dataclasses raise when mutable objects are used as
    # default field values.
    eq_ranges: ClassVar[Dict[str, Dict[str, Tuple[int, int]]]] = {
        'easy': {
            'coeff': (1, 10),
            'const': (1, 10),
            'x': (1, 10),
        },
        'medium': {
            'coeff': (1, 12),
            'const': (1, 20),
            'x': (1, 20),
        },
        'hard': {
            'coeff': (1, 20),
            'const': (1, 30),
            'x': (1, 30),
        },
        'impossible': {
            'coeff': (1, 30),
            'const': (1, 50),
            'x': (1, 50),
        },
    }

    distribution: ClassVar[Dict[str, List[str]]] = {
        'easy': ['one_step'] * 10,
        'medium': ['one_step'] * 2 + ['two_step'] * 4+['both_sides'] * 3+['multi_step'] * 1,
        'hard': ['one_step'] * 3 + ['two_step'] * 5 + ['both_sides'] * 2,
        'impossible': ['one_step'] * 2 + ['two_step'] * 4 + ['both_sides'] * 2 + ['multi_step'] * 2,
    }

    def _gen_one_step(self, difficulty: Difficulty) -> Question:
        r = self.eq_ranges[difficulty]
        # Choose a template randomly
        template = random.choice(['x + a = b', 'x - a = b', 'a + x = b', 'a x = b', 'x / a = b'])
        if template == 'x + a = b':
            x = _choose(r['x'])
            a = _choose(r['const'])
            b = x + a
            question = f"x + {a} = {b}"
            answer = str(x)
        elif template == 'x - a = b':
            # ensure x - a positive by choosing x >= a
            a = _choose(r['const'])
            x = random.randint(a, r['x'][1])
            b = x - a
            question = f"x − {a} = {b}"
            answer = str(x)
        elif template == 'a + x = b':
            a = _choose(r['const'])
            x = _choose(r['x'])
            b = a + x
            question = f"{a} + x = {b}"
            answer = str(x)
        elif template == 'a x = b':
            a = _choose(r['coeff'])
            x = _choose(r['x'])
            b = a * x
            question = f"{a}x = {b}"
            answer = str(x)
        elif template == 'x / a = b':
            a = _choose(r['coeff'])
            b = _choose(r['x'])
            x = a * b
            question = f"x ÷ {a} = {b}"
            answer = str(x)
        else:
            raise ValueError(f"Unknown template {template}")
        return question, answer

    def _gen_two_step(self, difficulty: Difficulty) -> Question:
        r = self.eq_ranges[difficulty]
        template = random.choice([
            'ax + b = c',
            'ax - b = c',
            'x/a + b = c',
            '(x + a)/b = c',
        ])
        if template == 'ax + b = c':
            x = _choose(r['x'])
            a = _choose(r['coeff'])
            b = _choose(r['const'])
            c = a * x + b
            question = f"{a}x + {b} = {c}"
            answer = str(x)
        elif template == 'ax - b = c':
            x = _choose(r['x'])
            a = _choose(r['coeff'])
            b = _choose(r['const'])
            c = a * x - b
            question = f"{a}x − {b} = {c}"
            answer = str(x)
        elif template == 'x/a + b = c':
            a = _choose(r['coeff'])
            k = _choose(r['x'])  # choose a multiple factor
            x = a * k
            b = _choose(r['const'])
            c = k + b
            question = f"x ÷ {a} + {b} = {c}"
            answer = str(x)
        elif template == '(x + a)/b = c':
            b = _choose(r['coeff'])
            x = _choose(r['x'])
            a = _choose(r['const'])
            numerator = x + a
            # adjust numerator to be divisible by b
            # If not divisible, adjust a accordingly
            remainder = numerator % b
            if remainder != 0:
                a += (b - remainder)
                numerator = x + a
            c = numerator // b
            question = f"(x + {a}) ÷ {b} = {c}"
            answer = str(x)
        else:
            raise ValueError(f"Unknown template {template}")
        return question, answer

    def _gen_both_sides(self, difficulty: Difficulty) -> Question:
        r = self.eq_ranges[difficulty]
        # Choose coefficients for x on both sides, ensure they are not equal
        a = _choose(r['coeff'])
        c = _choose(r['coeff'])
        while c == a:
            c = _choose(r['coeff'])
        x = _choose(r['x'])
        b = _choose(r['const'])
        d = a * x + b - c * x
        # Build equation a x + b = c x + d
        lhs = f"{a}x + {b}"
        rhs = f"{c}x + {d}"
        question = f"{lhs} = {rhs}"
        answer = str(x)
        return question, answer

    def _gen_multi_step(self, difficulty: Difficulty) -> Question:
        r = self.eq_ranges[difficulty]
        template = random.choice([
            'a(x + b) = c',
            '(x - a)/b + c = d',
        ])
        if template == 'a(x + b) = c':
            a = _choose(r['coeff'])
            x = _choose(r['x'])
            b = _choose(r['const'])
            c = a * (x + b)
            question = f"{a}(x + {b}) = {c}"
            answer = str(x)
        elif template == '(x - a)/b + c = d':
            b = _choose(r['coeff'])
            a_val = _choose(r['const'])
            x = _choose(r['x'])
            c = _choose(r['const'])
            numerator = x - a_val
            # adjust numerator to be divisible by b
            remainder = numerator % b
            if remainder != 0:
                # increase x to make divisible
                x += (b - remainder)
                numerator = x - a_val
            k = numerator // b
            d_val = k + c
            question = f"(x − {a_val}) ÷ {b} + {c} = {d_val}"
            answer = str(x)
        else:
            raise ValueError(f"Unknown template {template}")
        return question, answer

    def generate_slip(self, difficulty: Difficulty) -> Tuple[Slip, List[str]]:
        """Generate a single equation slip for the given difficulty level."""
        ops = self.distribution[difficulty][:]
        random.shuffle(ops)
        slip: Slip = []
        answers: List[str] = []
        for op in ops:
            if op == 'one_step':
                q, a = self._gen_one_step(difficulty)
            elif op == 'two_step':
                q, a = self._gen_two_step(difficulty)
            elif op == 'both_sides':
                q, a = self._gen_both_sides(difficulty)
            elif op == 'multi_step':
                q, a = self._gen_multi_step(difficulty)
            else:
                raise ValueError(f"Unknown equation type {op}")
            slip.append((q, a))
            answers.append(a)
        return slip, answers

    def generate_slips(self, difficulty: Difficulty, slips_per_page: int = 10) -> Tuple[List[Slip], List[List[str]]]:
        """Generate multiple equation slips for the specified difficulty."""
        slips: List[Slip] = []
        answers_list: List[List[str]] = []
        for _ in range(slips_per_page):
            slip, ans = self.generate_slip(difficulty)
            slips.append(slip)
            answers_list.append(ans)
        return slips, answers_list


class SlipPDFBuilder:
    """Build a PDF containing slips and optionally answers using Matplotlib.

    A new figure is created for each page.  The slips page draws a 2×5
    grid of boxes with headers and questions.  The answers page lays
    out the answers in a simple list.  All sizing is relative to an
    A4 page (210×297 mm).  We rely on Matplotlib’s ``PdfPages`` to
    assemble the multi‑page PDF.
    """

    def __init__(self, slips_per_page: int = 10, questions_per_slip: int = 10) -> None:
        self.slips_per_page = slips_per_page
        self.questions_per_slip = questions_per_slip

    def _create_slips_figure(self, slips: List[Slip], difficulty: Difficulty, title: str) -> plt.Figure:
        """Create a Matplotlib figure with a grid of slips."""
        # A4 size in inches (width × height)
        fig_width, fig_height = 8.27, 11.69
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        # Draw title
        ax.text(0.5, 0.98, title, ha='center', va='top', fontsize=14, fontweight='bold')
        # Grid parameters
        cols, rows = 2, 5
        # Margins as a fraction of the figure
        margin_x, margin_y = 0.05, 0.05
        gutter_x, gutter_y = 0.02, 0.02
        cell_w = (1 - 2 * margin_x - gutter_x * (cols - 1)) / cols
        cell_h = (1 - 2 * margin_y - gutter_y * (rows - 1)) / rows
        for i, slip in enumerate(slips):
            col, row = i % cols, i // cols
            x0 = margin_x + col * (cell_w + gutter_x)
            y0 = 1 - margin_y - (row + 1) * cell_h - row * gutter_y
            # Draw rectangle for slip
            rect = mpatches.Rectangle((x0, y0), cell_w, cell_h, fill=False, linewidth=0.5)
            ax.add_patch(rect)
            # Header text
            ax.text(
                x0 + 0.005,
                y0 + cell_h - 0.01,
                f"Slip {i + 1} – {difficulty.capitalize()}",
                fontsize=8,
                fontweight='bold',
                va='top',
            )
            # Compute vertical step so that all questions fit in the available space
            header_height = 0.01  # fraction of cell height reserved for header
            available_height = cell_h - header_height - 0.02  # leave a bottom margin
            n_questions = len(slip)
            if n_questions > 0:
                step_y = available_height / n_questions
            else:
                step_y = 0
            # Starting y position for first question
            y_start = y0 + cell_h - header_height
            for idx, (q, _) in enumerate(slip, 1):
                y_pos = y_start - idx * step_y
                ax.text(
                    x0 + 0.01,
                    y_pos,
                    f"{idx}. {q}",
                    fontsize=7,
                    va='top',
                )
        return fig

    def _create_answers_figures(self, all_answers: List[List[str]], difficulty: Difficulty, title: str) -> List[plt.Figure]:
        """Create a single figure containing all answers fitted onto one page.

        We lay out the lines in multiple columns to ensure everything fits
        on a single A4 page while keeping a readable font size.
        """
        # Prepare a flattened list of strings for answers
        lines: List[str] = []
        for i, answers in enumerate(all_answers, 1):
            lines.append(f"Slip {i}")
            for idx, ans in enumerate(answers, 1):
                lines.append(f"  {idx}. {ans}")
            lines.append("")  # blank line between slips

        # Create a single figure
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # Title
        ax.text(0.5, 0.98, f"Answers – {difficulty.capitalize()}", ha='center', va='top', fontsize=12, fontweight='bold')

        # Layout parameters (relative coords)
        top = 0.94
        bottom = 0.06
        left_margin = 0.06
        right_margin = 0.06
        line_height = 0.022  # vertical step between lines
        font_size = 8

        # Compute how many lines fit per column
        available_height = max(top - bottom, 0.01)
        lines_per_col = max(int(available_height / line_height), 1)
        total_lines = len(lines)
        n_cols = max((total_lines + lines_per_col - 1) // lines_per_col, 1)

        # If too many columns would be needed, tighten spacing slightly
        if n_cols > 4:
            line_height = 0.020
            lines_per_col = max(int(available_height / line_height), 1)
            n_cols = max((total_lines + lines_per_col - 1) // lines_per_col, 1)
        if n_cols > 6:
            # as a last resort reduce font a bit
            font_size = 7
            line_height = 0.018
            lines_per_col = max(int(available_height / line_height), 1)
            n_cols = max((total_lines + lines_per_col - 1) // lines_per_col, 1)

        # Column geometry
        usable_width = 1 - left_margin - right_margin
        col_width = usable_width / n_cols if n_cols > 0 else usable_width

        # Draw lines column by column
        for col in range(n_cols):
            x = left_margin + col * col_width
            y = top
            start = col * lines_per_col
            end = min(start + lines_per_col, total_lines)
            for line in lines[start:end]:
                ax.text(x, y, line, fontsize=font_size, family='monospace')
                y -= line_height

        return [fig]

    def build_pdf(
        self,
        slips: List[Slip],
        answers: List[List[str]],
        filename: str,
        title: str = "Math Slips",
        difficulty: Difficulty = "",
        include_answers: bool = True,
    ) -> str:
        """Render the slips and answers into a PDF file using Matplotlib."""
        with PdfPages(filename) as pdf:
            # Slips page
            fig_slips = self._create_slips_figure(slips, difficulty, title)
            pdf.savefig(fig_slips)
            plt.close(fig_slips)
            if include_answers:
                answer_figs = self._create_answers_figures(answers, difficulty, title)
                for fig in answer_figs:
                    pdf.savefig(fig)
                    plt.close(fig)
        print(f"✅ Created PDF: {filename}")
        return filename


def _generate_and_save(generator: Callable[[Difficulty, int], Tuple[List[Slip], List[List[str]]]],
                       difficulties: List[Difficulty],
                       base_filename: str,
                       title_prefix: str) -> List[str]:
    """Internal helper to generate PDFs for all difficulties for a given generator.

    :param generator: A function that accepts (difficulty, slips_per_page)
                      and returns (slips, answers)
    :param difficulties: The list of difficulty strings to process
    :param base_filename: Base name for PDF files without extension
    :param title_prefix: Prefix used in the PDF title
    :returns: A list of generated PDF filenames
    """
    builder = SlipPDFBuilder()
    filenames = []
    for diff in difficulties:
        slips, answers = generator(diff, slips_per_page=10)
        filename = f"{base_filename}_{diff}.pdf"
        title = f"{title_prefix} – {diff.capitalize()}"
        builder.build_pdf(slips, answers, filename=filename, title=title, difficulty=diff, include_answers=True)
        filenames.append(filename)
    return filenames


if __name__ == "__main__":
    # When executed directly, generate sample PDFs for both arithmetic
    # practice and equation practice across all difficulty levels.  This
    # creates eight PDF files in the working directory.
    difficulties = ['easy', 'medium', 'hard', 'impossible']
    # Arithmetic slips
    art_gen = ArithmeticSlipGenerator()
    eq_gen = EquationSlipGenerator()
    _generate_and_save(art_gen.generate_slips, difficulties, 'arithmetic_slips', 'Arithmetic Practice')
    _generate_and_save(eq_gen.generate_slips, difficulties, 'equation_slips', 'Equation Practice')