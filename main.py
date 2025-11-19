from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import random
import datetime
class SimpleSlipGenerator:
    def __init__(self, slips_per_page=10, questions_per_slip=10, seed=None):
        self.slips_per_page = slips_per_page
        self.questions_per_slip = questions_per_slip
        if seed is not None:
            random.seed(seed)

    def generate_short_problem(self):

        ptype = random.choice(["×", "÷", "+", "-", "%"])
        a = random.randint(10, 999)
        b = random.randint(2, 99)

        if ptype == "×":
            question = f"{a} × {b} ="
            answer = a * b
        elif ptype == "÷":
            # make division clean sometimes
            answer = random.randint(1,25 )
            question = f"{a*answer} ÷ {answer} ="
            answer = a
        elif ptype == "+":
            question = f"{a} + {b} ="
            answer = a + b
        elif ptype == "-":
            if b > a:
                a, b = b, a
            question = f"{a} − {b} ="
            answer = a - b
        else:  # percentage
            question = f"{b}% of {a} ="
            answer = round(a * b / 100, 2)

        return question, answer

    def make_pdf(self, filename="simple_slips.pdf", include_answers=False):
        """Generate an A4 PDF with 10 slips (2×5 grid)."""
        PAGE_W, PAGE_H = A4
        c = canvas.Canvas(filename, pagesize=A4)

        cols, rows = 2, 5
        margin = 15 * mm
        gutter_x, gutter_y = 6 * mm, 6 * mm
        usable_w = PAGE_W - 2 * margin - gutter_x * (cols - 1)
        usable_h = PAGE_H - 2 * margin - gutter_y * (rows - 1)
        cell_w = usable_w / cols
        cell_h = usable_h / rows

        c.setFont("Helvetica", 11)

        all_slips = []
        for s in range(self.slips_per_page):
            slip = [self.generate_short_problem() for _ in range(self.questions_per_slip)]
            all_slips.append(slip)

        for i, slip in enumerate(all_slips):
            col, row = i % cols, i // cols
            x = margin + col * (cell_w + gutter_x)
            y_top = PAGE_H - margin - row * (cell_h + gutter_y)

            # Draw border
            c.rect(x, y_top - cell_h, cell_w, cell_h)

            # Title
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x + 4 * mm, y_top - 6 * mm, f"Slip {i+1}")
            c.setFont("Helvetica", 9)

            # Problems
            y = y_top - 15 * mm
            for q, _ in slip:
                c.drawString(x + 8 * mm, y, q)
                y -= 10  # spacing between lines

        if include_answers:
            c.showPage()
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, PAGE_H - margin, "Answers")
            y = PAGE_H - margin - 15 * mm
            c.setFont("Helvetica", 9)
            for i, slip in enumerate(all_slips):
                c.drawString(margin, y, f"Slip {i+1}")
                y -= 8
                for q_idx, (_, a) in enumerate(slip, 1):
                    c.drawString(margin + 10 * mm, y, f"{q_idx}) {a}")
                    y -= 8
                y -= 5
                if y < 20 * mm:
                    c.showPage()
                    y = PAGE_H - margin

        c.save()
        print(f"✅ PDF created: {filename}")
        return filename


# Example usage:
if __name__ == "__main__":
    filename = datetime.date.strftime(datetime.date.today(), "%Y%m%d.pdf")
    gen = SimpleSlipGenerator()
    gen.make_pdf(filename=filename, include_answers=True)
