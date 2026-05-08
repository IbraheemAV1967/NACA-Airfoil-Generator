# ✈️ NACA 4-Digit Airfoil Generator

A desktop GUI application for generating and visualizing **NACA 4-digit airfoils** using Python, Tkinter, NumPy, and Matplotlib.

---

## 📸 Preview

> Launch the app and enter any 4-digit NACA code to instantly visualize the airfoil geometry — upper/lower surfaces, camber line, chord line, and key feature points.

---

## 🚀 Features

- **Interactive GUI** — Clean Tkinter interface with a welcome screen and form view.
- **Cosine-spaced point distribution** — Higher resolution near the leading edge for accurate geometry.
- **NACA 4-digit support** — Handles both **symmetric** (e.g., `0012`) and **cambered** airfoils (e.g., `2412`).
- **Live Matplotlib plot** embedded directly in the window, showing:
  - Upper surface
  - Lower surface
  - Mean camber line
  - Chord line
  - Max camber point (annotated)
  - Max thickness point (annotated)
- **Input validation** — Clear error messages for invalid NACA codes, non-positive chord lengths, or too-few points.
- **Configurable resolution** — Choose any number of points ≥ 40 for smooth rendering.

---

## 🧮 How It Works

The geometry is computed using the standard **NACA 4-digit series equations**:

Given a code `MPTT` (e.g., `2412`):

| Parameter | Symbol | Formula |
|---|---|---|
| Max camber | `m` | `M / 100` |
| Camber position | `p` | `P / 10` |
| Thickness | `t` | `TT / 100` |

**Thickness distribution:**

$$y_t = 5t \left[ 0.2969\sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4 \right]$$

**Camber line** (for `0 ≤ x < p`):

$$y_c = \frac{m}{p^2} \left(2px - x^2\right)$$

**Camber line** (for `p ≤ x ≤ 1`):

$$y_c = \frac{m}{(1-p)^2} \left(1 - 2p + 2px - x^2\right)$$

Upper and lower surface coordinates are computed by projecting thickness perpendicular to the camber line using the slope angle `θ = arctan(dyc/dx)`.

---

## 📋 Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| NumPy | ≥ 1.20 |
| Matplotlib | ≥ 3.4 |
| Tkinter | (bundled with Python) |

---

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/naca-airfoil-generator.git
   cd naca-airfoil-generator
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```

   > Tkinter is included with standard Python installations. If it's missing (e.g., on some Linux distros), install it via:
   > ```bash
   > sudo apt-get install python3-tk
   > ```

3. **Run the application:**
   ```bash
   python "Airfoil Generator.py"
   ```

---

## 🖥️ Usage

1. Launch the app — a welcome screen appears.
2. Click **Continue** to open the input form.
3. Enter:
   - **NACA 4-digit code** — e.g., `2412`, `0012`, `4415`
   - **Chord length** — any positive number (e.g., `1`, `1.5`)
   - **Number of points** — minimum `40` (default `100`)
4. Click **Generate Airfoil** to compute and display the plot.

---

## 📂 Project Structure

```
naca-airfoil-generator/
│
├── Airfoil Generator.py   # Main application (GUI + computation)
└── README.md              # Project documentation
```

---

## 📐 Input Validation Rules

| Input | Rule |
|---|---|
| NACA code | Must be exactly 4 digits |
| First two digits | Both must be `0` (symmetric) **or** both non-zero (cambered) |
| Chord length | Must be a positive number |
| Number of points | Must be ≥ 40 |

---

## 💡 Example NACA Codes

| Code | Description |
|---|---|
| `0012` | Symmetric, 12% thickness — common on tail surfaces |
| `2412` | 2% camber at 40% chord, 12% thickness — classic wing section |
| `4415` | 4% camber at 40% chord, 15% thickness — high-lift section |
| `6409` | 6% camber at 40% chord, 9% thickness — thin cambered airfoil |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or file an issue.

---

## ⭐ Acknowledgements

- NACA airfoil equations based on the original **NACA Technical Report 460** (1935).
- Visualization powered by [Matplotlib](https://matplotlib.org/).
- GUI built with Python's built-in [Tkinter](https://docs.python.org/3/library/tkinter.html).
