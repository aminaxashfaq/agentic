Agentic Calculator — Streamlit App

A free, local, and modern calculator UI with:

- Basic ops (add/sub/mul/div)
- Expression mode with math functions, combinatorics (nCr/nPr), postfix factorial (x!), percent handling, variables for graphing
- Cute keypad with animated, chunky buttons
- Memory keys: Ans, M+, M−, MR, MC, MS
- History with Load/Copy and toast feedback
- Graph panel: plot y = f(x) over a chosen x‑range
- Base conversion: bin/dec/hex
- Validation hints for unmatched parentheses and bad characters

---

Quick start (Windows, Conda)

1) Clone or open this folder in Cursor/VS Code
2) Create and activate an environment (Python 3.11 recommended)

```powershell
conda create -n agentic python=3.11 -y
conda activate agentic
```

3) Install dependencies

```powershell
pip install -r requirements.txt
```

4) Run the Streamlit app

```powershell
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
streamlit run src/app_streamlit.py --server.headless true
```

Then open http://localhost:8501 (Streamlit opens the browser automatically).

If your shell still points to another Python, explicitly call the env’s Python:

```powershell
& "C:\Users\amina\anaconda3\envs\agentic\python.exe" -m streamlit run src/app_streamlit.py --server.headless true
```

---

Features and usage

- Basic tab

  - Enter A and B, pick operation, press Compute
  - Result appears and is added to History; Ans is updated
- Expression tab

  - Write expressions like: `2*(3+4)^2 - sqrt(16) + sin(0.5)`
  - Supported: `sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, log (ln), log10, sqrt, exp, pow, pi, e, floor, ceil, abs, round`
  - Combinatorics: `ncr(n,r)`, `npr(n,r)`, or use `!` postfix like `5!`
  - Percent: `20%` becomes `0.2`
  - Buttons: Ans (append last result), M+, M−, MR, MC, MS; Mem metric shows current memory
  - Validation hints show unmatched parentheses or bad characters
- Keypad tab

  - Tap buttons to build expressions; input bar stays visible
  - Copy result / Copy expr buttons with toast feedback
- Graph tab

  - Enter `y = f(x)` (e.g., `sin(x)`, `x^2 + 3*x - 1`)
  - Choose x‑range and points, click Plot to visualize
- Base Conv tab

  - Convert between bases 2/10/16 (binary/decimal/hex)
- Keyboard shortcuts

  - Enter: Evaluate/Compute/Convert
  - Esc: Clear
  - A: Ans
  - M: M+ (Shift+M: M−)
  - R: MR
  - C: MC

---

Project structure

```
work/
  src/
    app_streamlit.py     # Main Streamlit UI
    hello_agent.py       # Minimal CLI agent used earlier
  requirements.txt       # Python deps
  README.md              # This file
```

---

