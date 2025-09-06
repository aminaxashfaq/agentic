from __future__ import annotations

import math
from typing import Dict
from datetime import datetime
import streamlit.components.v1 as components
import ast
import re
import altair as alt

import streamlit as st


# ---------- Core logic (mirrors hello_agent) ----------
def calculate(value_a: float, value_b: float, op: str) -> float:
	if op == "add":
		return value_a + value_b
	if op == "sub":
		return value_a - value_b
	if op == "mul":
		return value_a * value_b
	if op == "div":
		if value_b == 0:
			raise ValueError("Division by zero")
		return value_a / value_b
	raise ValueError("Unsupported operation")


LENGTH_FACTORS: Dict[str, float] = {
	"mm": 0.001,
	"cm": 0.01,
	"m": 1.0,
	"km": 1000.0,
}

WEIGHT_FACTORS: Dict[str, float] = {
	"g": 0.001,
	"kg": 1.0,
	"lb": 0.45359237,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
	fu = from_unit.lower()
	tu = to_unit.lower()

	# temperature
	if fu in {"c", "celsius"} and tu in {"f", "fahrenheit"}:
		return (value * 9.0 / 5.0) + 32.0
	if fu in {"f", "fahrenheit"} and tu in {"c", "celsius"}:
		return (value - 32.0) * 5.0 / 9.0

	# length
	if fu in LENGTH_FACTORS and tu in LENGTH_FACTORS:
		return value * (LENGTH_FACTORS[fu] / LENGTH_FACTORS[tu])

	# weight
	if fu in WEIGHT_FACTORS and tu in WEIGHT_FACTORS:
		return value * (WEIGHT_FACTORS[fu] / WEIGHT_FACTORS[tu])

	raise ValueError(f"Unsupported unit conversion: {fu} -> {tu}")


# ---------- UI ----------
st.set_page_config(
	page_title="Agentic Calculator",
	page_icon="🧮",
	layout="centered",
)

# Session state and sidebar settings
if "history" not in st.session_state:
	st.session_state["history"] = []

# Memory and last answer state
if "mem" not in st.session_state:
	st.session_state["mem"] = 0.0
if "ans" not in st.session_state:
	st.session_state["ans"] = None
if "last_result" not in st.session_state:
	st.session_state["last_result"] = ""
if "last_expr" not in st.session_state:
	st.session_state["last_expr"] = ""

st.sidebar.header("⚙️ Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)
precision = st.sidebar.slider("Precision (decimals)", 0, 8, 4)

if theme == "Dark":
	bg = "#0f172a"; text = "#e5e7eb"; card = "#111827"; border = "#334155"; accent = "#10b981"
else:
	bg = "#ffffff"; text = "#111827"; card = "#f9fafb"; border = "#e5e7eb"; accent = "#10b981"

st.markdown(
	f"""
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
		:root {{ --bg: {bg}; --text: {text}; --card: {card}; --border: {border}; --accent: {accent}; }}
		* {{ font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji' !important; }}
		.big-title {{font-size: 32px; font-weight: 800; margin-bottom: 0.25rem; background: linear-gradient(135deg, var(--accent), #34d399); -webkit-background-clip: text; background-clip: text; color: transparent; text-shadow: 0 2px 12px rgba(16,185,129,0.25)}}
		.sub {{color:#6b7280; margin-bottom:1rem}}
		.result {{font-size: 24px; font-weight: 700; color: var(--accent)}}
		/* Chunky buttons with bounce + glow */
		@keyframes glow {{ from {{ box-shadow: 0 0 0 rgba(16,185,129,0.0); }} to {{ box-shadow: 0 0 22px rgba(16,185,129,0.25); }} }}
		@keyframes bounceIn {{ 0% {{ transform: translateY(0); }} 50% {{ transform: translateY(-2px); }} 100% {{ transform: translateY(0); }} }}
		.stButton>button {{
			background: var(--card) !important; color: var(--text) !important;
			border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 12px 18px;
			box-shadow: -6px -6px 12px rgba(255,255,255,0.04), 6px 6px 16px rgba(0,0,0,0.45);
			transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
		}}
		.stButton>button:hover {{ animation: bounceIn 160ms ease; filter: brightness(1.03); box-shadow: -8px -8px 16px rgba(255,255,255,0.05), 10px 10px 20px rgba(0,0,0,0.5); }}
		.stButton>button:active {{ transform: scale(0.99); box-shadow: -4px -4px 8px rgba(255,255,255,0.03), 4px 4px 10px rgba(0,0,0,0.35); }}
		/* Inputs */
		input, textarea, select {{ border-radius: 12px !important; }}
		/* Tabs */
		.stTabs [role="tab"] {{ padding: 10px 14px; border-radius: 10px; }}
		/* Wider layout */
		.block-container {{ max-width: 1280px; }}
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
	<div style=\"display:flex;align-items:center;gap:10px;margin-bottom:6px;\">
		<div class=\"big-title\" style=\"margin:0\">Agentic Calculator</div>
	</div>
	<div class=\"sub\">Free, local, and fast — calculator and unit converter</div>
	<hr style=\"border:none;border-top:1px solid var(--border);margin:8px 0 16px 0\"/>
	""",
	unsafe_allow_html=True,
)

# Sidebar history
with st.sidebar.expander("🕘 History", expanded=True):
	if st.session_state["history"]:
		for idx, item in enumerate(reversed(st.session_state["history"][-20:])):
			c0, c1, c2 = st.columns([5,1,1])
			with c0:
				st.write(f"{item['time']} · {item['label']} → {item['result']}")
			with c1:
				if st.button("Load", key=f"hist_load_{idx}"):
					st.session_state["calc_expr"] = item['label'].replace('expr ','')
					st.experimental_rerun()
			with c2:
				if st.button("Copy", key=f"hist_copy_{idx}"):
					st.session_state["calc_expr"] = item['label'].replace('expr ','')
					st.toast("Copied", icon="✅")
		if st.button("Clear history", use_container_width=True, key="clear_history"):
			st.session_state["history"] = []
			st.rerun()
	else:
		st.caption("No calculations yet.")

tabs = st.tabs(["Calculator", "Unit Converter"])  # type: ignore


with tabs[0]:
	# ---------- Separate sections: Basic | Expression | Keypad ----------
	if "calc_expr" not in st.session_state:
		st.session_state["calc_expr"] = ""

	def set_expr(val: str) -> None:
		st.session_state["calc_expr"] = val

	def add_token(tok: str) -> None:
		st.session_state["calc_expr"] += tok

	def backspace() -> None:
		st.session_state["calc_expr"] = st.session_state["calc_expr"][:-1]

	def clear_expr() -> None:
		st.session_state["calc_expr"] = ""

	# Safe evaluation
	def eval_expression(expr: str, variables: Dict[str, float] | None = None) -> float:
		allowed_names = {k: getattr(math, k) for k in [
			"sin", "cos", "tan", "asin", "acos", "atan",
			"sinh", "cosh", "tanh", "log", "log10", "sqrt",
			"exp", "pow", "pi", "e", "floor", "ceil",
			"fabs", "factorial", "gamma", "lgamma", "degrees", "radians",
			"comb", "perm"] if hasattr(math, k)}
		# Aliases and helpers
		allowed_names.update({
			"ln": math.log,
			"abs": abs,
			"round": round,
			"ncr": getattr(math, "comb", None) or (lambda n, r: math.factorial(n)//(math.factorial(r)*math.factorial(n-r))),
			"npr": getattr(math, "perm", None) or (lambda n, r: math.factorial(n)//math.factorial(n-r)),
		})
		if variables:
			allowed_names.update(variables)
		# Add helpers
		allowed_names.update({
			"ln": math.log,
			"sqrt": math.sqrt,
			"abs": abs,
			"round": round,
		})

		class SafeEval(ast.NodeVisitor):
			def visit(self, node):
				if isinstance(node, ast.Expression):
					return self.visit(node.body)
				if isinstance(node, ast.Num):
					return node.n
				if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
					left = self.visit(node.left)
					right = self.visit(node.right)
					return eval(compile(ast.Expression(ast.BinOp(ast.Num(left), node.op, ast.Num(right))), "", "eval"))
				if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
					operand = self.visit(node.operand)
					return +operand if isinstance(node.op, ast.UAdd) else -operand
				if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in allowed_names:
					args = [self.visit(a) for a in node.args]
					return allowed_names[node.func.id](*args)
				if isinstance(node, ast.Name) and node.id in allowed_names:
					return allowed_names[node.id]
				raise ValueError("Unsupported expression")

		# Preprocess: power, percent, factorial postfix
		expr = expr.replace("^", "**")
		expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
		# simple postfix factorial: 5! => factorial(5)
		expr = re.sub(r"(\d+(?:\.\d+)?|\))!", lambda m: f"factorial({m.group(1)})", expr)
		expr = expr.strip()
		if not expr:
			raise ValueError("Empty expression")
		tree = ast.parse(expr, mode="eval")
		return float(SafeEval().visit(tree))

	# Lightweight validation: unmatched parens and invalid tokens
	def validate_expression(expr: str) -> list[str]:
		errors: list[str] = []
		# Parens
		balance = 0
		for ch in expr:
			if ch == '(': balance += 1
			elif ch == ')': balance -= 1
			if balance < 0: errors.append("Unexpected ')'"); break
		if balance > 0:
			errors.append("Unmatched '('")
		# Basic token check
		if re.search(r"[^0-9a-zA-Z_\s\+\-\*/\^\(\)\.,]", expr):
			errors.append("Unknown character present")
		return errors

	# Subsections
	sec = st.tabs(["Basic", "Expression", "Keypad", "Graph", "Base Conv"])  # type: ignore

	with sec[0]:
		col1, col2 = st.columns(2)
		with col1:
			value_a = st.number_input("A", value=0.0, step=1.0, key="calc_value_a")
		with col2:
			value_b = st.number_input("B", value=0.0, step=1.0, key="calc_value_b")
		if hasattr(st, "segmented_control"):
			op = st.segmented_control("Operation", options=["add", "sub", "mul", "div"], default="add", key="calc_op")  # type: ignore[attr-defined]
		else:
			op = st.radio("Operation", options=["add", "sub", "mul", "div"], index=0, horizontal=True, key="calc_op")
		if st.button("Compute", type="primary", key="basic_compute"):
			try:
				res = calculate(value_a, value_b, op)
				st.markdown(f"<div class='result'>Result: {round(res, precision)}</div>", unsafe_allow_html=True)
				st.session_state["history"].append({
					"time": datetime.now().strftime("%H:%M:%S"),
					"label": f"{value_a} {op} {value_b}",
					"result": str(round(res, precision)),
				})
				st.session_state["ans"] = float(res)
			except Exception as e:
				st.error(str(e))

	with sec[1]:
		st.text_input("Expression", value=st.session_state["calc_expr"], key="calc_expr_input", on_change=lambda: set_expr(st.session_state["calc_expr_input"]))
		col_r0, col_r1, col_r2, col_r3, col_r4, col_r5, col_r6 = st.columns([1,1,1,1,1,1,1])
		with col_r0:
			st.metric("Mem", f"{st.session_state['mem']:.4g}")
		with col_r1:
			if st.button("Evaluate", type="primary", use_container_width=True, key="expr_eval_btn"):
				try:
					res = eval_expression(st.session_state["calc_expr"])
					st.markdown(f"<div class='result'>Result: {round(res, precision)}</div>", unsafe_allow_html=True)
					st.session_state["history"].append({
						"time": datetime.now().strftime("%H:%M:%S"),
						"label": f"expr {st.session_state['calc_expr']}",
						"result": str(round(res, precision)),
					})
					st.session_state["ans"] = float(res)
					st.session_state["last_result"] = str(round(res, precision))
					st.session_state["last_expr"] = st.session_state["calc_expr"]
				except Exception as e:
					st.error(str(e))
		with col_r2:
			if st.button("Clear", use_container_width=True, key="expr_clear_btn"):
				clear_expr()
				st.rerun()
		with col_r3:
			if st.button("Ans", use_container_width=True, key="expr_ans_btn") and st.session_state["ans"] is not None:
				st.session_state["calc_expr"] += str(st.session_state["ans"])
				st.experimental_rerun()
		with col_r4:
			if st.button("M+", use_container_width=True, key="expr_mplus_btn") and st.session_state["ans"] is not None:
				st.session_state["mem"] += float(st.session_state["ans"])
				st.experimental_rerun()
		with col_r5:
			if st.button("MR", use_container_width=True, key="expr_mr_btn"):
				st.session_state["calc_expr"] += str(st.session_state["mem"])
				st.experimental_rerun()
		with col_r6:
			if st.button("MC", use_container_width=True, key="expr_mc_btn"):
				st.session_state["mem"] = 0.0
				st.experimental_rerun()
		col_r7, col_r8 = st.columns([1,1])
		with col_r7:
			if st.button("M-", use_container_width=True, key="expr_mminus_btn") and st.session_state["ans"] is not None:
				st.session_state["mem"] -= float(st.session_state["ans"])
				st.experimental_rerun()
		with col_r8:
			if st.button("MS", use_container_width=True, key="expr_ms_btn"):
				# Save current expression value to memory if evaluable
				try:
					val = eval_expression(st.session_state["calc_expr"]) if st.session_state["calc_expr"].strip() else 0.0
					st.session_state["mem"] = float(val)
				except Exception:
					pass

	with sec[2]:
		# Always show the live expression bar above the keypad
		st.text_input("Expression", value=st.session_state["calc_expr"], key="calc_expr_input_kp", on_change=lambda: set_expr(st.session_state["calc_expr_input_kp"]))
		rows = [
			['7','8','9','/','sqrt('],
			['4','5','6','*','^'],
			['1','2','3','-','('],
			['0','.',' )','+','log('],
			['ln(','%','abs(','deg(','rad('],
		]
		for r_idx, r in enumerate(rows):
			c = st.columns(5)
			for i, label in enumerate(r):
				btn_key = f"kp_{r_idx}_{i}_{label}"
				if c[i].button(label, use_container_width=True, key=btn_key):
					add_token(label)
		c_k1, c_k2, c_k3 = st.columns(3)
		with c_k1:
			if st.button("Back", use_container_width=True, key="kp_back"):
				backspace()
		with c_k2:
			if st.button("(" , use_container_width=True, key="kp_lparen"):
				add_token("(")
			if st.button(")" , use_container_width=True, key="kp_rparen"):
				add_token(")")
		with c_k3:
			if st.button("Evaluate", type="primary", use_container_width=True, key="kp_eval_btn"):
				try:
					res = eval_expression(st.session_state["calc_expr"])
					st.success(f"Result: {round(res, precision)}")
					st.session_state["history"].append({
						"time": datetime.now().strftime("%H:%M:%S"),
						"label": f"keypad {st.session_state['calc_expr']}",
						"result": str(round(res, precision)),
					})
					st.session_state["ans"] = float(res)
				except Exception as e:
					st.error(str(e))

		st.caption("Copy:")
		cc1, cc2 = st.columns(2)
		with cc1:
			if st.button("Copy result", key="copy_res_btn"):
				st.session_state["last_result"] = st.session_state.get("last_result", "")
				st.toast("Result copied (to state)", icon="📋")
		with cc2:
			if st.button("Copy expr", key="copy_expr_btn"):
				st.session_state["last_expr"] = st.session_state.get("calc_expr", "")
				st.toast("Expression copied (to state)", icon="📋")

	with sec[3]:
		st.write("Plot y = f(x). Use math functions like sin, cos, exp. Use x in expression.")
		xmin, xmax = st.columns(2)
		with xmin:
			x_min = st.number_input("x min", value=-10.0)
		with xmax:
			x_max = st.number_input("x max", value=10.0)
		n_points = st.slider("points", 50, 1000, 400)
		y_expr = st.text_input("y =", value="sin(x)")
		if st.button("Plot", key="plot_btn"):
			try:
				xs = [x_min + (x_max - x_min) * i / (n_points - 1) for i in range(n_points)]
				ys = [eval_expression(y_expr, {"x": x}) for x in xs]
				data = {"x": xs, "y": ys}
				chart = alt.Chart(alt.Data(values=[{"x": xs[i], "y": ys[i]} for i in range(len(xs))])).mark_line().encode(
					x="x:Q", y="y:Q"
				)
				st.altair_chart(chart, use_container_width=True)
			except Exception as e:
				st.error(str(e))

	with sec[4]:
		st.write("Base conversion")
		num_str = st.text_input("Number", value="255")
		from_base = st.selectbox("From base", [2, 10, 16], index=1)
		to_base = st.selectbox("To base", [2, 10, 16], index=0)
		def parse_int(s: str, base: int) -> int:
			return int(s, base)
		def to_base_str(n: int, base: int) -> str:
			if base == 10:
				return str(n)
			if base == 2:
				return bin(n)[2:]
			if base == 16:
				return hex(n)[2:].upper()
			return str(n)
		if st.button("Convert", key="base_convert"):
			try:
				val = parse_int(num_str, from_base)
				st.success(f"Result: {to_base_str(val, to_base)}")
			except Exception as e:
				st.error(str(e))

with tabs[1]:
	col3, col4 = st.columns(2)
	with col3:
		value = st.number_input("Value", value=1.0, step=1.0, key="conv_value")
	with col4:
		from_unit = st.selectbox("From", ["mm", "cm", "m", "km", "g", "kg", "lb", "C", "F"], index=2, key="conv_from")

	to_unit = st.selectbox("To", ["mm", "cm", "m", "km", "g", "kg", "lb", "C", "F"], index=0, key="conv_to")

	# (Examples removed per request)

	if st.button("Convert", type="primary"):
		try:
			res = convert_units(value, from_unit, to_unit)
			st.markdown(f"<div class='result'>Converted: {value} {from_unit} = {res} {to_unit}</div>", unsafe_allow_html=True)
			st.session_state["history"].append({
				"time": datetime.now().strftime("%H:%M:%S"),
				"label": f"convert {value} {from_unit} → {to_unit}",
				"result": str(res),
			})
		except Exception as e:
			st.error(str(e))

# Enter key triggers primary action (Compute/Convert)
components.html(
	"""
	<script>
	  document.addEventListener('keydown', (e) => {
		const buttons = Array.from(document.querySelectorAll('button'));
		const clickByText = (txt) => {
		  const t = buttons.find(b => b.innerText.trim() === txt);
		  if (t) t.click();
		};
		if (e.key === 'Enter' && !e.isComposing) {
		  clickByText('Evaluate');
		  clickByText('Compute');
		  clickByText('Convert');
		}
		if (e.key === 'Escape') {
		  clickByText('Clear');
		}
		if (e.key.toLowerCase() === 'a') {
		  clickByText('Ans');
		}
		if (e.key.toLowerCase() === 'm' && !e.shiftKey) {
		  clickByText('M+');
		}
		if (e.key.toLowerCase() === 'm' && e.shiftKey) {
		  clickByText('M-');
		}
		if (e.key.toLowerCase() === 'r') {
		  clickByText('MR');
		}
		if (e.key.toLowerCase() === 'c') {
		  clickByText('MC');
		}
	  });
	</script>
	""",
	height=0,
)

