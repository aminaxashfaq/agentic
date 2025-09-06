from __future__ import annotations

import re
import sys
from typing import Optional, Tuple

import typer
from rich.console import Console
from pydantic import BaseModel, Field

app = typer.Typer(add_completion=False)
console = Console()

class CalculatorRequest(BaseModel):
	value_a: float = Field(..., description="First number")
	value_b: float = Field(..., description="Second number")
	op: str = Field(..., description="Operation: add|sub|mul|div")

class UnitConvertRequest(BaseModel):
	value: float
	from_unit: str
	to_unit: str

LENGTH_FACTORS = {
	"mm": 0.001,
	"cm": 0.01,
	"m": 1.0,
	"km": 1000.0,
}

WEIGHT_FACTORS = {
	"g": 0.001,
	"kg": 1.0,
	"lb": 0.45359237,
}

def calculate(req: CalculatorRequest) -> float:
	if req.op == "add":
		return req.value_a + req.value_b
	if req.op == "sub":
		return req.value_a - req.value_b
	if req.op == "mul":
		return req.value_a * req.value_b
	if req.op == "div":
		if req.value_b == 0:
			raise ValueError("Division by zero")
		return req.value_a / req.value_b
	raise ValueError("Unsupported operation")

def convert_units(req: UnitConvertRequest) -> float:
	fu = req.from_unit.lower()
	tu = req.to_unit.lower()

	# temperature
	if fu in {"c", "celsius"} and tu in {"f", "fahrenheit"}:
		return (req.value * 9 / 5) + 32
	if fu in {"f", "fahrenheit"} and tu in {"c", "celsius"}:
		return (req.value - 32) * 5 / 9

	# length
	if fu in LENGTH_FACTORS and tu in LENGTH_FACTORS:
		return req.value * (LENGTH_FACTORS[fu] / LENGTH_FACTORS[tu])

	# weight
	if fu in WEIGHT_FACTORS and tu in WEIGHT_FACTORS:
		return req.value * (WEIGHT_FACTORS[fu] / WEIGHT_FACTORS[tu])

	raise ValueError(f"Unsupported unit conversion: {fu} -> {tu}")

def parse_calculation(query: str) -> Optional[CalculatorRequest]:
	patterns = [
		(r"\badd\s+(-?\d+(\.\d+)?)\s+and\s+(-?\d+(\.\d+)?)", "add"),
		(r"\bsubtract\s+(-?\d+(\.\d+)?)\s+from\s+(-?\d+(\.\d+)?)", "sub_rev"),
		(r"\bsub\s+(-?\d+(\.\d+)?)\s+and\s+(-?\d+(\.\d+)?)", "sub"),
		(r"\bmultiply\s+(-?\d+(\.\d+)?)\s+by\s+(-?\d+(\.\d+)?)", "mul"),
		(r"\bdivide\s+(-?\d+(\.\d+)?)\s+by\s+(-?\d+(\.\d+)?)", "div"),
	]
	for pattern, op in patterns:
		m = re.search(pattern, query, flags=re.IGNORECASE)
		if m:
			a = float(m.group(1))
			b = float(m.group(3))
			if op == "sub_rev":
				return CalculatorRequest(value_a=b, value_b=a, op="sub")
			return CalculatorRequest(value_a=a, value_b=b, op=op)
	return None

def parse_conversion(query: str) -> Optional[UnitConvertRequest]:
	# e.g., "convert 10 cm to m" or "convert 72 F to C"
	m = re.search(r"\bconvert\s+(-?\d+(\.\d+)?)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)", query, flags=re.IGNORECASE)
	if m:
		val = float(m.group(1))
		from_u = m.group(3)
		to_u = m.group(4)
		return UnitConvertRequest(value=val, from_unit=from_u, to_unit=to_u)
	return None


@app.command()
def ask(query: str = typer.Argument(..., help="Ask in natural language, e.g. 'add 5 and 7' or 'convert 10 cm to m'")):
	query = query.strip()
	try:
		calc_req = parse_calculation(query)
		if calc_req:
			result = calculate(calc_req)
			typer.echo(f"Result: {result}")
			return

		conv_req = parse_conversion(query)
		if conv_req:
			result = convert_units(conv_req)
			typer.echo(f"Converted: {conv_req.value} {conv_req.from_unit} = {result} {conv_req.to_unit}")
			return

		typer.echo("I didn't understand. Try 'add 5 and 7' or 'convert 10 cm to m'.")
	except Exception as e:
		typer.echo(f"Error: {e}")
		sys.exit(1)

if __name__ == "__main__":
	# Fallback runner: allow "python hello_agent.py 'query'" or "python hello_agent.py ask 'query'"
	args = sys.argv[1:]
	if args:
		if args[0].lower() == "ask":
			args = args[1:]
		query = " ".join(args)
		try:
			ask(query)
		except SystemExit:
			# Ignore Typer's SystemExit from error path to avoid noisy tracebacks
			pass
	else:
		# No args: show a tiny usage hint
		typer.echo("Usage: python src/hello_agent.py ask \"add 5 and 7\"  |  python src/hello_agent.py \"add 5 and 7\"")