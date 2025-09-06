from __future__ import annotations

import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CalculatorRequest(BaseModel):
	value_a: float = Field(...)
	value_b: float = Field(...)
	op: str = Field(..., description="add|sub|mul|div")


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
	m = re.search(r"\bconvert\s+(-?\d+(\.\d+)?)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)", query, flags=re.IGNORECASE)
	if m:
		val = float(m.group(1))
		from_u = m.group(3)
		to_u = m.group(4)
		return UnitConvertRequest(value=val, from_unit=from_u, to_unit=to_u)
	return None


class AskRequest(BaseModel):
	query: str


class AskResponse(BaseModel):
	type: str
	result: str


app = FastAPI(title="Agentic Hello API", version="0.1.0")


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
	query = req.query.strip()
	calc_req = parse_calculation(query)
	if calc_req:
		value = calculate(calc_req)
		return AskResponse(type="calc", result=str(value))

	conv_req = parse_conversion(query)
	if conv_req:
		value = convert_units(conv_req)
		return AskResponse(type="convert", result=str(value))

	return AskResponse(type="unknown", result="Try 'add 5 and 7' or 'convert 10 cm to m'.")

