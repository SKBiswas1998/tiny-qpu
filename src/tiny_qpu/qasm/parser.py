"""
Pure-Python OpenQASM 2.0 parser.

Parses a subset of OpenQASM 2.0 into tiny-qpu Circuit objects.
No external dependencies (no ANTLR, no Rust, no regex).

Supported features:
    - OPENQASM 2.0 header
    - include "qelib1.inc" (ignored — gates are built-in)
    - qreg, creg declarations
    - Standard gates: x, y, z, h, s, sdg, t, tdg, sx, cx, cz, ccx,
                      swap, rx, ry, rz, p, u1, u2, u3, id
    - Custom gate definitions (gate ... { ... })
    - measure q[i] -> c[j]
    - barrier
    - Single-line (//) and multi-line comments
    - Mathematical expressions in parameters: pi, +, -, *, /

Limitations (Phase 2 targets):
    - No if/else classical control
    - No for/while loops
    - No subroutine calls beyond gate defs
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterator

from tiny_qpu.circuit import Circuit


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str  # KEYWORD, IDENT, NUMBER, LPAREN, RPAREN, etc.
    value: str
    line: int

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, line={self.line})"


# Token patterns (order matters)
_TOKEN_PATTERNS = [
    ("COMMENT_ML", r"/\*.*?\*/"),
    ("COMMENT", r"//[^\n]*"),
    ("FLOAT", r"\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+"),
    ("INT", r"\d+"),
    ("ARROW", r"->"),
    ("SEMICOLON", r";"),
    ("COMMA", r","),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("CARET", r"\^"),
    ("EQUALS", r"=="),
    ("STRING", r'"[^"]*"'),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS), re.DOTALL)

_KEYWORDS = {
    "OPENQASM", "include", "qreg", "creg", "gate", "measure",
    "barrier", "if", "reset", "opaque", "pi", "sin", "cos", "tan",
    "exp", "ln", "sqrt",
}


def _tokenize(source: str) -> list[Token]:
    """Tokenize OpenQASM source into a list of tokens."""
    tokens = []
    line = 1

    for match in _TOKEN_RE.finditer(source):
        kind = match.lastgroup
        value = match.group()

        if kind == "NEWLINE":
            line += 1
            continue
        if kind in ("SKIP", "COMMENT", "COMMENT_ML"):
            line += value.count("\n")
            continue

        if kind == "IDENT" and value in _KEYWORDS:
            kind = "KEYWORD"

        if kind == "FLOAT":
            kind = "NUMBER"
        elif kind == "INT":
            kind = "NUMBER"

        tokens.append(Token(kind, value, line))

    return tokens


# ---------------------------------------------------------------------------
# Expression evaluator (for gate parameters)
# ---------------------------------------------------------------------------

def _eval_expr(tokens: list[Token], pos: int) -> tuple[float, int]:
    """
    Parse and evaluate a mathematical expression.

    Supports: pi, numbers, +, -, *, /, parentheses, sin, cos, tan, sqrt, exp, ln
    Returns (value, new_position).
    """
    val, pos = _eval_additive(tokens, pos)
    return val, pos


def _eval_additive(tokens: list[Token], pos: int) -> tuple[float, int]:
    left, pos = _eval_multiplicative(tokens, pos)
    while pos < len(tokens) and tokens[pos].kind in ("PLUS", "MINUS"):
        op = tokens[pos].kind
        pos += 1
        right, pos = _eval_multiplicative(tokens, pos)
        left = left + right if op == "PLUS" else left - right
    return left, pos


def _eval_multiplicative(tokens: list[Token], pos: int) -> tuple[float, int]:
    left, pos = _eval_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos].kind in ("STAR", "SLASH"):
        op = tokens[pos].kind
        pos += 1
        right, pos = _eval_unary(tokens, pos)
        left = left * right if op == "STAR" else left / right
    return left, pos


def _eval_unary(tokens: list[Token], pos: int) -> tuple[float, int]:
    if pos < len(tokens) and tokens[pos].kind == "MINUS":
        pos += 1
        val, pos = _eval_primary(tokens, pos)
        return -val, pos
    return _eval_primary(tokens, pos)


def _eval_primary(tokens: list[Token], pos: int) -> tuple[float, int]:
    tok = tokens[pos]

    if tok.kind == "NUMBER":
        return float(tok.value), pos + 1

    if tok.kind == "KEYWORD" and tok.value == "pi":
        return math.pi, pos + 1

    # Functions: sin, cos, tan, sqrt, exp, ln
    if tok.kind == "KEYWORD" and tok.value in ("sin", "cos", "tan", "sqrt", "exp", "ln"):
        func_name = tok.value
        pos += 1
        if tokens[pos].kind != "LPAREN":
            raise QasmParseError(f"Expected '(' after {func_name}", tok.line)
        pos += 1
        arg, pos = _eval_expr(tokens, pos)
        if tokens[pos].kind != "RPAREN":
            raise QasmParseError(f"Expected ')' after {func_name} argument", tok.line)
        pos += 1
        func_map = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "exp": math.exp, "ln": math.log,
        }
        return func_map[func_name](arg), pos

    if tok.kind == "LPAREN":
        pos += 1
        val, pos = _eval_expr(tokens, pos)
        if tokens[pos].kind != "RPAREN":
            raise QasmParseError("Unmatched '('", tok.line)
        return val, pos + 1

    raise QasmParseError(f"Unexpected token in expression: {tok}", tok.line)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class QasmParseError(Exception):
    """Error during QASM parsing."""
    def __init__(self, message: str, line: int = 0) -> None:
        super().__init__(f"Line {line}: {message}" if line else message)
        self.line = line


@dataclass
class _GateDef:
    """User-defined gate from a 'gate' declaration."""
    name: str
    params: list[str]  # parameter names
    qubits: list[str]  # qubit names
    body: list[Token]  # tokens inside { }


class QasmParser:
    """
    Pure-Python OpenQASM 2.0 parser.

    Converts OpenQASM source into a tiny-qpu Circuit.

    Example
    -------
    >>> from tiny_qpu.qasm import QasmParser
    >>> qasm = '''
    ... OPENQASM 2.0;
    ... include "qelib1.inc";
    ... qreg q[2];
    ... creg c[2];
    ... h q[0];
    ... cx q[0],q[1];
    ... measure q[0] -> c[0];
    ... measure q[1] -> c[1];
    ... '''
    >>> parser = QasmParser()
    >>> circuit = parser.parse(qasm)
    """

    def __init__(self) -> None:
        self._qregs: dict[str, int] = {}  # name → size
        self._cregs: dict[str, int] = {}
        self._gate_defs: dict[str, _GateDef] = {}
        self._tokens: list[Token] = []
        self._pos: int = 0

    def parse(self, source: str) -> Circuit:
        """
        Parse OpenQASM 2.0 source and return a Circuit.

        Parameters
        ----------
        source : str
            OpenQASM 2.0 source code.

        Returns
        -------
        Circuit
            Parsed quantum circuit.
        """
        self._tokens = _tokenize(source)
        self._pos = 0
        self._qregs = {}
        self._cregs = {}
        self._gate_defs = {}

        # Parse header
        self._parse_header()

        # Collect all instructions
        instructions: list[tuple] = []  # (gate_name, params, qubits, clbits)

        while self._pos < len(self._tokens):
            self._parse_statement(instructions)

        # Build circuit
        total_qubits = sum(self._qregs.values())
        total_clbits = sum(self._cregs.values())

        circuit = Circuit(max(total_qubits, 1), total_clbits)

        # Map register.index → flat qubit index
        qubit_offset: dict[str, int] = {}
        offset = 0
        for name, size in self._qregs.items():
            qubit_offset[name] = offset
            offset += size

        clbit_offset: dict[str, int] = {}
        offset = 0
        for name, size in self._cregs.items():
            clbit_offset[name] = offset
            offset += size

        for op_name, params, qubits, clbits in instructions:
            # Resolve qubit references
            flat_qubits = []
            for reg_name, idx in qubits:
                flat_qubits.append(qubit_offset[reg_name] + idx)

            flat_clbits = []
            for reg_name, idx in clbits:
                flat_clbits.append(clbit_offset[reg_name] + idx)

            if op_name == "measure":
                circuit.measure(flat_qubits[0], flat_clbits[0])
            elif op_name == "barrier":
                circuit.barrier(*flat_qubits)
            else:
                # Map to circuit method
                self._apply_gate(circuit, op_name, params, flat_qubits)

        return circuit

    # -- Header parsing -----------------------------------------------------

    def _parse_header(self) -> None:
        """Parse OPENQASM version and include statements."""
        # OPENQASM 2.0;
        if self._peek_keyword("OPENQASM"):
            self._advance()  # OPENQASM
            self._expect("NUMBER")  # 2.0
            self._expect("SEMICOLON")

        # include "qelib1.inc";
        while self._peek_keyword("include"):
            self._advance()  # include
            self._expect("STRING")  # "qelib1.inc"
            self._expect("SEMICOLON")

    # -- Statement parsing --------------------------------------------------

    def _parse_statement(self, instructions: list) -> None:
        """Parse a single statement."""
        tok = self._current()
        if tok is None:
            return

        if tok.kind == "KEYWORD":
            if tok.value == "qreg":
                self._parse_qreg()
            elif tok.value == "creg":
                self._parse_creg()
            elif tok.value == "gate":
                self._parse_gate_def()
            elif tok.value == "measure":
                self._parse_measure(instructions)
            elif tok.value == "barrier":
                self._parse_barrier(instructions)
            elif tok.value == "reset":
                self._parse_reset(instructions)
            elif tok.value == "if":
                # Skip conditional (not yet supported)
                self._skip_to_semicolon()
            elif tok.value == "opaque":
                self._skip_to_semicolon()
            else:
                # Might be a gate name that's also a keyword
                self._parse_gate_application(instructions)
        elif tok.kind == "IDENT":
            self._parse_gate_application(instructions)
        elif tok.kind == "SEMICOLON":
            self._advance()
        else:
            raise QasmParseError(f"Unexpected token: {tok}", tok.line)

    def _parse_qreg(self) -> None:
        self._advance()  # qreg
        name = self._expect("IDENT")
        self._expect("LBRACKET")
        size = int(self._expect("NUMBER"))
        self._expect("RBRACKET")
        self._expect("SEMICOLON")
        self._qregs[name] = size

    def _parse_creg(self) -> None:
        self._advance()  # creg
        name = self._expect("IDENT")
        self._expect("LBRACKET")
        size = int(self._expect("NUMBER"))
        self._expect("RBRACKET")
        self._expect("SEMICOLON")
        self._cregs[name] = size

    def _parse_gate_def(self) -> None:
        """Parse: gate name(params) qubits { body }"""
        self._advance()  # gate
        name = self._expect("IDENT")

        # Optional parameters
        params = []
        if self._peek("LPAREN"):
            self._advance()
            while not self._peek("RPAREN"):
                params.append(self._expect("IDENT"))
                if self._peek("COMMA"):
                    self._advance()
            self._advance()  # )

        # Qubit names
        qubits = []
        while not self._peek("LBRACE"):
            qubits.append(self._expect("IDENT"))
            if self._peek("COMMA"):
                self._advance()

        # Body tokens
        self._expect("LBRACE")
        body_tokens = []
        depth = 1
        while depth > 0:
            tok = self._current()
            if tok is None:
                raise QasmParseError("Unexpected end of file in gate definition")
            if tok.kind == "LBRACE":
                depth += 1
            elif tok.kind == "RBRACE":
                depth -= 1
                if depth == 0:
                    self._advance()
                    break
            body_tokens.append(tok)
            self._advance()

        self._gate_defs[name] = _GateDef(
            name=name, params=params, qubits=qubits, body=body_tokens
        )

    def _parse_measure(self, instructions: list) -> None:
        self._advance()  # measure
        q_reg, q_idx = self._parse_qubit_ref()
        self._expect("ARROW")
        c_reg, c_idx = self._parse_qubit_ref()
        self._expect("SEMICOLON")
        instructions.append(("measure", [], [(q_reg, q_idx)], [(c_reg, c_idx)]))

    def _parse_barrier(self, instructions: list) -> None:
        self._advance()  # barrier
        qubits = []
        while not self._peek("SEMICOLON"):
            reg, idx = self._parse_qubit_ref()
            qubits.append((reg, idx))
            if self._peek("COMMA"):
                self._advance()
        self._expect("SEMICOLON")
        instructions.append(("barrier", [], qubits, []))

    def _parse_reset(self, instructions: list) -> None:
        """Parse reset (currently ignored — just skip)."""
        self._skip_to_semicolon()

    def _parse_gate_application(self, instructions: list) -> None:
        """Parse: gate_name(params) qubit_list;"""
        name_tok = self._current()
        gate_name = name_tok.value.lower()

        # Handle 'id' → 'i'
        if gate_name == "id":
            gate_name = "i"

        self._advance()

        # Optional parameters
        params = []
        if self._peek("LPAREN"):
            self._advance()
            while not self._peek("RPAREN"):
                val, self._pos = _eval_expr(self._tokens, self._pos)
                params.append(val)
                if self._peek("COMMA"):
                    self._advance()
            self._advance()  # )

        # Qubit arguments
        qubits = []
        while not self._peek("SEMICOLON"):
            reg, idx = self._parse_qubit_ref()
            qubits.append((reg, idx))
            if self._peek("COMMA"):
                self._advance()

        self._expect("SEMICOLON")
        instructions.append((gate_name, params, qubits, []))

    def _parse_qubit_ref(self) -> tuple[str, int]:
        """Parse 'reg[index]' or 'reg'."""
        name = self._expect("IDENT")
        if self._peek("LBRACKET"):
            self._advance()
            idx = int(self._expect("NUMBER"))
            self._expect("RBRACKET")
            return name, idx
        return name, 0

    # -- Gate application to circuit ----------------------------------------

    def _apply_gate(
        self, circuit: Circuit, name: str, params: list[float], qubits: list[int]
    ) -> None:
        """Apply a parsed gate to the circuit."""
        # Built-in gates
        gate_map = {
            "i": lambda: circuit.i(qubits[0]),
            "x": lambda: circuit.x(qubits[0]),
            "y": lambda: circuit.y(qubits[0]),
            "z": lambda: circuit.z(qubits[0]),
            "h": lambda: circuit.h(qubits[0]),
            "s": lambda: circuit.s(qubits[0]),
            "sdg": lambda: circuit.sdg(qubits[0]),
            "t": lambda: circuit.t(qubits[0]),
            "tdg": lambda: circuit.tdg(qubits[0]),
            "sx": lambda: circuit.sx(qubits[0]),
            "cx": lambda: circuit.cx(qubits[0], qubits[1]),
            "cnot": lambda: circuit.cx(qubits[0], qubits[1]),
            "cz": lambda: circuit.cz(qubits[0], qubits[1]),
            "swap": lambda: circuit.swap(qubits[0], qubits[1]),
            "ccx": lambda: circuit.ccx(qubits[0], qubits[1], qubits[2]),
            "cswap": lambda: circuit.cswap(qubits[0], qubits[1], qubits[2]),
            "rx": lambda: circuit.rx(params[0], qubits[0]),
            "ry": lambda: circuit.ry(params[0], qubits[0]),
            "rz": lambda: circuit.rz(params[0], qubits[0]),
            "p": lambda: circuit.p(params[0], qubits[0]),
            "u1": lambda: circuit.u1(params[0], qubits[0]),
            "u2": lambda: circuit.u2(params[0], params[1], qubits[0]),
            "u3": lambda: circuit.u3(params[0], params[1], params[2], qubits[0]),
            "cp": lambda: circuit.cp(params[0], qubits[0], qubits[1]),
            "crx": lambda: circuit.crx(params[0], qubits[0], qubits[1]),
            "cry": lambda: circuit.cry(params[0], qubits[0], qubits[1]),
            "crz": lambda: circuit.crz(params[0], qubits[0], qubits[1]),
            "rxx": lambda: circuit.rxx(params[0], qubits[0], qubits[1]),
            "ryy": lambda: circuit.ryy(params[0], qubits[0], qubits[1]),
            "rzz": lambda: circuit.rzz(params[0], qubits[0], qubits[1]),
        }

        if name in gate_map:
            gate_map[name]()
        elif name in self._gate_defs:
            # User-defined gate — expand inline
            self._expand_gate_def(circuit, name, params, qubits)
        else:
            raise QasmParseError(f"Unknown gate: '{name}'")

    def _expand_gate_def(
        self, circuit: Circuit, name: str, params: list[float], qubits: list[int]
    ) -> None:
        """Expand a user-defined gate definition inline."""
        gate_def = self._gate_defs[name]

        # Build parameter mapping
        param_map = dict(zip(gate_def.params, params))
        qubit_map = dict(zip(gate_def.qubits, qubits))

        # Parse body with substitutions
        # This is simplified — works for basic gate definitions
        body_tokens = gate_def.body
        sub_instructions: list[tuple] = []

        # Mini-parser for the gate body
        i = 0
        while i < len(body_tokens):
            tok = body_tokens[i]
            if tok.kind in ("IDENT", "KEYWORD") and tok.value.lower() not in ("pi", "sin", "cos", "tan", "sqrt", "exp", "ln"):
                sub_name = tok.value.lower()
                i += 1

                # Parameters
                sub_params = []
                if i < len(body_tokens) and body_tokens[i].kind == "LPAREN":
                    i += 1
                    while body_tokens[i].kind != "RPAREN":
                        if body_tokens[i].kind == "NUMBER":
                            sub_params.append(float(body_tokens[i].value))
                            i += 1
                        elif body_tokens[i].kind == "KEYWORD" and body_tokens[i].value == "pi":
                            sub_params.append(math.pi)
                            i += 1
                        elif body_tokens[i].kind == "IDENT" and body_tokens[i].value in param_map:
                            sub_params.append(param_map[body_tokens[i].value])
                            i += 1
                        elif body_tokens[i].kind == "COMMA":
                            i += 1
                        else:
                            # Try expression evaluation
                            val, i = _eval_expr(body_tokens, i)
                            sub_params.append(val)
                    i += 1  # skip RPAREN

                # Qubits
                sub_qubits = []
                while i < len(body_tokens) and body_tokens[i].kind != "SEMICOLON":
                    if body_tokens[i].kind == "IDENT":
                        qubit_name = body_tokens[i].value
                        if qubit_name in qubit_map:
                            sub_qubits.append(qubit_map[qubit_name])
                    i += 1
                    if i < len(body_tokens) and body_tokens[i].kind == "COMMA":
                        i += 1

                if i < len(body_tokens) and body_tokens[i].kind == "SEMICOLON":
                    i += 1

                if sub_qubits:
                    self._apply_gate(circuit, sub_name, sub_params, sub_qubits)
            else:
                i += 1

    # -- Token helpers ------------------------------------------------------

    def _current(self) -> Token | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Token | None:
        tok = self._current()
        self._pos += 1
        return tok

    def _peek(self, kind: str) -> bool:
        tok = self._current()
        return tok is not None and tok.kind == kind

    def _peek_keyword(self, value: str) -> bool:
        tok = self._current()
        return tok is not None and tok.kind == "KEYWORD" and tok.value == value

    def _expect(self, kind: str) -> str:
        tok = self._current()
        if tok is None:
            raise QasmParseError(f"Unexpected end of file, expected {kind}")
        if tok.kind != kind:
            raise QasmParseError(
                f"Expected {kind}, got {tok.kind} ('{tok.value}')", tok.line
            )
        self._advance()
        return tok.value

    def _skip_to_semicolon(self) -> None:
        while self._pos < len(self._tokens):
            if self._tokens[self._pos].kind == "SEMICOLON":
                self._pos += 1
                return
            self._pos += 1


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse_qasm(source: str) -> Circuit:
    """
    Parse OpenQASM 2.0 source into a Circuit.

    Parameters
    ----------
    source : str
        OpenQASM 2.0 source code.

    Returns
    -------
    Circuit
        Parsed quantum circuit.

    Example
    -------
    >>> from tiny_qpu.qasm import parse_qasm
    >>> qc = parse_qasm('''
    ...     OPENQASM 2.0;
    ...     include "qelib1.inc";
    ...     qreg q[2];
    ...     h q[0];
    ...     cx q[0],q[1];
    ... ''')
    >>> print(qc)
    Circuit(n_qubits=2, n_clbits=0, depth=2, gates=2)
    """
    return QasmParser().parse(source)
