// Package hands implements tiny-llm's tools — the capabilities the small Cortex offloads to exact,
// deterministic code (the "Hands" of the CLH-C architecture). A calculator is the canonical example:
// small LLMs are unreliable at exact arithmetic, so we let the model call this instead.
package hands

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Calc evaluates an arithmetic expression (+ - * / ^ %, parentheses, unary minus, decimals) and
// returns the exact result formatted cleanly. It is a safe recursive-descent parser — no eval, no
// code execution.
func Calc(expr string) (string, error) {
	p := &parser{src: expr}
	v, err := p.parseExpr()
	if err != nil {
		return "", err
	}
	p.skipSpace()
	if p.pos != len(p.src) {
		return "", fmt.Errorf("unexpected %q", p.src[p.pos:])
	}
	if math.IsInf(v, 0) || math.IsNaN(v) {
		return "", fmt.Errorf("result is not finite")
	}
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10), nil
	}
	return strconv.FormatFloat(v, 'g', 12, 64), nil
}

type parser struct {
	src string
	pos int
}

func (p *parser) skipSpace() {
	for p.pos < len(p.src) && (p.src[p.pos] == ' ' || p.src[p.pos] == '\t') {
		p.pos++
	}
}

func (p *parser) peek() byte {
	p.skipSpace()
	if p.pos < len(p.src) {
		return p.src[p.pos]
	}
	return 0
}

// expr := term (('+'|'-') term)*
func (p *parser) parseExpr() (float64, error) {
	v, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		switch p.peek() {
		case '+':
			p.pos++
			t, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			v += t
		case '-':
			p.pos++
			t, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			v -= t
		default:
			return v, nil
		}
	}
}

// term := power (('*'|'/'|'%') power)*
func (p *parser) parseTerm() (float64, error) {
	v, err := p.parsePower()
	if err != nil {
		return 0, err
	}
	for {
		switch p.peek() {
		case '*':
			p.pos++
			f, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			v *= f
		case '/':
			p.pos++
			f, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			if f == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			v /= f
		case '%':
			p.pos++
			f, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			v = math.Mod(v, f)
		default:
			return v, nil
		}
	}
}

// power := unary ('^' power)?
func (p *parser) parsePower() (float64, error) {
	base, err := p.parseUnary()
	if err != nil {
		return 0, err
	}
	if p.peek() == '^' {
		p.pos++
		exp, err := p.parsePower()
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

// unary := ('-'|'+')? primary
func (p *parser) parseUnary() (float64, error) {
	switch p.peek() {
	case '-':
		p.pos++
		v, err := p.parseUnary()
		return -v, err
	case '+':
		p.pos++
		return p.parseUnary()
	}
	return p.parsePrimary()
}

// primary := number | '(' expr ')'
func (p *parser) parsePrimary() (float64, error) {
	if p.peek() == '(' {
		p.pos++
		v, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("missing ')'")
		}
		p.pos++
		return v, nil
	}
	p.skipSpace()
	start := p.pos
	for p.pos < len(p.src) {
		c := p.src[p.pos]
		if (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
			((c == '+' || c == '-') && p.pos > start && (p.src[p.pos-1] == 'e' || p.src[p.pos-1] == 'E')) {
			p.pos++
		} else {
			break
		}
	}
	tok := strings.TrimSpace(p.src[start:p.pos])
	if tok == "" {
		return 0, fmt.Errorf("expected a number at %q", p.src[start:])
	}
	v, err := strconv.ParseFloat(tok, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number %q", tok)
	}
	return v, nil
}
