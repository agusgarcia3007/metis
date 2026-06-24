//! A safe recursive-descent arithmetic evaluator (+ - * / ^ %, parentheses, unary minus, decimals).
//! No eval, no code execution — the calculator the small Cortex offloads exact arithmetic to.

/// Calc evaluates an arithmetic expression and returns the exact result formatted cleanly.
pub fn calc(expr: &str) -> Result<String, String> {
    let mut p = Parser {
        src: expr.as_bytes(),
        pos: 0,
    };
    let v = p.parse_expr()?;
    p.skip_space();
    if p.pos != p.src.len() {
        let rest = std::str::from_utf8(&p.src[p.pos..]).unwrap_or("");
        return Err(format!("unexpected {rest:?}"));
    }
    if v.is_infinite() || v.is_nan() {
        return Err("result is not finite".to_string());
    }
    if v == v.trunc() && v.abs() < 1e15 {
        return Ok(format!("{}", v as i64));
    }
    Ok(format_g12(v))
}

/// Approximates Go's strconv.FormatFloat(v, 'g', 12, 64): up to 12 significant digits, trimmed.
fn format_g12(v: f64) -> String {
    let s = format!("{v:.12e}"); // e.g. 1.234567890123e2
    // Parse back and re-render compactly via Rust's shortest representation, capped to 12 sig digits.
    let rounded: f64 = s.parse().unwrap_or(v);
    let mut out = format!("{rounded}");
    if out.len() > 16 {
        out = format!("{rounded:.12}");
        while out.contains('.') && (out.ends_with('0') || out.ends_with('.')) {
            out.pop();
        }
    }
    out
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_space(&mut self) {
        while self.pos < self.src.len() && (self.src[self.pos] == b' ' || self.src[self.pos] == b'\t')
        {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> u8 {
        self.skip_space();
        if self.pos < self.src.len() {
            self.src[self.pos]
        } else {
            0
        }
    }

    // expr := term (('+'|'-') term)*
    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut v = self.parse_term()?;
        loop {
            match self.peek() {
                b'+' => {
                    self.pos += 1;
                    v += self.parse_term()?;
                }
                b'-' => {
                    self.pos += 1;
                    v -= self.parse_term()?;
                }
                _ => return Ok(v),
            }
        }
    }

    // term := power (('*'|'/'|'%') power)*
    fn parse_term(&mut self) -> Result<f64, String> {
        let mut v = self.parse_power()?;
        loop {
            match self.peek() {
                b'*' => {
                    self.pos += 1;
                    v *= self.parse_power()?;
                }
                b'/' => {
                    self.pos += 1;
                    let f = self.parse_power()?;
                    if f == 0.0 {
                        return Err("division by zero".to_string());
                    }
                    v /= f;
                }
                b'%' => {
                    self.pos += 1;
                    let f = self.parse_power()?;
                    v %= f; // f64 remainder == Go math.Mod (fmod)
                }
                _ => return Ok(v),
            }
        }
    }

    // power := unary ('^' power)?
    fn parse_power(&mut self) -> Result<f64, String> {
        let base = self.parse_unary()?;
        if self.peek() == b'^' {
            self.pos += 1;
            let exp = self.parse_power()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    // unary := ('-'|'+')? primary
    fn parse_unary(&mut self) -> Result<f64, String> {
        match self.peek() {
            b'-' => {
                self.pos += 1;
                Ok(-self.parse_unary()?)
            }
            b'+' => {
                self.pos += 1;
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    // primary := number | '(' expr ')'
    fn parse_primary(&mut self) -> Result<f64, String> {
        if self.peek() == b'(' {
            self.pos += 1;
            let v = self.parse_expr()?;
            if self.peek() != b')' {
                return Err("missing ')'".to_string());
            }
            self.pos += 1;
            return Ok(v);
        }
        self.skip_space();
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if c.is_ascii_digit()
                || c == b'.'
                || c == b'e'
                || c == b'E'
                || ((c == b'+' || c == b'-')
                    && self.pos > start
                    && (self.src[self.pos - 1] == b'e' || self.src[self.pos - 1] == b'E'))
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        let tok = std::str::from_utf8(&self.src[start..self.pos])
            .unwrap_or("")
            .trim();
        if tok.is_empty() {
            let rest = std::str::from_utf8(&self.src[start..]).unwrap_or("");
            return Err(format!("expected a number at {rest:?}"));
        }
        tok.parse::<f64>().map_err(|_| format!("invalid number {tok:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::calc;

    #[test]
    fn test_calc() {
        for (e, w) in [
            ("84937*2261", "192042557"),
            ("(5+3)/2", "4"),
            ("2^10", "1024"),
            ("10%3", "1"),
            ("-3+5", "2"),
        ] {
            let g = calc(e);
            assert!(
                matches!(&g, Ok(s) if s == w),
                "Calc({e:?})={g:?} want {w:?}"
            );
        }
    }
}
