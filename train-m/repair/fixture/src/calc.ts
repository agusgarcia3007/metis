export function add(a: number, b: number): number {
  return a + b;
}

export function subtract(a: number, b: number): number {
  return a - b;
}

export function scale(values: number[], factor: number): number[] {
  return values.map((v) => v * factor);
}
