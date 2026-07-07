export function add(a: number, b: number): number {
  return a + b;
}

export function double(value: number): number {
  return value * 2;
}

export function pipeline(value: number): number {
  return double(add(value, 1));
}
