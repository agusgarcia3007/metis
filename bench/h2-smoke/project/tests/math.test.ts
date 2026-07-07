import { describe, expect, it } from "vitest";
import { add, double, pipeline } from "../src/math";

describe("math pipeline", () => {
  it("adds two values", () => {
    expect(add(4, 3)).toBe(7);
  });

  it("doubles a value", () => {
    expect(double(5)).toBe(10);
  });

  it("composes both operations", () => {
    expect(pipeline(4)).toBe(10);
  });
});
