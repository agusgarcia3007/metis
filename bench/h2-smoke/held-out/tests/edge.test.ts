import { describe, expect, it } from "vitest";
import { add, double, pipeline } from "../src/math";

describe("held-out edge cases", () => {
  it("handles negative and zero values", () => {
    expect(add(-4, 4)).toBe(0);
    expect(double(0)).toBe(0);
    expect(pipeline(-1)).toBe(0);
  });
});
