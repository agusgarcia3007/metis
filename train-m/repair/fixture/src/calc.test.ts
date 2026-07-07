import { expect, test } from "bun:test";
import { add, subtract, scale } from "./calc";

test("add", () => { expect(add(2, 3)).toBe(5); });
test("subtract", () => { expect(subtract(10, 4)).toBe(6); });
test("scale", () => { expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]); });
