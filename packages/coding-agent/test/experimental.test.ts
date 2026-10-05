import { afterEach, describe, expect, it } from "vitest";
import { areExperimentalFeaturesEnabled } from "../src/core/experimental.ts";

describe("areExperimentalFeaturesEnabled", () => {
	const originalPiExperimental = process.env.JEFF_EXPERIMENTAL;

	afterEach(() => {
		if (originalPiExperimental === undefined) {
			delete process.env.JEFF_EXPERIMENTAL;
		} else {
			process.env.JEFF_EXPERIMENTAL = originalPiExperimental;
		}
	});

	it("returns false when JEFF_EXPERIMENTAL is unset", () => {
		delete process.env.JEFF_EXPERIMENTAL;

		expect(areExperimentalFeaturesEnabled()).toBe(false);
	});

	it("returns false when JEFF_EXPERIMENTAL is empty", () => {
		process.env.JEFF_EXPERIMENTAL = "";

		expect(areExperimentalFeaturesEnabled()).toBe(false);
	});

	it("returns true when JEFF_EXPERIMENTAL is set to 1", () => {
		process.env.JEFF_EXPERIMENTAL = "1";

		expect(areExperimentalFeaturesEnabled()).toBe(true);
	});

	it("returns false when JEFF_EXPERIMENTAL is set to 0", () => {
		process.env.JEFF_EXPERIMENTAL = "0";

		expect(areExperimentalFeaturesEnabled()).toBe(false);
	});

	it("returns false when JEFF_EXPERIMENTAL is set to a non-1 value", () => {
		process.env.JEFF_EXPERIMENTAL = "true";

		expect(areExperimentalFeaturesEnabled()).toBe(false);
	});
});
