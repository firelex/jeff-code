import { describe, expect, it } from "vitest";
import { liveFacts } from "../src/core/jeff-first/facts.ts";

describe("JeffFirst live facts", () => {
	// Reading /proc/sysrq-trigger to check whether it is text failed with EIO and ended a session's menu building
	// (headless-terminal, 2026-10-04). Files under the kernel's pseudo file systems are never offered as files.
	it("treats paths under /proc, /sys and /dev as neither file nor folder", () => {
		const facts = liveFacts();
		for (const path of [
			"/proc",
			"/proc/sysrq-trigger",
			"/proc/self/status",
			"/sys",
			"/sys/kernel",
			"/dev",
			"/dev/null",
		]) {
			expect(facts.kind(path), path).toBeUndefined();
		}
	});

	it("still knows ordinary folders whose names only start like those", () => {
		const facts = liveFacts();
		expect(facts.kind("/procedures-that-do-not-exist")).toBe("missing");
		expect(facts.kind("/tmp")).toBe("folder");
	});
});
