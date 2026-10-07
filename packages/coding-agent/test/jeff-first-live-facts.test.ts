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

	// On Windows, //github.com/x/y.git is a network share path: stat'ing it tried to reach a server and failed with
	// UNKNOWN, ending the scout's decision (firelex/jeff-code#1). Network paths are never project files.
	it("treats network share paths as neither file nor folder", () => {
		const facts = liveFacts();
		for (const path of ["//github.com/xxx/yyy.git", "\\\\github.com\\xxx\\yyy.git", "//server/share"]) {
			expect(facts.kind(path), path).toBeUndefined();
		}
	});

	it("still knows ordinary folders whose names only start like those", () => {
		const facts = liveFacts();
		expect(facts.kind("/procedures-that-do-not-exist")).toBe("missing");
		expect(facts.kind("/tmp")).toBe("folder");
	});
});
