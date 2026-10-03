#!/usr/bin/env node
/**
 * Builds JeffFirst menus (the scout's option lists) for points in past sessions whose machines are gone, with the
 * same list code as live runs, reading facts rebuilt from the transcript instead of a disk.
 *
 * Usage: node scripts/jeff-first-menus.ts [--live] < points.jsonl > menus.jsonl
 *
 * Each input line is one point before a coding-model turn:
 *   {id, cwd, task, steps: [{command, output, byScout}], events: [FactEvent...], activeTools: ["bash"],
 *    runApproval: "all" | "seen" | "never"}
 * steps are the session's shell commands so far with their output (output null when none was recorded); events are
 * the evidence about files and programs up to that point (see FactEvent in virtual-facts.ts).
 * Each output line, in input order: {id, tools, argumentsByTool}, as buildLists returns them.
 * A bad input line stops the run with an error naming its line number and id.
 *
 * With --live (node scripts/jeff-first-menus.ts --live), the facts come from this machine's own disk and PATH, as in a
 * live run (liveFacts), instead of from events; a point must then carry no events. Used to replay a session inside
 * the task's container, where the files are real.
 */

import { createInterface } from "node:readline";
import { detectCheckCommands } from "../packages/coding-agent/src/core/jeff-first/check-commands.ts";
import type { RunApproval } from "../packages/coding-agent/src/core/jeff-first/config.ts";
import { type FileFacts, liveFacts } from "../packages/coding-agent/src/core/jeff-first/facts.ts";
import { buildLists } from "../packages/coding-agent/src/core/jeff-first/lists.ts";
import type { Step } from "../packages/coding-agent/src/core/jeff-first/transcript.ts";
import { type FactEvent, virtualFacts } from "../packages/coding-agent/src/core/jeff-first/virtual-facts.ts";

const RUN_APPROVALS: RunApproval[] = ["all", "seen", "never"];

const flags = process.argv.slice(2);
const unknownFlag = flags.find((flag) => flag !== "--live");
if (unknownFlag !== undefined) throw new Error(`unknown flag ${unknownFlag}; the only flag is --live`);
const live = flags.includes("--live");

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function menusFor(point: Record<string, unknown>): string {
	const { id, cwd, task, steps, events, activeTools, runApproval } = point;
	if (typeof cwd !== "string" || !cwd.startsWith("/")) throw new Error("cwd must be an absolute path");
	if (typeof task !== "string") throw new Error("task must be text");
	if (!Array.isArray(steps)) throw new Error("steps must be an array");
	if (live && events !== undefined) throw new Error("with --live the facts come from the disk, so events must be left out");
	if (!live && !Array.isArray(events)) throw new Error("events must be an array");
	if (!Array.isArray(activeTools) || !activeTools.every((tool) => typeof tool === "string")) {
		throw new Error("activeTools must be an array of tool names");
	}
	if (!RUN_APPROVALS.includes(runApproval as RunApproval)) {
		throw new Error(`runApproval must be all, seen or never, got ${JSON.stringify(runApproval)}`);
	}
	const built: Step[] = steps.map((step: unknown, index) => {
		if (
			!isRecord(step) ||
			typeof step.command !== "string" ||
			(typeof step.output !== "string" && step.output !== null) ||
			typeof step.byScout !== "boolean"
		) {
			throw new Error(`step ${index} must be {command: text, output: text or null, byScout: true or false}`);
		}
		return {
			call: { type: "toolCall", id: `step-${index}`, name: "bash", arguments: { command: step.command } },
			output: step.output,
			// The transcript does not say whether a command failed; the list code never reads this flag.
			isError: false,
			byScout: step.byScout,
		};
	});
	const facts: FileFacts = live ? liveFacts() : virtualFacts(events as FactEvent[]);
	const checks = detectCheckCommands({ cwd, task, steps: built, facts });
	const lists = buildLists({
		cwd,
		task,
		steps: built,
		activeTools: new Set(activeTools),
		checkCommands: checks.commands,
		facts,
		runApproval: runApproval as RunApproval,
	});
	return JSON.stringify({ id, tools: lists.tools, argumentsByTool: lists.argumentsByTool });
}

let lineNumber = 0;
for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
	lineNumber++;
	if (line.trim() === "") continue;
	let point: unknown;
	try {
		point = JSON.parse(line);
	} catch (error) {
		throw new Error(`input line ${lineNumber} is not JSON: ${(error as Error).message}`);
	}
	if (!isRecord(point) || typeof point.id !== "string") {
		throw new Error(`input line ${lineNumber} must be a JSON object with a text id`);
	}
	try {
		process.stdout.write(`${menusFor(point)}\n`);
	} catch (error) {
		throw new Error(`input line ${lineNumber} (id ${point.id}): ${(error as Error).message}`);
	}
}
