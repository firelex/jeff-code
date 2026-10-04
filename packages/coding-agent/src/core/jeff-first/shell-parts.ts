import { isAbsolute, resolve } from "node:path";

/**
 * A small reader of the bash commands the coding model types, enough to tell which files a command writes and which
 * script or program it runs, and in which folder. It does not run or expand anything.
 *
 * A command is split into parts (simple commands) at `&&`, `||`, `;`, `&` and new lines, outside quotes and outside
 * `$(...)`; a part's pipeline stages after the first are its filters. Here-document bodies are taken out of the text
 * first and kept with the part that opened them. Control-flow keywords (`for`, `while`, `if`, ...), `{ }` groups and
 * `( )` subshells are dropped from the parts, and every part of such a command is marked `inControlFlow` (it may run
 * any number of times, or not at all).
 */
export interface ShellPart {
	/** The first pipeline stage, as typed (a here-document body taken out). */
	head: string;
	/** The later pipeline stages, as typed. */
	filters: string[];
	/** The body of the here-document this part opened, if any. */
	heredoc: string | undefined;
	/** True when the part runs in the background (`&` after it). */
	background: boolean;
	inControlFlow: boolean;
	/** The folder the part runs in: the starting folder, moved by each `cd` before it; undefined once unknown. */
	folder: string | undefined;
}

export interface ShellWords {
	/** The words, quotes removed, redirections left out. */
	words: string[];
	/** Files written by `>`, `>>` or `&>` (any stream), as typed; /dev/null and other /dev files are left out. */
	writes: string[];
	/** Whether a word holds an expansion the shell would make (`$`, a backtick, a `*`, `?` or `[` outside quotes), so
	 * the words are not exactly what the program receives. */
	expands: boolean;
}

const HEREDOC_OPENER = /<<-?\s*(['"]?)([\w.-]+)\1/g;
const CONTROL_KEYWORDS = /^(for|while|until|if|elif|else|then|do|done|fi|case|esac|select|function)\b/;

/** The command with every here-document body taken out, and the bodies in the order their openers appear. */
function takeOutHeredocs(command: string): { text: string; bodies: string[] } {
	const lines = command.split("\n");
	const out: string[] = [];
	const bodies: string[] = [];
	let index = 0;
	while (index < lines.length) {
		const line = lines[index];
		index++;
		const openers = [...line.matchAll(HEREDOC_OPENER)].filter((match) => line[match.index - 1] !== "<");
		let marked = line;
		for (const opener of openers) {
			marked = marked.replace(opener[0], ` __HEREDOC${bodies.length}__ `);
			const body: string[] = [];
			const strip = opener[0].startsWith("<<-");
			while (index < lines.length && (strip ? lines[index].replace(/^\t+/, "") : lines[index]) !== opener[2]) {
				body.push(lines[index]);
				index++;
			}
			index++;
			bodies.push(body.join("\n"));
		}
		out.push(marked);
	}
	return { text: out.join("\n"), bodies };
}

interface RawPipeline {
	stages: string[];
	background: boolean;
}

/** The pipelines of the text (here-documents taken out), split outside quotes and `$(...)`. */
function pipelines(text: string): RawPipeline[] {
	const found: RawPipeline[] = [];
	let stages: string[] = [];
	let segment = "";
	let quote: string | undefined;
	let depth = 0;
	const endSegment = () => {
		if (segment.trim() !== "") stages.push(segment.trim());
		segment = "";
	};
	const endPipeline = (background: boolean) => {
		endSegment();
		if (stages.length > 0) found.push({ stages, background });
		stages = [];
	};
	for (let index = 0; index < text.length; index++) {
		const char = text[index];
		const next = text[index + 1];
		if (quote !== undefined) {
			if (char === "\\" && quote === '"' && next !== undefined) {
				segment += char + next;
				index++;
				continue;
			}
			if (char === quote) quote = undefined;
			segment += char;
			continue;
		}
		if (char === "'" || char === '"') {
			quote = char;
			segment += char;
			continue;
		}
		if (char === "\\" && next !== undefined) {
			segment += next === "\n" ? " " : char + next;
			index++;
			continue;
		}
		if (char === "$" && next === "(") {
			depth++;
			segment += "$(";
			index++;
			continue;
		}
		if (depth > 0) {
			if (char === "(") depth++;
			if (char === ")") depth--;
			segment += char;
			continue;
		}
		if (char === "#" && (segment === "" || /\s$/.test(segment))) {
			const end = text.indexOf("\n", index);
			index = end < 0 ? text.length : end - 1;
			continue;
		}
		if ((char === "&" && next === "&") || (char === "|" && next === "|")) {
			endPipeline(false);
			index++;
			continue;
		}
		if (char === ";" || char === "\n") {
			endPipeline(false);
			continue;
		}
		if (char === "&" && next !== ">" && !segment.endsWith(">")) {
			endPipeline(true);
			continue;
		}
		if (char === "|") {
			endSegment();
			continue;
		}
		segment += char;
	}
	endPipeline(false);
	return found;
}

/** A stage without control-flow keywords and group or subshell brackets around it, and whether any were dropped;
 * undefined when nothing runs in it (a lone keyword or bracket). The condition of `if`/`while` runs, so it stays. */
function withoutControlFlow(stage: string): { text: string | undefined; changed: boolean } {
	let text = stage;
	let changed = false;
	const count = (value: string, char: string) => value.split(char).length - 1;
	for (;;) {
		const before = text;
		text = text.replace(/^(?:then|do|else)\b\s*|^\{\s+/, "").replace(/\s*;?\s*\}$/, "");
		while (text.startsWith("(") && count(text, "(") > count(text, ")")) text = text.slice(1).trimStart();
		while (text.endsWith(")") && count(text, ")") > count(text, "(")) text = text.slice(0, -1).trimEnd();
		const keyword = CONTROL_KEYWORDS.exec(text);
		if (keyword !== null) {
			const condition = /^(?:if|elif|while|until)\s+!?\s*(.*)$/s.exec(text);
			text = condition !== null && !condition[1].startsWith("[") ? condition[1] : "";
		}
		if (text !== before) changed = true;
		if (text === before || text === "") break;
	}
	return { text: text.trim() === "" ? undefined : text.trim(), changed };
}

/** The words of one stage (quotes removed, redirections left out), the files it redirects output into, and whether
 * the shell would expand any word. Undefined when a quote is not closed (bash would wait for more input). */
export function shellWords(stage: string): ShellWords | undefined {
	const tokens: Array<{ text: string; bare: string }> = [];
	let text = "";
	let bare = "";
	let inToken = false;
	let quote: string | undefined;
	let expands = false;
	for (let index = 0; index < stage.length; index++) {
		const char = stage[index];
		if (quote === "'") {
			if (char === "'") quote = undefined;
			else text += char;
			continue;
		}
		if (quote === '"') {
			if (char === "\\" && index + 1 < stage.length && '"\\$`'.includes(stage[index + 1])) {
				text += stage[index + 1];
				index++;
			} else if (char === '"') quote = undefined;
			else {
				if (char === "$" || char === "`") expands = true;
				text += char;
			}
			continue;
		}
		if (/\s/.test(char)) {
			if (inToken) tokens.push({ text, bare });
			text = "";
			bare = "";
			inToken = false;
			continue;
		}
		inToken = true;
		if (char === "'" || char === '"') {
			quote = char;
			continue;
		}
		if (char === "\\" && index + 1 < stage.length) {
			text += stage[index + 1];
			index++;
			continue;
		}
		if ("$`*?[".includes(char)) expands = true;
		text += char;
		bare += char;
	}
	if (quote !== undefined) return undefined;
	if (inToken) tokens.push({ text, bare });
	const words: string[] = [];
	const writes: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];
		// A redirection is written without quotes: `2>&1`, `>out`, `> out`, `2>/dev/null`, `&>log`, `<in`.
		const redirect = /^(\d*>>?|&>>?|\d*<|\d*>&)(.*)$/.exec(token.bare);
		if (redirect === null || !token.text.startsWith(redirect[1])) {
			words.push(token.text);
			continue;
		}
		let targetText = token.text.slice(redirect[1].length);
		if (targetText === "" && index + 1 < tokens.length) {
			index++;
			targetText = tokens[index].text;
		}
		const writesFile = !redirect[1].includes("<") && !redirect[1].endsWith(">&");
		if (writesFile && targetText !== "" && !targetText.startsWith("/dev/") && !/^&?\d$/.test(targetText)) {
			writes.push(targetText);
		}
	}
	return { words, writes, expands };
}

/** The wrappers a part may start with before the program it runs: `timeout N`, `time`, `nohup`, `nice`, `sudo`,
 * `env A=1`, `stdbuf X`, and variable assignments. The index of the program's word, or undefined when there is none. */
export function programIndex(words: string[]): number | undefined {
	let index = 0;
	while (index < words.length) {
		const word = words[index];
		if (/^\w+=/.test(word) || ["time", "nohup", "nice", "sudo"].includes(word)) {
			index++;
		} else if (word === "timeout" || word === "stdbuf") {
			index += 2;
		} else if (word === "env") {
			index++;
			while (index < words.length && /^\w+=/.test(words[index])) index++;
		} else {
			return index;
		}
	}
	return undefined;
}

/** The parts of a command, each with the folder it runs in when the command starts in `cwd`. */
export function shellParts(command: string, cwd: string): ShellPart[] {
	const { text, bodies } = takeOutHeredocs(command);
	const raw = pipelines(text);
	const kept: Array<{ stages: string[]; background: boolean }> = [];
	let controlFlow = false;
	for (const pipeline of raw) {
		const stages: string[] = [];
		for (const stage of pipeline.stages) {
			const stripped = withoutControlFlow(stage);
			controlFlow ||= stripped.changed;
			if (stripped.text !== undefined) stages.push(stripped.text);
		}
		if (stages.length > 0) kept.push({ stages, background: pipeline.background });
	}
	const parts: ShellPart[] = [];
	let folder: string | undefined = cwd;
	for (const { stages, background } of kept) {
		const marker = /__HEREDOC(\d+)__/.exec(stages[0]);
		const head = stages[0];
		parts.push({
			head,
			filters: stages.slice(1),
			heredoc: marker === null ? undefined : bodies[Number(marker[1])],
			background,
			inControlFlow: controlFlow,
			folder,
		});
		const words = shellWords(head)?.words;
		if (words?.[0] === "cd" || words?.[0] === "pushd" || words?.[0] === "popd") {
			const target = words[1];
			folder =
				words[0] === "popd" ||
				target === undefined ||
				target === "-" ||
				target.startsWith("~") ||
				/[$`*?]/.test(target)
					? undefined
					: folder === undefined && !isAbsolute(target)
						? undefined
						: resolve(folder ?? "/", target);
		}
	}
	return parts;
}

/** The part's head without the here-document marker, as it would be typed again (the body is not part of it). */
export function headText(part: ShellPart): string {
	return part.head.replace(/\s*__HEREDOC\d+__\s*/g, " ").trim();
}
