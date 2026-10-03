import type { ArgumentOption, Lists, ToolKind } from "./lists.ts";
import type { ShownOption } from "./teacher-prompt.ts";

export const PAGE_SIZE = 10;
/** Three pages per question: in Gate 0, 98% of the teacher's argument picks were among the first ten options. */
export const MAX_PAGES = 3;
export const SHOW_MORE: ShownOption = { id: "show_more", description: "Show more options" };
export const NONE_OF_THESE: ShownOption = {
	id: "none_of_these",
	description: "None of these: hand over to the coding model",
};

function pageOf<T>(items: T[], page: number): T[] {
	return items.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
}

function moreAfter(count: number, page: number): boolean {
	return page < MAX_PAGES && count > page * PAGE_SIZE;
}

/** The tool question's options on this page: each tool that has options here, with them written out, then hand over. */
export function toolPage(lists: Lists, page: number): ShownOption[] {
	const shown: ShownOption[] = [];
	let more = false;
	for (const tool of lists.tools) {
		if (tool.id === "hand_over") continue;
		const options = lists.argumentsByTool[tool.id as ToolKind];
		if (!options) throw new Error(`the tool ${tool.id} is offered but has no argument list`);
		if (moreAfter(options.length, page)) more = true;
		const here = pageOf(options, page);
		if (here.length === 0) continue;
		const preview = here.map((option) => option.description).join("; ");
		const count = options.length === 1 ? "1 option" : `${options.length} options`;
		shown.push({ id: tool.id, description: `${tool.description}: ${preview} (${count})` });
	}
	const handOver = lists.tools.find((tool) => tool.id === "hand_over");
	if (!handOver) throw new Error("the tool list has no hand-over option");
	shown.push(handOver);
	if (more) shown.push(SHOW_MORE);
	return shown;
}

/** One tool's argument options on this page, then None of these, then Show more options while more remain. */
export function argumentPage(options: ArgumentOption[], page: number): Array<ArgumentOption | ShownOption> {
	const shown: Array<ArgumentOption | ShownOption> = [...pageOf(options, page), NONE_OF_THESE];
	if (moreAfter(options.length, page)) shown.push(SHOW_MORE);
	return shown;
}
