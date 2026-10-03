import { createServer, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export interface FakeTeacher {
	url: string;
	requests: Array<Record<string, unknown>>;
	close(): Promise<void>;
}

/**
 * An OpenAI-style chat server whose answer is `pick(options, prompt)`. A pick of "!500" answers with status 500; a
 * pick starting with "!raw:" answers status 200 with the rest of the pick as the whole response body; "!status:N:body"
 * answers status N with that body; "!sleep:MS:rest" waits MS milliseconds, then answers as `rest` would.
 */
export async function startFakeTeacher(
	pick: (options: Array<{ code: string; description: string }>, prompt: string) => string,
): Promise<FakeTeacher> {
	const requests: Array<Record<string, unknown>> = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk: Buffer) => {
			body += chunk.toString();
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { messages: Array<{ content: string }> };
			requests.push({ ...parsed, path: request.url, authorization: request.headers.authorization });
			const prompt = parsed.messages.at(-1)?.content ?? "";
			const options = [...prompt.matchAll(/^([A-Z]): (.*)$/gm)].map((m) => ({ code: m[1], description: m[2] }));
			respond(response, pick(options, prompt));
		});
	});
	const respond = (response: ServerResponse, content: string): void => {
		const sleep = /^!sleep:(\d+):([\s\S]*)$/.exec(content);
		if (sleep) {
			setTimeout(() => respond(response, sleep[2]), Number(sleep[1]));
			return;
		}
		const status = /^!status:(\d+):([\s\S]*)$/.exec(content);
		if (status) {
			response.writeHead(Number(status[1]), { "content-type": "text/html" });
			response.end(status[2]);
			return;
		}
		if (content === "!500") {
			response.writeHead(500, { "content-type": "application/json" });
			response.end('{"error":"overloaded"}');
			return;
		}
		if (content.startsWith("!raw:")) {
			response.writeHead(200, { "content-type": "application/json" });
			response.end(content.slice("!raw:".length));
			return;
		}
		response.writeHead(200, { "content-type": "application/json" });
		response.end(JSON.stringify({ choices: [{ message: { role: "assistant", content } }] }));
	};
	await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
	const { port } = server.address() as AddressInfo;
	return {
		url: `http://127.0.0.1:${port}`,
		requests,
		close: () => new Promise<void>((done) => server.close(() => done())),
	};
}

/** The JSON answer the fake teacher returns for the option whose description starts with `start`. */
export function answerFor(options: Array<{ code: string; description: string }>, start: string): string {
	const option = options.find((o) => o.description.startsWith(start));
	if (!option) throw new Error(`no option starts with "${start}"`);
	return JSON.stringify({ reason: `chose ${start}`, choice: option.code });
}
