import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

export interface Sent {
	model: string;
	state: string;
	questions: Record<string, { type: string; instructions: string; criteria: Record<string, string> }>;
}

export type Reply = { status: number; body: string; delayMs?: number };

export interface FakeJeff {
	url: string;
	requests: Sent[];
	close(): Promise<void>;
}

/** A fake jeff-serve: POST /v1/systemone answers with `reply(request)`. */
export async function startFakeJeff(reply: (sent: Sent, index: number) => Reply): Promise<FakeJeff> {
	const requests: Sent[] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk: Buffer) => {
			body += chunk.toString();
		});
		request.on("end", () => {
			const sent = JSON.parse(body) as Sent;
			requests.push(sent);
			if (request.url !== "/v1/systemone") {
				response.writeHead(404);
				response.end("not found");
				return;
			}
			const answer = reply(sent, requests.length - 1);
			setTimeout(() => {
				response.writeHead(answer.status, { "content-type": "application/json" });
				response.end(answer.body);
			}, answer.delayMs ?? 0);
		});
	});
	await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
	const { port } = server.address() as AddressInfo;
	return {
		url: `http://127.0.0.1:${port}`,
		requests,
		close: () =>
			new Promise<void>((done) => {
				server.closeAllConnections();
				server.close(() => done());
			}),
	};
}

/** jeff-serve's answer for one question with these probabilities. */
export function answer(model: string, probabilities: Record<string, number>): Reply {
	const best = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];
	return {
		status: 200,
		body: JSON.stringify({
			model,
			answers: { q: { type: "choice", probabilities, choice: best, confidence: 0.5 } },
			usage: { input_tokens: 1000, output_tokens: 0, orders: 1 },
		}),
	};
}
