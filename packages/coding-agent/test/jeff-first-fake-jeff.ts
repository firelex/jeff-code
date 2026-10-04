import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

export interface Sent {
	model: string;
	state: string;
	questions: Record<string, { type: string; instructions: string; criteria: Record<string, string> }>;
}

/** What the scout sends to POST /v1/fit before each question. */
export interface FitSent {
	state: string;
	question: { type: string; instructions: string; criteria: Record<string, string> };
}

export type Reply = { status: number; body: string; delayMs?: number };

export interface FakeJeff {
	url: string;
	/** The POST /v1/systemone requests. */
	requests: Sent[];
	/** The POST /v1/fit requests. */
	fits: FitSent[];
	close(): Promise<void>;
}

/** POST /v1/fit's answer when the question fits as it is. */
export const FITS: Reply = { status: 200, body: '{"cut":null}' };

/** A fake jeff-serve: POST /v1/systemone answers with `reply(request)`, POST /v1/fit with `fitReply(request)`. */
export async function startFakeJeff(
	reply: (sent: Sent, index: number) => Reply,
	fitReply: (sent: FitSent, index: number) => Reply = () => FITS,
): Promise<FakeJeff> {
	const requests: Sent[] = [];
	const fits: FitSent[] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk: Buffer) => {
			body += chunk.toString();
		});
		request.on("end", () => {
			let answer: Reply;
			if (request.url === "/v1/fit") {
				const sent = JSON.parse(body) as FitSent;
				fits.push(sent);
				answer = fitReply(sent, fits.length - 1);
			} else if (request.url === "/v1/systemone") {
				const sent = JSON.parse(body) as Sent;
				requests.push(sent);
				answer = reply(sent, requests.length - 1);
			} else {
				response.writeHead(404);
				response.end("not found");
				return;
			}
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
		fits,
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
