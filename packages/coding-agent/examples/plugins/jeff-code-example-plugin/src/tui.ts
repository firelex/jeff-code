import { AgentController, PresentationUI, SlashCommands } from "@jeffhub/jeff-code/experimental/plugin";
import { defineFacet } from "@jeffhub/jeff-code-chord";
import { ExampleFacetService } from "./contract.ts";

export default defineFacet({
	id: "@jeffhub/jeff-code-example-plugin/tui",
	setup(env) {
		const example = env.use(ExampleFacetService);
		const commands = env.use(SlashCommands);
		const controller = env.use(AgentController);
		const ui = env.use(PresentationUI);
		env.onActivate(() => {
			env.own(
				commands.replace({
					name: "hello",
					description: "Call the bundled Session worker facet",
					argumentHint: "<name>",
					async run(args, context) {
						const message = args.length === 0 ? "from the TUI" : args;
						const reply = await example.greet({ name: message }, context);
						ui.showStatus(
							`We got this: ${reply.message} Worker activations: ${reply.workerActivations}.`,
							context,
						);
						return controller.prompt({ message: `pong: ${message}`, images: null }, context);
					},
				}),
			);
		});
	},
});
