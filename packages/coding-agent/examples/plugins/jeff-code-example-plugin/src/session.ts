import { defineFacet } from "@jeffhub/jeff-code-chord";
import { BACKGROUND_CONTEXT } from "@jeffhub/jeff-code-chord/context";
import { ExampleFacetService } from "./contract.ts";

export default defineFacet({
	id: "@jeffhub/jeff-code-example-plugin/session",
	setup(env) {
		const workerActivations = env.replicatedState({ count: 0 });
		env.provide(ExampleFacetService, {
			workerActivations,
			async greet({ name }) {
				return {
					message: `Hello ${name} from the bundled Session worker facet!!!`,
					workerActivations: workerActivations.value.count,
				};
			},
		});
		env.onActivate(() => {
			workerActivations.change(BACKGROUND_CONTEXT, (draft) => {
				draft.count += 1;
			});
		});
	},
});
