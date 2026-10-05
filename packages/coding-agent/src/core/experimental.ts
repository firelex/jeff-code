export function areExperimentalFeaturesEnabled(): boolean {
	return process.env.JEFF_EXPERIMENTAL === "1";
}
