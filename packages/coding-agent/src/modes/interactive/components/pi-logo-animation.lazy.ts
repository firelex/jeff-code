import { type TUI, TuiAltScreen } from "@jeffhub/jeff-code-tui";

/**
 * Plays the logo easter egg (see pi-logo-animation.ts), which loads on the first click. Only fullscreen mode
 * can show it, because it dissolves the rendered screen. The screen is captured before loading.
 */
export function playPiLogoAnimation(tui: TUI, logoColumn: number, logoRow: number): void {
	if (!(tui instanceof TuiAltScreen) || tui.hasOverlay()) return;
	const screen = tui.getScreenLines();
	import("./pi-logo-animation.ts").then(
		(module) => module.playPiLogoAnimation(tui, { screen, logoColumn, logoRow }),
		() => {},
	);
}
