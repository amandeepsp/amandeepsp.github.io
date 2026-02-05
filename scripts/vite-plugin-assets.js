/**
 * Vite plugin to build Python asset generators.
 *
 * - Runs the asset build script on server start
 * - Watches scripts/assets for changes during dev
 * - Triggers incremental rebuilds when Python files change
 */

import { spawn } from "node:child_process";
import path from "node:path";

const ASSETS_DIR = "scripts/assets";
const BUILD_SCRIPT = "scripts/assets/build.py";

/**
 * Run the Python build script.
 * @param {string} cwd - Working directory
 * @param {boolean} force - Force rebuild all assets
 * @returns {Promise<void>}
 */
function runBuild(cwd, force = false) {
    return new Promise((resolve, reject) => {
        const args = ["run", "python", BUILD_SCRIPT];
        if (force) args.push("--force");

        const proc = spawn("uv", args, {
            cwd,
            stdio: "inherit",
        });

        proc.on("close", (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Asset build failed with code ${code}`));
            }
        });

        proc.on("error", reject);
    });
}

/**
 * @returns {import('vite').Plugin}
 */
export default function assetsPlugin() {
    let root = "";

    return {
        name: "vite-plugin-assets",

        configResolved(config) {
            root = config.root;
        },

        async buildStart() {
            console.log("\n[assets] Building Python assets...");
            try {
                await runBuild(root);
                console.log("[assets] Done.\n");
            } catch (err) {
                console.error("[assets] Build failed:", err.message);
            }
        },

        configureServer(server) {
            // Watch the assets directory for changes
            const assetsPath = path.join(root, ASSETS_DIR);

            server.watcher.add(assetsPath);

            server.watcher.on("change", async (file) => {
                if (file.startsWith(assetsPath) && file.endsWith(".py")) {
                    const relPath = path.relative(root, file);
                    console.log(`\n[assets] ${relPath} changed, rebuilding...`);
                    try {
                        await runBuild(root);
                        console.log("[assets] Done.\n");
                        // Trigger a full reload since assets are in public/
                        server.ws.send({ type: "full-reload" });
                    } catch (err) {
                        console.error("[assets] Build failed:", err.message);
                    }
                }
            });
        },
    };
}
