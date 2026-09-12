import type { NodeCompiler } from "@myriaddreamin/typst-ts-node-compiler";

let compilerPromise: Promise<NodeCompiler> | undefined;

async function getCompiler() {
    compilerPromise ??= import("@myriaddreamin/typst-ts-node-compiler").then(({ NodeCompiler }) =>
        NodeCompiler.create({ workspace: process.cwd() })
    );
    return compilerPromise;
}

export async function renderTypst(mainFilePath: string) {
    const compiler = await getCompiler();
    const compilation = compiler.compile({ mainFilePath, resetRead: import.meta.env.DEV });
    const document = compilation.result;

    if (!document) {
        const error = compilation.takeError() ?? compilation.takeDiagnostics();
        const details = error
            ? compiler
                  .fetchDiagnostics(error)
                  .map(({ message, path }) => `${path}: ${message}`)
                  .join("\n")
            : "Unknown Typst compilation error";
        throw new Error(`Could not compile ${mainFilePath}:\n${details}`);
    }

    const svg = compiler.svg(document);
    compiler.evictCache(import.meta.env.DEV ? 30 : 10);
    return svg;
}
