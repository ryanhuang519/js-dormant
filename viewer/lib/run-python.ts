import { spawn } from "child_process";
import path from "path";

const REPO_ROOT = path.join(process.cwd(), "..");

export function runPython(
  scriptPath: string,
  args: string[],
  timeoutMs = 120_000,
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  return new Promise((resolve, reject) => {
    const proc = spawn("uv", ["run", "python", scriptPath, ...args], {
      cwd: REPO_ROOT,
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (d) => (stdout += d));
    proc.stderr.on("data", (d) => (stderr += d));

    const timer = setTimeout(() => {
      proc.kill("SIGTERM");
      reject(new Error(`Timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    proc.on("close", (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, code });
    });

    proc.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}
