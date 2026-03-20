import { NextRequest, NextResponse } from "next/server";
import { runPython } from "@/lib/run-python";
import { tmpdir } from "os";
import { writeFile, readFile, unlink } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { system_prompt, user_message, models } = body as {
    system_prompt?: string;
    user_message: string;
    models: string[];
  };

  if (!user_message || !models?.length) {
    return NextResponse.json(
      { error: "user_message and models are required" },
      { status: 400 },
    );
  }

  const outputFile = path.join(tmpdir(), `dormant-chat-${randomUUID()}.json`);

  const args = [
    "--user",
    user_message,
    "--models",
    ...models,
    "--output",
    outputFile,
  ];

  if (system_prompt) {
    args.push("--system", system_prompt);
  }

  try {
    const { stdout, stderr, code } = await runPython(
      "scripts/api_probes/run_one_off.py",
      args,
      120_000,
    );

    if (code !== 0) {
      return NextResponse.json(
        { error: `Python exited with code ${code}: ${stderr || stdout}` },
        { status: 500 },
      );
    }

    const raw = await readFile(outputFile, "utf-8");
    const results = JSON.parse(raw);

    // Clean up temp file
    await unlink(outputFile).catch(() => {});

    return NextResponse.json({ results });
  } catch (err) {
    return NextResponse.json(
      { error: String(err) },
      { status: 500 },
    );
  }
}
