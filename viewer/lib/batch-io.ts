import { readdir, readFile } from "fs/promises";
import path from "path";
import type { Batch } from "./types";

const BATCHES_DIR = path.join(process.cwd(), "..", "batches");

export async function listBatches(): Promise<Batch[]> {
  let files: string[];
  try {
    files = await readdir(BATCHES_DIR);
  } catch {
    return [];
  }

  const jsonFiles = files.filter((f) => f.endsWith(".json")).sort();
  const batches: Batch[] = [];

  for (const file of jsonFiles) {
    try {
      const raw = await readFile(path.join(BATCHES_DIR, file), "utf-8");
      batches.push(JSON.parse(raw));
    } catch {
      // skip malformed files
    }
  }

  return batches;
}

export async function getBatch(id: string): Promise<Batch | null> {
  const filePath = path.join(BATCHES_DIR, `${id}.json`);
  try {
    const raw = await readFile(filePath, "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
