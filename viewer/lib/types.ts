export interface Batch {
  id: string;
  title: string;
  description?: string;
  created_at: string;
  models: string[];
  status: "pending" | "running" | "completed" | "error";
  prompts: Prompt[];
}

export interface Prompt {
  id: string;
  system_prompt?: string;
  user_message: string;
  tags?: string[];
  category?: string;
  outputs?: Record<string, { content: string }>;
}
