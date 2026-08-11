const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface PredictResponse {
  predicted_class: string;
  confidence: number;
  suggested_questions: string[];
}

export interface ChatResponse {
  answer: string;
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return body.detail || `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

export async function createSession(): Promise<string> {
  const res = await fetch(`${API_BASE}/session`, { method: "POST" });
  if (!res.ok) throw new Error(await parseError(res));
  const data = await res.json();
  return data.session_id as string;
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function sendChatMessage(
  message: string,
  predictedLabel: string,
  sessionId: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      predicted_label: predictedLabel,
      session_id: sessionId,
    }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
