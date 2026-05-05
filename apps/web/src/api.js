const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function predictText(text) {
  const response = await fetch(`${DEFAULT_API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Prediction failed." }));
    throw new Error(payload.detail || "Prediction failed.");
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${DEFAULT_API_BASE}/health`);
  if (!response.ok) {
    throw new Error("API Offline");
  }
  return response.json();
}

