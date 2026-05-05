import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  predictText: vi.fn(async () => ({
    label: "Bullying",
    confidence: 0.91,
    probabilities: { "Non-Bullying": 0.09, Bullying: 0.91 },
    flagged_tokens: ["cheap"],
    normalized_text: "tumhara attitude [ABUSIVE] hai",
    model_version: "baseline"
  }))
}));

describe("App", () => {
  it("renders a prediction after submit", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /analyze text/i }));
    await waitFor(() => {
      expect(screen.getByText(/bullying/i)).toBeInTheDocument();
      expect(screen.getByText(/cheap/i)).toBeInTheDocument();
    });
  });
});

