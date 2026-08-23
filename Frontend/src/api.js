// Talks to the ProductShala Flask backend (see /Backend in the repo root).
// Configure via .env: VITE_API_URL=http://localhost:5000
// Falls back to localhost so `npm run dev` works out of the box against
// a locally-running backend without any setup.

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export class ApiError extends Error {}

export async function analyzeReview(review) {
  let response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });
  } catch (err) {
    throw new ApiError(
      `Couldn't reach the ProductShala API at ${API_URL}. Is the backend running? (python app.py)`
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.error || `Request failed with status ${response.status}`);
  }

  return response.json(); // { review, prediction, polarity }
}
