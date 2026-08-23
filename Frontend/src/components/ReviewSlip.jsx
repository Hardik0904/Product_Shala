import { useState } from "react";
import { analyzeReview, ApiError } from "../api";
import StampBadge from "./StampBadge";
import "./ReviewSlip.css";

const EXAMPLES = [
  "The battery life on this is not good.",
  "Honestly, this isn't bad at all.",
  "Fast shipping, but the packaging was a disaster.",
];

export default function ReviewSlip() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null); // { prediction, polarity }
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMessage, setErrorMessage] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!review.trim()) return;

    setStatus("loading");
    setResult(null);
    setErrorMessage("");

    try {
      const data = await analyzeReview(review.trim());
      setResult(data);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
    }
  };

  const fillExample = (text) => {
    setReview(text);
    setResult(null);
    setStatus("idle");
    setErrorMessage("");
  };

  return (
    <div className="review-slip">
      <div className="review-slip__perforation" aria-hidden="true" />
      <div className="review-slip__paper">
        <div className="review-slip__letterhead">
          <span>PRODUCTSHALA</span>
          <span>REVIEW SLIP NO. {String(review.length).padStart(4, "0")}</span>
        </div>

        <form onSubmit={handleSubmit} className="review-slip__form">
          <label htmlFor="review-input" className="review-slip__label">
            Describe the product, honestly
          </label>
          <textarea
            id="review-input"
            className="review-slip__textarea"
            placeholder="e.g. Not bad for the price, but the battery drains fast..."
            value={review}
            onChange={(e) => setReview(e.target.value)}
            rows={4}
          />

          <div className="review-slip__examples">
            <span>Try:</span>
            {EXAMPLES.map((ex) => (
              <button
                type="button"
                key={ex}
                className="review-slip__chip"
                onClick={() => fillExample(ex)}
              >
                {ex}
              </button>
            ))}
          </div>

          <button
            type="submit"
            className="review-slip__submit"
            disabled={status === "loading" || !review.trim()}
          >
            {status === "loading" ? "Stamping…" : "Stamp it"}
          </button>
        </form>

        {status === "error" && (
          <p className="review-slip__error" role="alert">
            {errorMessage}
          </p>
        )}

        {result && (
          <div className="review-slip__result">
            <StampBadge prediction={result.prediction} polarity={result.polarity} />
          </div>
        )}
      </div>
      <div className="review-slip__perforation" aria-hidden="true" />
    </div>
  );
}
