import "./StampBadge.css";

const VERDICT_COPY = {
  Positive: { label: "APPROVED", sub: "positive sentiment" },
  Negative: { label: "REJECTED", sub: "negative sentiment" },
  Neutral: { label: "ON THE FENCE", sub: "neutral sentiment" },
};

// A rotated, grain-textured "ink stamp" that lands on the review slip once
// a verdict comes back. The grain comes from an SVG feTurbulence filter
// applied to the badge, not an image asset, so it stays crisp at any size.
export default function StampBadge({ prediction, polarity }) {
  if (!prediction) return null;

  const copy = VERDICT_COPY[prediction] || { label: prediction.toUpperCase(), sub: "" };
  const tone = prediction.toLowerCase(); // "positive" | "negative" | "neutral"

  return (
    <div className={`stamp-badge stamp-badge--${tone}`} role="status" aria-live="polite">
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
        <filter id="stamp-grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" result="noise" />
          <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.55 0" />
          <feComposite operator="in" in2="SourceGraphic" result="grain" />
          <feComposite operator="over" in="grain" in2="SourceGraphic" />
        </filter>
      </svg>
      <div className="stamp-badge__ring">
        <span className="stamp-badge__label">{copy.label}</span>
        <span className="stamp-badge__sub">{copy.sub}</span>
        <span className="stamp-badge__score">
          polarity {polarity !== undefined ? polarity.toFixed(3) : "—"}
        </span>
      </div>
    </div>
  );
}
