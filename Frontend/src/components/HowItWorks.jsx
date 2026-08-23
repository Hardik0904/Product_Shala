import "./HowItWorks.css";

const STEPS = [
  {
    n: "01",
    title: "Clean & mark negation",
    body: "The review is lowercased and stripped down, but words after \u201cnot\u201d, \u201cisn't\u201d, \u201cnever\u201d stay flagged as negated instead of being treated the same as their un-negated selves.",
  },
  {
    n: "02",
    title: "Tokenize & pad",
    body: "Flagged text is converted to a fixed-length sequence of learned word IDs the model can actually read.",
  },
  {
    n: "03",
    title: "Predict & stamp",
    body: "A bidirectional LSTM scores the sequence from 0 to 1. Above 0.55 gets APPROVED, below 0.45 gets REJECTED, the middle sits ON THE FENCE.",
  },
];

export default function HowItWorks() {
  return (
    <section className="how-it-works" aria-labelledby="how-it-works-heading">
      <h2 id="how-it-works-heading" className="how-it-works__heading">
        What happens after you hit stamp
      </h2>
      <ol className="how-it-works__list">
        {STEPS.map((step) => (
          <li key={step.n} className="how-it-works__step">
            <span className="how-it-works__n">{step.n}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
