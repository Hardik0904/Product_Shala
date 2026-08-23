import ReviewSlip from "./components/ReviewSlip";
import HowItWorks from "./components/HowItWorks";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app__header">
        <p className="app__eyebrow">sentiment analysis for product reviews</p>
        <h1 className="app__title">ProductShala</h1>
        <p className="app__tagline">
          Write a review. We'll read between the lines — including the "not"s.
        </p>
      </header>

      <main>
        <ReviewSlip />
        <HowItWorks />
      </main>

      <footer className="app__footer">
        <span>ProductShala</span>
        <span>model: bidirectional LSTM · trained on 4.3k labelled reviews</span>
      </footer>
    </div>
  );
}

export default App;
