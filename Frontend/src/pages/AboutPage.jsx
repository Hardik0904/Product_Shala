import "../styles/Aboutpage.css";
import Developer1 from "../assets/Developer1.jpg"; 

function AboutPage() {
  return (
    <main>
      <section className="about-section">
        <h1 className="about-heading">About the Developer</h1>
        <div className="about-container">
          <div className="developer-card">
            <img
              src={Developer1}
              alt="Hardik Shah"
              className="developer-img"
            />
            <div className="developer-info">
              <h2>Hardik Shah</h2>
              <h3>Full Stack & AI Enthusiast</h3>
              <p>
                Hi, I’m Hardik Shah, a Computer Science student with a strong
                interest in software development, artificial intelligence, and
                problem-solving. I am currently working on projects like
                ProductShala, where I focus on building practical applications
                using modern technologies.
                
                My core skills include Python, Data Structures & Algorithms,
                and web development using the MERN stack. I am also exploring
                deep learning and machine learning frameworks like TensorFlow
                and PyTorch.

                I enjoy building real-world projects, preparing for competitive
                exams like GATE, and continuously improving my technical and
                analytical skills.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

export default AboutPage;