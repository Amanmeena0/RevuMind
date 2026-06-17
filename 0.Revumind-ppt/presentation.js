const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const path = require("path");
const sharp = require("sharp");
const { FaBrain, FaChartBar, FaCogs, FaDatabase, FaPython, FaCheckCircle, FaRocket, FaLayerGroup, FaSearch } = require("react-icons/fa");

// ─── Icon helper ───────────────────────────────────────────────────────────────
async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// ─── Color palette ─────────────────────────────────────────────────────────────
const DARK   = "0F1C2E";   // deep navy – title slide & conclusion
const MID    = "1B3A5C";   // mid navy – header circles, cards
const ACCENT = "00B4D8";   // electric cyan
const LIGHT  = "EAF6FB";   // very light blue – content bg
const WHITE  = "FFFFFF";
const MUTED  = "64748B";
const TEXT   = "1E293B";
const GREEN  = "00B894";

// Shadow factory (fresh object every call — pptxgenjs mutates shadow objects in place)
const mkShadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 45, opacity: 0.13 });

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title  = "RevMind – Customer Sentiment Analysis";
  pres.author = "ananaymeena3";

  // Pre-render icons (white, sized for the header-circle motif used on every content slide)
  const iconBrain  = await iconToBase64Png(FaBrain,      WHITE, 300);
  const iconChart  = await iconToBase64Png(FaChartBar,   WHITE, 300);
  const iconCogs   = await iconToBase64Png(FaCogs,       WHITE, 300);
  const iconDB     = await iconToBase64Png(FaDatabase,   WHITE, 300);
  const iconPy     = await iconToBase64Png(FaPython,     WHITE, 300);
  const iconRocket = await iconToBase64Png(FaRocket,     WHITE, 300);
  const iconLayers = await iconToBase64Png(FaLayerGroup, WHITE, 300);
  const iconSearch = await iconToBase64Png(FaSearch,     WHITE, 300);
  const iconCheck  = await iconToBase64Png(FaCheckCircle, GREEN, 300); // used inline on dark bg, not in a circle

  // Shared header: icon-in-circle + title. Repeats on every content slide as the visual motif.
  function addHeader(slide, iconData, titleText, { circleColor = MID, titleColor = DARK, fontSize = 25 } = {}) {
    slide.addShape(pres.shapes.OVAL, { x: 0.5, y: 0.42, w: 0.62, h: 0.62, fill: { color: circleColor }, shadow: mkShadow() });
    slide.addImage({ data: iconData, x: 0.66, y: 0.58, w: 0.3, h: 0.3 });
    slide.addText(titleText, {
      x: 1.32, y: 0.4, w: 8.2, h: 0.66,
      fontSize, fontFace: "Cambria", bold: true, color: titleColor, valign: "middle", margin: 0
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 1 – Title
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: DARK };

    // Decorative circle (top right) — fine as a motif; only edge stripes/bars are avoided
    s.addShape(pres.shapes.OVAL, { x: 7.8, y: -0.6, w: 3.2, h: 3.2, fill: { color: MID, transparency: 30 }, line: { color: MID } });

    // Brain icon — placed to the left of and vertically aligned with the title so it
    // no longer collides with the "RevMind" text (original had both starting at x≈0.5)
    s.addImage({ data: iconBrain, x: 0.5, y: 0.78, w: 0.72, h: 0.72 });

    s.addText("RevMind", {
      x: 1.38, y: 0.6, w: 7.2, h: 1.0,
      fontSize: 50, fontFace: "Cambria", bold: true, color: WHITE, valign: "middle", margin: 0
    });

    s.addText("Customer Sentiment Analysis using Product Reviews", {
      x: 0.5, y: 1.75, w: 8.5, h: 0.5,
      fontSize: 19, fontFace: "Calibri", color: ACCENT, margin: 0
    });

    // Sub-details (moved up to fill the space the old accent-line divider used to take)
    s.addText([
      { text: "Electronics & Communication Engineering  |  Year 3", options: { breakLine: true } },
      { text: "Subject: Machine Learning  |  Language: Python + Scikit-learn", options: {} }
    ], {
      x: 0.5, y: 2.45, w: 9, h: 0.9,
      fontSize: 14, fontFace: "Calibri", color: "A8C8E8"
    });

    // GitHub tag
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.55, w: 4.0, h: 0.45, fill: { color: MID }, rectRadius: 0.08 });
    s.addText("github.com/ananaymeena3/RevMind", {
      x: 0.5, y: 3.55, w: 4.0, h: 0.45,
      fontSize: 11, fontFace: "Calibri", color: ACCENT, align: "center", valign: "middle", margin: 0
    });

    s.addNotes("Introduce project: RevMind is a semester ML project for ECE Year 3 that classifies product reviews into Positive, Negative, and Neutral sentiments.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 2 – Problem Statement / Project Overview
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconSearch, "Project Overview");

    // Left column – What it does
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("What RevMind Does", {
      x: 0.7, y: 1.42, w: 3.9, h: 0.45, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0
    });
    s.addText([
      { text: "Analyzes customer product reviews and classifies them as:", options: { breakLine: true, color: TEXT, fontSize: 13 } },
      { text: " ", options: { breakLine: true, fontSize: 8 } },
      { text: "😊  Positive", options: { breakLine: true, bold: true, color: "00A86B", fontSize: 16 } },
      { text: "😞  Negative", options: { breakLine: true, bold: true, color: "E53935", fontSize: 16 } },
      { text: "😐  Neutral",  options: { bold: true, color: "FB8C00", fontSize: 16 } },
    ], { x: 0.7, y: 2.0, w: 3.9, h: 2.9, fontFace: "Calibri" });

    // Right column – Context
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Project Context", {
      x: 5.4, y: 1.42, w: 3.9, h: 0.45, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0
    });
    s.addText([
      { text: "Academic Semester Project", options: { bold: true, breakLine: true } },
      { text: "ECE Year 3 | 6th Semester", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 8 } },
      { text: "Approach: Classical Machine Learning", options: { bold: true, breakLine: true } },
      { text: "Feature extraction via TF-IDF Vectorization", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 8 } },
      { text: "Dual interface:", options: { bold: true, breakLine: true } },
      { text: "• CLI app  (app.py)", options: { breakLine: true } },
      { text: "• Streamlit web app  (streamlit_app.py)", options: {} },
    ], { x: 5.4, y: 2.0, w: 3.9, h: 2.9, fontSize: 13, fontFace: "Calibri", color: TEXT });

    s.addNotes("Explain the core goal: classifying reviews into 3 sentiment classes using classical ML with TF-IDF features. Two deployment options: CLI and Streamlit web app.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 3 – Repository Structure
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconLayers, "Repository Structure");

    // File tree box
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 4.3, h: 3.85, fill: { color: DARK }, shadow: mkShadow(), rectRadius: 0.1 });
    s.addText([
      { text: "RevMind/", options: { bold: true, color: ACCENT, breakLine: true } },
      { text: "├── train_model.py", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── app.py", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── streamlit_app.py", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── requirements.txt", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── utils/", options: { breakLine: true, color: ACCENT } },
      { text: "│   └── preprocess.py", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── models/", options: { breakLine: true, color: ACCENT } },
      { text: "│   ├── best_model.pkl", options: { breakLine: true, color: "A8D5E2" } },
      { text: "│   └── tfidf_vectorizer.pkl", options: { breakLine: true, color: "A8D5E2" } },
      { text: "└── plots/", options: { breakLine: true, color: ACCENT } },
      { text: "    ├── confusion_matrix.png", options: { breakLine: true, color: "A8D5E2" } },
      { text: "    ├── model_comparison.png", options: { breakLine: true, color: "A8D5E2" } },
      { text: "    └── sentiment_distribution.png", options: { color: "A8D5E2" } },
    ], { x: 0.7, y: 1.4, w: 3.9, h: 3.55, fontSize: 11, fontFace: "Courier New", color: WHITE });

    // Right: description cards
    const fileDesc = [
      { file: "train_model.py", desc: "Trains models, saves best one, generates evaluation plots" },
      { file: "app.py", desc: "Command-line interactive sentiment predictor" },
      { file: "streamlit_app.py", desc: "Streamlit web interface for predictions" },
      { file: "utils/preprocess.py", desc: "Text cleaning and dataset loading utilities" },
    ];
    fileDesc.forEach((f, i) => {
      const y = 1.25 + i * 0.95;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y, w: 4.3, h: 0.82, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.1 });
      s.addText(f.file, { x: 5.4, y: y + 0.08, w: 3.9, h: 0.3, fontSize: 12, fontFace: "Courier New", bold: true, color: MID, margin: 0 });
      s.addText(f.desc, { x: 5.4, y: y + 0.42, w: 3.9, h: 0.35, fontSize: 11, fontFace: "Calibri", color: MUTED, margin: 0 });
    });

    s.addNotes("Walk through the repo layout. Models and plots folders are auto-created at runtime after training. The utils folder holds shared preprocessing code.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 4 – Dataset
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconDB, "Dataset");

    const cards = [
      {
        title: "Default – Synthetic Dataset",
        tag: "No download needed",
        tagColor: "00A86B",
        points: [
          "Built-in product reviews included in utils/preprocess.py",
          "Covers Positive, Negative & Neutral examples",
          "Ready to use out-of-the-box for demonstration",
        ]
      },
      {
        title: "Recommended – Amazon Reviews",
        tag: "For submission",
        tagColor: "00B4D8",
        points: [
          "Real-world Amazon product review dataset",
          "Source: Kunal-Kumar-Das191049/Sentimental-Analysis-of-Amazon-Reviews",
          "Stored in data/ folder after download",
        ]
      }
    ];

    cards.forEach((c, i) => {
      const x = 0.5 + i * 4.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addText(c.title, { x: x + 0.2, y: 1.42, w: 3.9, h: 0.5, fontSize: 15, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 2.0, w: 1.7, h: 0.32, fill: { color: c.tagColor }, rectRadius: 0.06 });
      s.addText(c.tag, { x: x + 0.2, y: 2.0, w: 1.7, h: 0.32, fontSize: 10, fontFace: "Calibri", bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
      c.points.forEach((p, j) => {
        s.addText([{ text: p, options: { bullet: true } }], {
          x: x + 0.2, y: 2.55 + j * 0.78, w: 3.9, h: 0.7,
          fontSize: 13, fontFace: "Calibri", color: TEXT
        });
      });
    });

    s.addNotes("Two dataset options: a built-in synthetic set (great for demos) and Amazon product reviews (recommended for graded submission). Data goes in the data/ folder.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 5 – Text Preprocessing & TF-IDF
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconCogs, "Text Preprocessing & Feature Extraction", { fontSize: 22 });

    const steps = [
      { n: "1", label: "Load Dataset", detail: "utils/preprocess.py\nload_dataset()" },
      { n: "2", label: "Clean Text", detail: "clean_text()\nRemove noise & punctuation" },
      { n: "3", label: "Train/Test Split", detail: "80% train / 20% test\nstratified, random_state=42" },
      { n: "4", label: "TF-IDF Vectorizer", detail: "max_features=5000\nngram_range=(1,2)" },
    ];

    steps.forEach((st, i) => {
      const x = 0.4 + i * 2.35;
      s.addShape(pres.shapes.OVAL, { x: x + 0.6, y: 1.25, w: 1.0, h: 1.0, fill: { color: MID }, shadow: mkShadow() });
      s.addText(st.n, { x: x + 0.6, y: 1.25, w: 1.0, h: 1.0, fontSize: 22, fontFace: "Cambria", bold: true, color: ACCENT, align: "center", valign: "middle", margin: 0 });
      if (i < 3) {
        s.addShape(pres.shapes.RECTANGLE, { x: x + 1.63, y: 1.71, w: 0.7, h: 0.08, fill: { color: ACCENT } });
      }
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 2.45, w: 2.0, h: 1.3, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.1 });
      s.addText(st.label, { x: x + 0.25, y: 2.53, w: 1.9, h: 0.35, fontSize: 12, fontFace: "Cambria", bold: true, color: MID, align: "center", margin: 0 });
      s.addText(st.detail, { x: x + 0.25, y: 2.9, w: 1.9, h: 0.78, fontSize: 10, fontFace: "Calibri", color: MUTED, align: "center" });
    });

    // Key TF-IDF note
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.95, w: 9.0, h: 1.15, fill: { color: MID }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Key TF-IDF Design Choices", {
      x: 0.75, y: 4.05, w: 8.5, h: 0.35, fontSize: 14, fontFace: "Cambria", bold: true, color: ACCENT, margin: 0
    });
    s.addText([
      { text: "Negation words preserved  ", options: { bold: true, color: WHITE } },
      { text: "('not', 'no', 'never' kept in the vocabulary to retain sentiment signal)   ", options: { color: "A8C8E8" } },
      { text: " | ", options: { color: MUTED } },
      { text: "Fit only on train set  ", options: { bold: true, color: WHITE } },
      { text: "(critical to prevent data leakage)", options: { color: "A8C8E8" } }
    ], { x: 0.75, y: 4.45, w: 8.5, h: 0.55, fontSize: 12, fontFace: "Calibri" });

    s.addNotes("Highlight that negation words are deliberately kept in the vocabulary, and the TF-IDF vectorizer is fit ONLY on training data to prevent leakage.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 6 – ML Models
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconBrain, "Machine Learning Models");

    const models = [
      {
        name: "Logistic Regression",
        badge: "Linear Classifier",
        color: "1565C0",
        points: [
          "class_weight='balanced' for imbalanced data",
          "max_iter=200, random_state=42",
          "Strong baseline for text classification",
          "Outputs probability scores per class"
        ]
      },
      {
        name: "SVM (LinearSVC)",
        badge: "Best for Text",
        color: "6A1B9A",
        points: [
          "class_weight='balanced'",
          "SVM variant optimized for high-dim sparse data",
          "Often best-performing on NLP tasks",
          "Automatically saved as best_model.pkl if winner"
        ]
      }
    ];

    // Cards sized so the winner banner below never collides with them (was overlapping by ~0.25" originally)
    models.forEach((m, i) => {
      const x = 0.5 + i * 4.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 4.3, h: 3.35, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 1.42, w: 1.5, h: 0.32, fill: { color: m.color }, rectRadius: 0.07 });
      s.addText(m.badge, { x: x + 0.2, y: 1.42, w: 1.5, h: 0.32, fontSize: 10, fontFace: "Calibri", bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
      s.addText(m.name, { x: x + 0.2, y: 1.84, w: 3.9, h: 0.42, fontSize: 17, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
      m.points.forEach((p, j) => {
        s.addText([{ text: p, options: { bullet: true } }], {
          x: x + 0.2, y: 2.35 + j * 0.55, w: 3.9, h: 0.5,
          fontSize: 12.5, fontFace: "Calibri", color: TEXT
        });
      });
    });

    // Winner-logic note — now starts at 4.78, well clear of the cards which end at 4.60
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.78, w: 9.0, h: 0.45, fill: { color: "EEF2FF" }, shadow: mkShadow(), rectRadius: 0.08 });
    s.addText("★  The best-performing model is automatically saved to models/best_model.pkl after training", {
      x: 0.7, y: 4.78, w: 8.6, h: 0.45, fontSize: 12, fontFace: "Calibri", color: MID, bold: true, valign: "middle", margin: 0
    });

    s.addNotes("RevMind trains both models in one run and auto-selects the winner. Class balancing is applied to handle uneven sentiment distributions.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 7 – Evaluation Metrics
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconChart, "Evaluation Metrics");

    const metrics = [
      { name: "Accuracy", formula: "Correct / Total", desc: "Overall fraction of correct predictions", color: "0077B6" },
      { name: "Precision", formula: "TP / (TP + FP)", desc: "Of predicted positives, how many are correct", color: "00897B" },
      { name: "Recall", formula: "TP / (TP + FN)", desc: "Of actual positives, how many we caught", color: "7B1FA2" },
      { name: "F1-Score", formula: "2 × (P × R) / (P + R)", desc: "Harmonic mean of Precision & Recall", color: "E65100" },
    ];

    // 2×2 grid sized so the confusion-matrix note below never collides with row 2
    // (row 2 previously ended at 5.25 while the note started at 5.05 — a 0.2" overlap)
    metrics.forEach((m, i) => {
      const row = Math.floor(i / 2);
      const col = i % 2;
      const x = 0.5 + col * 4.7;
      const y = 1.25 + row * 1.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.3, h: 1.55, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: y + 0.14, w: 1.3, h: 0.3, fill: { color: m.color }, rectRadius: 0.06 });
      s.addText(m.name, { x: x + 0.2, y: y + 0.14, w: 1.3, h: 0.3, fontSize: 11, fontFace: "Calibri", bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
      s.addText(m.formula, { x: x + 0.2, y: y + 0.52, w: 4.0, h: 0.38, fontSize: 15, fontFace: "Courier New", bold: true, color: m.color, margin: 0 });
      s.addText(m.desc, { x: x + 0.2, y: y + 0.92, w: 3.9, h: 0.55, fontSize: 11.5, fontFace: "Calibri", color: TEXT });
    });

    // Confusion Matrix mention — starts at 4.80, rows end at 4.65 (0.15" clear gap)
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.8, w: 9.0, h: 0.42, fill: { color: MID }, rectRadius: 0.08 });
    s.addText("+ Confusion Matrix  ·  Visual breakdown of TP / FP / TN / FN for all 3 sentiment classes", {
      x: 0.7, y: 4.8, w: 8.6, h: 0.42, fontSize: 12, fontFace: "Calibri", color: WHITE, valign: "middle", margin: 0
    });

    s.addNotes("These four metrics plus the confusion matrix give a full picture of model performance. F1 is most relevant when class distribution is imbalanced.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 8 – Results & Visualizations (placeholder for stats)
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconChart, "Results & Visualizations");

    const plots = [
      { title: "Model Accuracy Comparison", file: "plots/model_comparison.png" },
      { title: "Sentiment Distribution", file: "plots/sentiment_distribution.png" },
      { title: "Confusion Matrix", file: "plots/confusion_matrix.png" },
    ];

    // Card width/positions recomputed for 0.5" side margins and 0.3" gaps (previously 0.22" — too tight)
    plots.forEach((p, i) => {
      const x = 0.5 + i * 3.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 2.8, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: 1.4, w: 2.5, h: 2.3, fill: { color: "EBF5FB" }, rectRadius: 0.08 });
      s.addText("[ Image Placeholder ]", {
        x: x + 0.15, y: 1.4, w: 2.5, h: 1.1,
        fontSize: 11, fontFace: "Calibri", color: ACCENT, align: "center", valign: "bottom", italic: true
      });
      s.addText("Insert your\nresult screenshot", {
        x: x + 0.15, y: 2.5, w: 2.5, h: 1.0,
        fontSize: 12, fontFace: "Calibri", bold: true, color: MID, align: "center", valign: "middle"
      });
      s.addText(p.title, {
        x: x + 0.1, y: 3.85, w: 2.6, h: 0.4, fontSize: 11.5, fontFace: "Cambria", bold: true, color: MID, align: "center", margin: 0
      });
      s.addText(p.file, {
        x: x + 0.1, y: 4.3, w: 2.6, h: 0.3, fontSize: 9, fontFace: "Courier New", color: MUTED, align: "center", margin: 0
      });
    });

    s.addNotes("Three auto-generated plots: model accuracy bar chart, pie chart of sentiment distribution in the dataset, and a confusion matrix for the best model. Insert your actual screenshots here.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 9 – Tech Stack & How to Run
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconPy, "Tech Stack & How to Run", { fontSize: 23 });

    // Left: Tech Stack
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 3.7, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Tech Stack", { x: 0.7, y: 1.4, w: 3.3, h: 0.4, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    const stack = [
      { lib: "Python 3.8+", role: "Core language" },
      { lib: "Scikit-learn", role: "ML models & TF-IDF" },
      { lib: "Pandas / NumPy", role: "Data manipulation" },
      { lib: "Matplotlib / Seaborn", role: "Visualization & plots" },
      { lib: "Streamlit", role: "Web UI (optional)" },
    ];
    stack.forEach((item, i) => {
      const y = 1.92 + i * 0.66;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.65, y, w: 3.4, h: 0.55, fill: { color: LIGHT }, rectRadius: 0.07 });
      s.addText(item.lib, { x: 0.8, y: y + 0.02, w: 1.85, h: 0.5, fontSize: 11.5, fontFace: "Courier New", bold: true, color: MID, valign: "middle", margin: 0 });
      s.addText(item.role, { x: 2.65, y: y + 0.02, w: 1.3, h: 0.5, fontSize: 10, fontFace: "Calibri", color: MUTED, valign: "middle", margin: 0 });
    });

    // Right: Steps to run
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.4, y: 1.25, w: 5.1, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Steps to Run", { x: 4.6, y: 1.4, w: 4.7, h: 0.4, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    const cmds = [
      { n: "1", step: "Install dependencies", cmd: "pip install -r requirements.txt" },
      { n: "2", step: "Train all models", cmd: "python train_model.py" },
      { n: "3A", step: "CLI prediction app", cmd: "python app.py" },
      { n: "3B", step: "Streamlit web app", cmd: "streamlit run streamlit_app.py" },
    ];
    cmds.forEach((c, i) => {
      const y = 1.92 + i * 0.79;
      s.addShape(pres.shapes.OVAL, { x: 4.6, y: y + 0.1, w: 0.36, h: 0.36, fill: { color: ACCENT } });
      s.addText(c.n, { x: 4.6, y: y + 0.1, w: 0.36, h: 0.36, fontSize: 9.5, fontFace: "Calibri", bold: true, color: DARK, align: "center", valign: "middle", margin: 0 });
      s.addText(c.step, { x: 5.05, y, w: 4.3, h: 0.32, fontSize: 12, fontFace: "Calibri", bold: true, color: TEXT, margin: 0 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.05, y: y + 0.33, w: 4.3, h: 0.35, fill: { color: DARK }, rectRadius: 0.06 });
      s.addText(c.cmd, { x: 5.1, y: y + 0.33, w: 4.2, h: 0.35, fontSize: 11, fontFace: "Courier New", color: ACCENT, valign: "middle", margin: 0 });
    });

    s.addNotes("Run train_model.py first — it outputs the saved model and all plots. Then choose CLI or Streamlit for interactive predictions. Streamlit opens at localhost:8501.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 10 – Future Improvements & Conclusion
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: DARK };

    s.addShape(pres.shapes.OVAL, { x: 7.5, y: 3.0, w: 4.0, h: 4.0, fill: { color: MID, transparency: 40 }, line: { color: MID } });

    addHeader(s, iconRocket, "Future Improvements & Conclusion", { circleColor: ACCENT, titleColor: WHITE, fontSize: 24 });

    const improvements = [
      { title: "LSTM / RNN", detail: "Sequence models to capture context & word order" },
      { title: "BERT / Transformers", detail: "State-of-the-art language understanding via fine-tuning" },
      { title: "Larger Dataset", detail: "More real-world Amazon / product review data" },
      { title: "Aspect-level Sentiment", detail: "Detect sentiment on specific product features" },
    ];

    s.addText("Possible Extensions", {
      x: 0.5, y: 1.3, w: 6, h: 0.4, fontSize: 15, fontFace: "Cambria", bold: true, color: ACCENT, margin: 0
    });

    improvements.forEach((imp, i) => {
      const y = 1.8 + i * 0.78;
      s.addImage({ data: iconCheck, x: 0.5, y: y + 0.06, w: 0.36, h: 0.36 });
      s.addText(imp.title, { x: 1.02, y, w: 5.6, h: 0.36, fontSize: 14, fontFace: "Cambria", bold: true, color: WHITE, margin: 0 });
      s.addText(imp.detail, { x: 1.02, y: y + 0.36, w: 5.6, h: 0.35, fontSize: 12, fontFace: "Calibri", color: "A8C8E8", margin: 0 });
    });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 5.0, w: 9.0, h: 0.45, fill: { color: ACCENT, transparency: 15 }, rectRadius: 0.1 });
    s.addText("RevMind demonstrates end-to-end ML pipeline: preprocess → feature extract → train → evaluate → deploy  |  Python · Scikit-learn · Streamlit", {
      x: 0.7, y: 5.0, w: 8.6, h: 0.45, fontSize: 11.5, fontFace: "Calibri", color: WHITE, valign: "middle", margin: 0
    });

    s.addNotes("Mention these are highlighted in the README's Viva Tips section. BERT is the natural next step. Wrap up by summarizing what RevMind achieves end-to-end.");
  }

  await pres.writeFile({ fileName: path.join(__dirname, "RevMind_Presentation.pptx") });
  console.log("Done! RevMind_Presentation.pptx written.");
})();