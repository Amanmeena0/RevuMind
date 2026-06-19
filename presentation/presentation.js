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
const DARK   = "0B0C10";   // deep charcoal - title & final slides
const MID    = "1F2833";   // slate blue - cards, headers
const ACCENT = "66FCF1";   // electric cyan - branding accent
const LIGHT  = "F4F6F9";   // clean gray-white - content background
const WHITE  = "FFFFFF";
const MUTED  = "64748B";
const TEXT   = "1E293B";
const GREEN  = "00B894";

// Shadow factory
const mkShadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 45, opacity: 0.13 });

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title  = "RevuMind V2 – AI-Powered Review Intelligence Platform";
  pres.author = "Aman Meena";

  // Pre-render icons (white for headers, green for checklist points)
  const iconBrain  = await iconToBase64Png(FaBrain,      WHITE, 300);
  const iconChart  = await iconToBase64Png(FaChartBar,   WHITE, 300);
  const iconCogs   = await iconToBase64Png(FaCogs,       WHITE, 300);
  const iconDB     = await iconToBase64Png(FaDatabase,   WHITE, 300);
  const iconPy     = await iconToBase64Png(FaPython,     WHITE, 300);
  const iconRocket = await iconToBase64Png(FaRocket,     WHITE, 300);
  const iconLayers = await iconToBase64Png(FaLayerGroup, WHITE, 300);
  const iconSearch = await iconToBase64Png(FaSearch,     WHITE, 300);
  const iconCheck  = await iconToBase64Png(FaCheckCircle, GREEN, 300);

  // Shared header motif used on content slides
  function addHeader(slide, iconData, titleText, { circleColor = MID, titleColor = DARK, fontSize = 24 } = {}) {
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

    // Decorative circle (top right)
    s.addShape(pres.shapes.OVAL, { x: 7.8, y: -0.6, w: 3.2, h: 3.2, fill: { color: MID, transparency: 30 }, line: { color: MID } });

    // Brain icon
    s.addImage({ data: iconBrain, x: 0.5, y: 0.78, w: 0.72, h: 0.72 });

    s.addText("RevuMind V2", {
      x: 1.38, y: 0.6, w: 7.2, h: 1.0,
      fontSize: 50, fontFace: "Cambria", bold: true, color: WHITE, valign: "middle", margin: 0
    });

    s.addText("AI-Powered Review Intelligence Platform", {
      x: 0.5, y: 1.75, w: 8.5, h: 0.5,
      fontSize: 19, fontFace: "Calibri", color: ACCENT, margin: 0
    });

    s.addText([
      { text: "Electronics & Communication Engineering  |  Year 3", options: { breakLine: true } },
      { text: "Subject: Machine Learning (V2 Upgrade)  |  Stack: PyTorch + Transformers + XGBoost", options: {} }
    ], {
      x: 0.5, y: 2.45, w: 9, h: 0.9,
      fontSize: 14, fontFace: "Calibri", color: "A8C8E8"
    });

    // GitHub tag
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.55, w: 4.2, h: 0.45, fill: { color: MID }, rectRadius: 0.08 });
    s.addText("github.com/ananaymeena3/RevuMind", {
      x: 0.5, y: 3.55, w: 4.2, h: 0.45,
      fontSize: 11, fontFace: "Calibri", color: ACCENT, align: "center", valign: "middle", margin: 0
    });

    s.addNotes("Introduce project: RevuMind V2 is an upgraded Review Intelligence Platform. It shifts from traditional text baselines to a complete 7-model deep learning and gradient boosted cascade, extracting fine-grained business insights.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 2 – Project Overview & Goal
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconSearch, "Project Overview");

    // Left column – Platform Objectives
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("The Goal of RevuMind V2", {
      x: 0.7, y: 1.42, w: 3.9, h: 0.45, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0
    });
    s.addText([
      { text: "Transform thousands of customer reviews into structured, actionable business insights.", options: { breakLine: true, color: TEXT, fontSize: 13 } },
      { text: "\nKey Capabilities:\n", options: { breakLine: true, bold: true, color: MID, fontSize: 14 } },
      { text: "🏷️  RoBERTa Overall Sentiment\n", options: { breakLine: true, color: TEXT, fontSize: 13 } },
      { text: "📍  spaCy TRF Named Entity Boundaries\n", options: { breakLine: true, color: TEXT, fontSize: 13 } },
      { text: "🔍  DeBERTa Aspect-Based Polarity\n", options: { breakLine: true, color: TEXT, fontSize: 13 } },
      { text: "📈  XGBoost Helpfulness Regression", options: { color: TEXT, fontSize: 13 } },
    ], { x: 0.7, y: 2.0, w: 3.9, h: 2.9, fontFace: "Calibri" });

    // Right column – Scope & Dataset
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Platform Ingestion", {
      x: 5.4, y: 1.42, w: 3.9, h: 0.45, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0
    });
    s.addText([
      { text: "Dataset: Amazon Fine Food Reviews", options: { bold: true, breakLine: true } },
      { text: "Ingests local dataset at archive/Reviews.csv", options: { breakLine: true } },
      { text: "Raw size: ~300MB containing 500k+ reviews", options: { breakLine: true } },
      { text: "\nIngested Columns (Step 1):", options: { bold: true, breakLine: true } },
      { text: "• ProductId & Score (Stars)\n• Summary & Text (Reviews)\n• HelpfulnessNumerator & Denominator\n• Time (Unix Timestamp)", options: { breakLine: true } },
      { text: "\nFocus: 100% Text Intelligence", options: { bold: true, color: "E53935" } },
    ], { x: 5.4, y: 2.0, w: 3.9, h: 2.9, fontSize: 13, fontFace: "Calibri", color: TEXT });

    s.addNotes("Explain the shift: the platform focuses entirely on text intelligence now. We ingest raw files from the local archive directory, parsing text columns, ratings, helpfulness votes, and timestamps.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 3 – Feature-Based Repository Layout
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconLayers, "Redesigned Repository Structure");

    // File tree box
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 4.3, h: 3.85, fill: { color: MID }, shadow: mkShadow(), rectRadius: 0.1 });
    s.addText([
      { text: "revumind/", options: { bold: true, color: ACCENT, breakLine: true } },
      { text: "├── core/            # Database configurations", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── db/              # ORM Schemas & models", options: { breakLine: true, color: "A8D5E2" } },
      { text: "├── pipeline/        # Ingestion, inference, ETL", options: { breakLine: true, color: "A8D5E2" } },
      { text: "│   ├── preprocess.py\n│   ├── train_all.py\n│   └── inference.py", options: { breakLine: true, color: "88C5C2" } },
      { text: "├── models/          # Model wrappers & weights", options: { breakLine: true, color: ACCENT } },
      { text: "│   ├── sentiment/ | ner/ | embeddings/\n│   └── absa/ | topics/ | helpfulness/", options: { breakLine: true, color: "88C5C2" } },
      { text: "├── utils/           # Text stats & readability", options: { breakLine: true, color: "A8D5E2" } },
      { text: "└── visualization/   # Streamlit UI layers", options: { color: "A8D5E2" } },
    ], { x: 0.7, y: 1.4, w: 3.9, h: 3.55, fontSize: 11, fontFace: "Courier New", color: WHITE });

    // Right: Pipeline Architecture Card
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Pipeline & Concerns Isolation", {
      x: 5.4, y: 1.42, w: 3.9, h: 0.45, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0
    });
    s.addText([
      { text: "Feature-Based Architecture", options: { bold: true, breakLine: true } },
      { text: "Each ML component (Sentiment, NER, Topic Model) is a self-contained directory containing model loading, evaluation, and weights configurations.", options: { breakLine: true } },
      { text: "\nPipeline Architecture", options: { bold: true, breakLine: true } },
      { text: "Separates ingestion (preprocess.py) from model wrappers, inference orchestrators (inference.py), database connections, and dashboards.", options: { breakLine: true } },
      { text: "\nGraceful Fallbacks", options: { bold: true, breakLine: true } },
      { text: "All deep learning wrappers fall back to lightweight models or rule-based heuristics to allow instant CPU-bound local debugging.", options: {} }
    ], { x: 5.4, y: 1.95, w: 3.9, h: 2.9, fontSize: 12.5, fontFace: "Calibri", color: TEXT });

    s.addNotes("Walk through the feature-based repository design. We isolate database connection setups (core/) from models, pipelines, and visualizations. Each model directory holds its training and inference wrappers.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 4 – Ingestion & Preprocessing Pipeline
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconCogs, "Data Ingestion & Text Preprocessing");

    const steps = [
      { n: "1", label: "Chunk Ingestion", detail: "preprocess.py\nStreams Reviews.csv in chunks to prevent memory limits." },
      { n: "2", label: "Noise Stripping", detail: "Cleans HTML tags, URLs, and strips emoji/unicode symbols." },
      { n: "3", label: "Language Filter", detail: "Applies heuristic character ratio limits to isolate English reviews." },
      { n: "4", label: "Unified Text", detail: "Merges Summary + Text into a clean review text format." }
    ];

    steps.forEach((st, i) => {
      const x = 0.4 + i * 2.35;
      s.addShape(pres.shapes.OVAL, { x: x + 0.6, y: 1.25, w: 1.0, h: 1.0, fill: { color: MID }, shadow: mkShadow() });
      s.addText(st.n, { x: x + 0.6, y: 1.25, w: 1.0, h: 1.0, fontSize: 22, fontFace: "Cambria", bold: true, color: ACCENT, align: "center", valign: "middle", margin: 0 });
      if (i < 3) {
        s.addShape(pres.shapes.RECTANGLE, { x: x + 1.63, y: 1.71, w: 0.7, h: 0.08, fill: { color: ACCENT } });
      }
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 2.45, w: 2.0, h: 2.5, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.1 });
      s.addText(st.label, { x: x + 0.25, y: 2.55, w: 1.9, h: 0.35, fontSize: 12, fontFace: "Cambria", bold: true, color: MID, align: "center", margin: 0 });
      s.addText(st.detail, { x: x + 0.25, y: 3.0, w: 1.9, h: 1.8, fontSize: 10.5, fontFace: "Calibri", color: MUTED, align: "center" });
    });

    s.addNotes("Explain Step 2 Preprocessing: We clean raw inputs in chunks, remove HTML/URLs/emojis, apply custom ASCII filters to keep English, and merge fields into a clean review text for downstream modeling.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 5 – Model Stack (Sentiment, NER, ABSA, Embeddings)
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconBrain, "State-of-the-Art Deep Learning Models");

    const models = [
      {
        name: "RoBERTa (Sentiment)",
        badge: "Overall Tone",
        color: "1565C0",
        points: [
          "Fine-tuned classification head for positive, negative, and neutral.",
          "Handles negation and context better than VADER rules.",
          "Outputs confidence scores for overall sentiment classification."
        ]
      },
      {
        name: "spaCy TRF + DeBERTa (NER / ABSA)",
        badge: "Aspect-Based Focus",
        color: "6A1B9A",
        points: [
          "spaCy Transformer extracts entity boundaries (Brands, Features).",
          "DeBERTa ABSA evaluates sentiment for each specific aspect.",
          "Resolves mixed sentiments (e.g., 'vivid screen but slow charging')."
        ]
      }
    ];

    models.forEach((m, i) => {
      const x = 0.5 + i * 4.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 4.3, h: 3.35, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 1.42, w: 1.8, h: 0.32, fill: { color: m.color }, rectRadius: 0.07 });
      s.addText(m.badge, { x: x + 0.2, y: 1.42, w: 1.8, h: 0.32, fontSize: 10, fontFace: "Calibri", bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
      s.addText(m.name, { x: x + 0.2, y: 1.84, w: 3.9, h: 0.42, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
      m.points.forEach((p, j) => {
        s.addText([{ text: p, options: { bullet: true } }], {
          x: x + 0.2, y: 2.35 + j * 0.55, w: 3.9, h: 0.5,
          fontSize: 12, fontFace: "Calibri", color: TEXT
        });
      });
    });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.78, w: 9.0, h: 0.45, fill: { color: "EEF2FF" }, shadow: mkShadow(), rectRadius: 0.08 });
    s.addText("★  Embeddings: all-MiniLM-L6-v2 maps text to 384-dimensional vectors for semantic searches", {
      x: 0.7, y: 4.78, w: 8.6, h: 0.45, fontSize: 12, fontFace: "Calibri", color: MID, bold: true, valign: "middle", margin: 0
    });

    s.addNotes("Explain the core deep learning stack: RoBERTa for overall sentiment, spaCy Transformer + DeBERTa for aspect sentiment mapping, and Sentence-Transformers for semantic vector search.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 6 – Topics & Helpfulness Modeling
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconChart, "Topic Modeling & Helpfulness Prediction");

    const components = [
      {
        title: "BERTopic Modeling (Step 7)",
        points: [
          "Ingests 384-dimensional Sentence Embeddings.",
          "Reduces space using UMAP projections.",
          "Clusters data points using HDBSCAN density models.",
          "Extracts topic keywords using class-based TF-IDF.",
          "Maps reviews to coherent feedback groups (e.g. Battery)."
        ],
        color: "0077B6"
      },
      {
        title: "XGBoost Helpfulness Regressor (Step 9)",
        points: [
          "Predicts the continuous helpfulness ratio (0.0 to 1.0).",
          "Calculates text stats (word/sentence count, character lengths).",
          "Extracts Flesch Reading Ease & Flesch-Kincaid Grade scores.",
          "Incorporate features: Sentiment score, aspect count, topic assigned.",
          "Saves fitted scaler and booster trees to disk."
        ],
        color: "E65100"
      }
    ];

    components.forEach((c, i) => {
      const x = 0.5 + i * 4.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      s.addText(c.title, { x: x + 0.2, y: 1.42, w: 3.9, h: 0.4, fontSize: 15, fontFace: "Cambria", bold: true, color: c.color, margin: 0 });
      c.points.forEach((p, j) => {
        s.addText([{ text: p, options: { bullet: true } }], {
          x: x + 0.2, y: 1.95 + j * 0.55, w: 3.9, h: 0.5,
          fontSize: 11.5, fontFace: "Calibri", color: TEXT
        });
      });
    });

    s.addNotes("Outline the machine learning processes: BERTopic coordinates clusters using UMAP + HDBSCAN + c-TF-IDF. XGBoost regression models helpfulness scores from engineered tabular features (lengths, readability, sentiment).");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 7 – Ingestions, Summaries, and Insights
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconDB, "DB Ingestions, Cohort Summaries & Insights");

    // Ingestions
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Database Ingestion Pipeline", { x: 0.7, y: 1.42, w: 3.9, h: 0.4, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    s.addText([
      { text: "DB Pipeline: ingest_to_db.py", options: { bold: true, breakLine: true } },
      { text: "Ingests raw reviews, feeds them through the inference engine, and populates SQLAlchemy relational database tables.", options: { breakLine: true } },
      { text: "\nORM Schemas:", options: { bold: true, breakLine: true } },
      { text: "• products & reviews (core data)\n• review_embeddings (384-d vectors)\n• entities & aspect_sentiments\n• topics & summaries (aggregations)", options: {} }
    ], { x: 0.7, y: 1.95, w: 3.9, h: 2.9, fontSize: 12.5, fontFace: "Calibri", color: TEXT });

    // Summaries
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.25, w: 4.3, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("Step 8 BART Summaries & Step 10 Insights", { x: 5.4, y: 1.42, w: 3.9, h: 0.4, fontSize: 15, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    s.addText([
      { text: "Grouped BART Summaries", options: { bold: true, breakLine: true } },
      { text: "Groups reviews by cohort (product, sentiment type, or topic cluster) and generates concise executive summaries using BART weights.", options: { breakLine: true } },
      { text: "\nBusiness Insights Engine", options: { bold: true, breakLine: true } },
      { text: "Aggregates positive praises vs. negative complaints and automatically synthesizes actionable business recommendations to resolve pain points.", options: {} }
    ], { x: 5.4, y: 1.95, w: 3.9, h: 2.9, fontSize: 12.5, fontFace: "Calibri", color: TEXT });

    s.addNotes("Describe the connection: ingest_to_db.py orchestrates DB insertion. ExecutiveSummarizer runs BART to summarize review clusters, while the Insights engine generates praises, complaints, and product recommendations.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 8 – Visual Dashboard Layer
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconChart, "Visual Dashboard Layer (Step 11)");

    const elements = [
      { title: "Real-Time KPI Cards", desc: "Monitors review volumes, sentiment ratios, and helpfulness metrics dynamically.", img: "real_time_kpi_cards.jpg" },
      { title: "Aspect Sentiment Heatmap", desc: "Displays positive, neutral, and negative metrics across key product aspect features.", img: "aspect_sentiment_heatmap.jpg" },
      { title: "Extractive Cohort summaries", desc: "Renders executive summary blocks generated from review cohorts.", img: "extractive_cohort_summaries.jpg" }
    ];

    elements.forEach((p, i) => {
      const x = 0.5 + i * 3.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 2.8, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
      
      const imgPath = path.join(__dirname, "../reports/figures", p.img);
      s.addImage({
        path: imgPath,
        x: x + 0.15, y: 1.4, w: 2.5, h: 2.1
      });

      s.addText(p.title, {
        x: x + 0.1, y: 3.7, w: 2.6, h: 0.4, fontSize: 12, fontFace: "Cambria", bold: true, color: MID, align: "center", margin: 0
      });
      s.addText(p.desc, {
        x: x + 0.1, y: 4.15, w: 2.6, h: 0.8, fontSize: 10.5, fontFace: "Calibri", color: MUTED, align: "center", margin: 0
      });
    });

    s.addNotes("Review the dashboard implementation: We updated dashboard.py to load real data directly from the SQLite database. It runs interactive charts, filters products dynamically, and includes a fallback mode.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 9 – V2 Tech Stack & Execution Commands
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: LIGHT };
    addHeader(s, iconPy, "V2 Tech Stack & Commands", { fontSize: 23 });

    // Left: Tech Stack
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.25, w: 3.7, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("V2 Tech Stack", { x: 0.7, y: 1.4, w: 3.3, h: 0.4, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    const stack = [
      { lib: "PyTorch & Transformers", role: "RoBERTa, DeBERTa, BART" },
      { lib: "spaCy & regex Matchers", role: "NER entity boundary mapping" },
      { lib: "XGBoost & Scikit-Learn", role: "Regressor & fallbacks" },
      { lib: "SQLAlchemy & pgvector", role: "ORM database schemas" },
      { lib: "Streamlit & Plotly", role: "Interactive dashboard layer" },
    ];
    stack.forEach((item, i) => {
      const y = 1.92 + i * 0.66;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.65, y, w: 3.4, h: 0.55, fill: { color: LIGHT }, rectRadius: 0.07 });
      s.addText(item.lib, { x: 0.75, y: y + 0.02, w: 1.95, h: 0.5, fontSize: 10.5, fontFace: "Courier New", bold: true, color: MID, valign: "middle", margin: 0 });
      s.addText(item.role, { x: 2.7, y: y + 0.02, w: 1.25, h: 0.5, fontSize: 9.5, fontFace: "Calibri", color: MUTED, valign: "middle", margin: 0 });
    });

    // Right: Commands
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.4, y: 1.25, w: 5.1, h: 3.85, fill: { color: WHITE }, shadow: mkShadow(), rectRadius: 0.12 });
    s.addText("How to Run Pipelines", { x: 4.6, y: 1.4, w: 4.7, h: 0.4, fontSize: 16, fontFace: "Cambria", bold: true, color: MID, margin: 0 });
    const cmds = [
      { n: "1", step: "Setup DB Tables", cmd: "python revumind/db/init_db.py" },
      { n: "2", step: "Run Training Pipeline", cmd: "python revumind/pipeline/train_all.py --sample_size 5000" },
      { n: "3", step: "Ingest Reviews to DB", cmd: "python revumind/pipeline/ingest_to_db.py --sample_size 100" },
      { n: "4", step: "Run Interactive UI", cmd: "streamlit run revumind/visualization/dashboard.py" },
    ];
    cmds.forEach((c, i) => {
      const y = 1.92 + i * 0.79;
      s.addShape(pres.shapes.OVAL, { x: 4.6, y: y + 0.1, w: 0.36, h: 0.36, fill: { color: ACCENT } });
      s.addText(c.n, { x: 4.6, y: y + 0.1, w: 0.36, h: 0.36, fontSize: 9.5, fontFace: "Calibri", bold: true, color: DARK, align: "center", valign: "middle", margin: 0 });
      s.addText(c.step, { x: 5.05, y, w: 4.3, h: 0.32, fontSize: 12, fontFace: "Calibri", bold: true, color: TEXT, margin: 0 });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.05, y: y + 0.33, w: 4.3, h: 0.35, fill: { color: DARK }, rectRadius: 0.06 });
      s.addText(c.cmd, { x: 5.1, y: y + 0.33, w: 4.2, h: 0.35, fontSize: 11, fontFace: "Courier New", color: ACCENT, valign: "middle", margin: 0 });
    });

    s.addNotes("Highlight execution commands: Run init_db.py to configure schemas, run train_all.py to fit topics/helpfulness, run ingest_to_db.py to run the ML cascade and load DB rows, then launch Streamlit.");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 10 – Future Scalability & Conclusion
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: DARK };

    s.addShape(pres.shapes.OVAL, { x: 7.5, y: 3.0, w: 4.0, h: 4.0, fill: { color: MID, transparency: 40 }, line: { color: MID } });

    addHeader(s, iconRocket, "Future Scalability & Conclusion", { circleColor: ACCENT, titleColor: WHITE, fontSize: 24 });

    const improvements = [
      { title: "Multi-Tenant isolation", detail: "Tenant database schemas to support multiple business accounts." },
      { title: "Distributed Streaming Pipelines", detail: "Integrate Kafka + Flink to process reviews in near real-time." },
      { title: "Conversational RAG Agent", detail: "Expose vector embeddings to LLMs so users can query the reviews." },
      { title: "Knowledge Distillation", detail: "Distill models into student models to cut GPU hosting costs." },
    ];

    s.addText("Future Scaling Path", {
      x: 0.5, y: 1.3, w: 6, h: 0.4, fontSize: 15, fontFace: "Cambria", bold: true, color: ACCENT, margin: 0
    });

    improvements.forEach((imp, i) => {
      const y = 1.8 + i * 0.78;
      s.addImage({ data: iconCheck, x: 0.5, y: y + 0.06, w: 0.36, h: 0.36 });
      s.addText(imp.title, { x: 1.02, y, w: 5.6, h: 0.36, fontSize: 14, fontFace: "Cambria", bold: true, color: WHITE, margin: 0 });
      s.addText(imp.detail, { x: 1.02, y: y + 0.36, w: 5.6, h: 0.35, fontSize: 12, fontFace: "Calibri", color: "A8C8E8", margin: 0 });
    });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 5.0, w: 9.0, h: 0.45, fill: { color: ACCENT, transparency: 15 }, rectRadius: 0.1 });
    s.addText("RevuMind V2 provides a production-ready AI Review Intelligence system  |  PyTorch · Transformers · XGBoost · FastAPI", {
      x: 0.7, y: 5.0, w: 8.6, h: 0.45, fontSize: 11.5, fontFace: "Calibri", color: WHITE, valign: "middle", margin: 0
    });

    s.addNotes("Wrap up: RevuMind V2 upgrades from classical baselines to a production 7-model NLP cascade. Future targets include multi-tenant databases, Kafka stream ingestion, RAG conversational agents, and student model compression.");
  }

  await pres.writeFile({ fileName: path.join(__dirname, "RevuMind_V2_Presentation.pptx") });
  console.log("Done! RevuMind_V2_Presentation.pptx written.");
})();