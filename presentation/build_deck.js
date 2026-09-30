const pptxgen = require("pptxgenjs");

const NAVY = "1F3864";
const LIGHT_BG = "F2F5FA";
const ACCENT = "C0621B";
const TEXT = "1F3864";
const MUTED = "595959";
const WHITE = "FFFFFF";
const FIG = "../reports/figures";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

function header(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.5, y: 0.35, w: 12.3, h: 0.7, fontSize: 28, bold: true, color: TEXT,
    fontFace: "Calibri", isTextBox: true, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 0.98, w: 12.3, h: 0.4, fontSize: 14, italic: true, color: MUTED,
      fontFace: "Calibri", isTextBox: true, margin: 0,
    });
  }
}

function bullets(slide, x, y, w, h, items, fontSize = 15, color = TEXT) {
  slide.addText(
    items.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize, color, fontFace: "Calibri", isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 10 }
  );
}

function imageSlide(title, subtitle, path, imgW, imgH, takeaways) {
  const slide = pres.addSlide();
  header(slide, title, subtitle);
  const maxW = 7.6, maxH = 5.3;
  const ratio = Math.min(maxW / imgW, maxH / imgH);
  const w = imgW * ratio, h = imgH * ratio;
  slide.addImage({ path, x: 0.5, y: 1.6, w, h });
  slide.addShape("roundRect", { x: 8.5, y: 1.6, w: 4.3, h: 5.3, rectRadius: 0.06, fill: { color: LIGHT_BG }, line: { color: LIGHT_BG } });
  slide.addText("Key takeaway", { x: 8.75, y: 1.8, w: 3.8, h: 0.35, fontSize: 14, bold: true, color: ACCENT, fontFace: "Calibri", isTextBox: true, margin: 0 });
  bullets(slide, 8.75, 2.25, 3.8, 4.5, takeaways, 13);
  return slide;
}

// 1. Title
let slide = pres.addSlide();
slide.background = { color: NAVY };
slide.addText("Sales Forecasting &\nRegional Performance", {
  x: 0.7, y: 2.2, w: 11.5, h: 2.0, fontSize: 38, bold: true, color: WHITE,
  fontFace: "Calibri", isTextBox: true, margin: 0,
});
slide.addText("Comprehensive Data Science Project", {
  x: 0.7, y: 4.1, w: 10, h: 0.5, fontSize: 16, italic: true, color: "C9D6EE",
  fontFace: "Calibri", isTextBox: true, margin: 0,
});
slide.addText("Data Science Internship Capstone  |  100 transactions, Jan\u2013Apr 2024", {
  x: 0.7, y: 6.7, w: 10, h: 0.4, fontSize: 12, color: "C9D6EE", fontFace: "Calibri", isTextBox: true, margin: 0,
});

// 2. Business problem
slide = pres.addSlide();
header(slide, "Business Problem", "What leadership wants to know from 3 months of sales data");
bullets(slide, 0.6, 1.9, 11.8, 4.5, [
  "Where is sales performance strong or weak across regions and products?",
  "Can we forecast revenue for the next few weeks?",
  "Can we estimate what a new deal is worth, before it's priced?",
], 18);
slide.addText("Dataset: 100 transactions \u00b7 4 regions \u00b7 5 products \u00b7 \u20b912.4M total revenue \u00b7 no missing/duplicate data", {
  x: 0.6, y: 6.3, w: 11.8, h: 0.5, fontSize: 13, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0,
});

// 3. Revenue by region
imageSlide("Revenue by Region", "North and South lead; West trails", `${FIG}/01_revenue_by_region.png`, 1184, 731, [
  "North (\u20b93.98M) and South (\u20b93.74M) bring in the most revenue.",
  "West (\u20b92.12M) is clearly behind the other three regions.",
]);

// 4. Regional gap significance
slide = pres.addSlide();
header(slide, "Is the Regional Gap Real?", "A one-way ANOVA test on deal size across regions");
slide.addText("p \u2248 0.10", { x: 0.6, y: 2.0, w: 5, h: 1.0, fontSize: 44, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
slide.addText("(not significant at 5%, but close)", { x: 0.6, y: 3.0, w: 6, h: 0.5, fontSize: 14, italic: true, color: MUTED, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(slide, 0.6, 3.8, 11.8, 3, [
  "West's average deal (\u20b981.7K) is the lowest of all 4 regions (vs \u20b9132\u2013142K elsewhere).",
  "The test doesn't quite reach the standard 5% cutoff \u2014 likely because the sample is small (100 deals).",
  "The gap comes from fewer units sold per deal in West, not a different product mix.",
], 16);

// 5. Region x product heatmap
imageSlide("Revenue by Region & Product", "Checking if West's gap is about product mix", `${FIG}/04_region_product_heatmap.png`, 1097, 731, [
  "West sells across the same 5 products as other regions.",
  "The shortfall isn't concentrated in one product \u2014 it's a broad, region-wide pattern.",
]);

// 6. Weekly trend + forecast
imageSlide("Weekly Revenue Trend", "14 weeks of history \u2014 is there a trend?", `${FIG}/05_weekly_trend.png`, 1328, 732, [
  "No clear upward or downward trend visible yet.",
  "A flat historical average beat a linear trend line when tested on the last 3 weeks.",
  "Forecast for next 4 weeks: ~\u20b9824K/week.",
]);

// 7. Model comparison
imageSlide("Comparing Prediction Models", "5-fold cross-validation on 3 regression models", `${FIG}/06_model_comparison.png`, 1034, 731, [
  "Ridge Regression scored best on cross-validation.",
  "Random Forest overfit slightly with this little data.",
  "Ridge selected as the final model.",
]);

// 8. Actual vs predicted
imageSlide("Model Performance", "Ridge Regression on the test set", `${FIG}/07_actual_vs_predicted.png`, 884, 881, [
  "R\u00b2 \u2248 0.46, MAE \u2248 \u20b960,600.",
  "Points cluster reasonably around the perfect-prediction line.",
  "Good for rough pipeline sizing \u2014 not precise enough for a final quote.",
]);

// 9. Recommendations
slide = pres.addSlide();
header(slide, "Recommendations");
const recs = [
  ["Look into West's sales process", "Focus on deal size (bundling, upselling) rather than the product catalog."],
  ["Budget to the average, not a trend", "Use ~\u20b9824K/week; revisit monthly as more data arrives."],
  ["Use the model for early planning", "R\u00b2 0.46 \u2014 good for pipeline sizing, not final pricing."],
  ["Re-test significance later", "Regional gap is descriptively real; re-run the ANOVA as more deals come in."],
];
const w = 5.85, gap = 0.4;
recs.forEach((r, i) => {
  const x = 0.6 + (i % 2) * (w + gap);
  const y = 1.7 + Math.floor(i / 2) * 2.5;
  slide.addShape("roundRect", { x, y, w, h: 2.2, rectRadius: 0.06, fill: { color: LIGHT_BG }, line: { color: LIGHT_BG } });
  slide.addText(`${i + 1}. ${r[0]}`, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.5, fontSize: 15, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true, margin: 0 });
  bullets(slide, x + 0.25, y + 0.75, w - 0.5, 1.3, [r[1]], 13);
});

// 10. Thank you / deployment
slide = pres.addSlide();
slide.background = { color: NAVY };
slide.addText("Deployed Dashboard", { x: 0.7, y: 2.2, w: 10, h: 1.0, fontSize: 32, bold: true, color: WHITE, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(slide, 0.7, 3.4, 10.5, 2.5, [
  "Streamlit dashboard: sales overview, forecast, and deal-value estimator",
  "Full analysis, code, and reports on GitHub",
  "Thank you!",
], 16, WHITE);

pres.writeFile({ fileName: "capstone_presentation.pptx" }).then(() => console.log("done"));
