import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const root = '/Users/harshkumarjha/mine';
const runtimeModules = '/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtimeModules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const skillDir = '/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: 'Arial' });
const tmpDir = path.join(root, '.codex-build/ml-feasibility-tradeoff');
const outputDir = path.join(root, 'submit/ml_feasibility_slide');
const imagePath = path.join(outputDir, 'ml_feasibility_preview.png');
const finalPath = path.join(outputDir, 'ML_Feasibility_Viability_tradeoff_v2.pptx');
const image = new Uint8Array(await fs.readFile(imagePath));

const pres = Presentation.create({ slideSize: { width: 1964, height: 1100 } });
const slide = pres.slides.add();
slide.background.fill = '#FFFFFF';
slide.images.add({ blob: image, contentType: 'image/png', alt: 'Existing Feasibility and Viability slide screenshot', fit: 'cover', position: { left: 0, top: 0, width: 1964, height: 1100 } });

// Cover only the contents of the middle ML validation card; preserve its frame and both side columns.
slide.shapes.add({ geometry: 'rect', name: 'cover-previous-ml-validation-content', position: { left: 484, top: 238, width: 904, height: 594 }, fill: '#F2EEFF', line: { fill: 'none', width: 0 } });

function textBox(name, text, x, y, w, h, size, color, bold = false) {
  const s = slide.shapes.add({ geometry: 'textbox', name, position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  s.text = text;
  s.text.style = { typeface: font, fontSize: size, color, bold, alignment: 'left', autoFit: 'shrinkText' };
  return s;
}

textBox('ml-validation-kicker', 'TECHNICAL FEASIBILITY  /  ML VALIDATION', 513, 253, 840, 22, 13, '#596777', true);
textBox('ml-validation-title', 'ML Performance Trade-off (Locked Synthetic Test)', 513, 277, 850, 31, 23, '#26313D', true);
textBox('ml-validation-subtitle', 'Critical recall compared with normal false alarms · same held-out test', 513, 309, 850, 24, 14, '#586675');

const chart = slide.charts.add('bar', {
  position: { left: 504, top: 335, width: 861, height: 413 },
  categories: ['Tuned XGBoost', 'Default XGBoost', 'Logistic Regression', 'Threshold Rule'],
  series: [
    { name: 'Normal false alarms', values: [15.1, 3.7, 62.6, 1.7], valuesFormatCode: '0.0"%"', fill: '#E6A23C' },
    { name: 'Critical recall', values: [9.3, 4.6, 76.5, 4.2], valuesFormatCode: '0.0"%"', fill: '#2D8A4B' },
  ],
  barOptions: { direction: 'bar', grouping: 'clustered', gapWidth: 72, overlap: 0 },
  hasLegend: true,
  legend: { position: 'bottom', overlay: false, textStyle: { typeface: font, fontSize: 13, fill: '#26313D' } },
  xAxis: { min: 0, max: 100, majorUnit: 20, numberFormatCode: '0', title: { text: 'Percent (%)', textStyle: { typeface: font, fontSize: 12, fill: '#394653' } }, textStyle: { typeface: font, fontSize: 12, fill: '#4D5966' }, line: { fill: '#87929E', width: 1 }, majorGridlines: { fill: '#D9DDE5', width: 1 } },
  yAxis: { textStyle: { typeface: font, fontSize: 14, fill: '#34404C' }, line: { fill: 'none', width: 0 }, majorGridlines: null },
  dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: font, fontSize: 13, bold: true, fill: '#26313D' } },
  chartFill: '#F2EEFF', chartLine: { fill: 'none', width: 0 }, plotAreaFill: '#F2EEFF', plotAreaLine: { fill: 'none', width: 0 },
});
applyPresentationChartFont(chart, { fontFamily: font });

textBox('ml-validation-footnote', 'Evaluation on held-out synthetic data; real-mine performance remains unvalidated.', 513, 779, 850, 28, 14, '#586675');
slide.speakerNotes.textFrame.setText('ML Validation results from the one-time locked synthetic evaluation in reports/final_eval.md; reports/test_lock.json records evals_run=1. Values displayed exactly as provided: Threshold Rule: normal false alarms 1.7%, Critical recall 4.2%; Logistic Regression: 62.6%, 76.5%; Default XGBoost: 3.7%, 4.6%; Tuned XGBoost: 15.1%, 9.3%. This is held-out synthetic data, not real-mine evaluation.');

await fs.mkdir(tmpDir, { recursive: true });
const draftPath = path.join(tmpDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draftPath);
execFileSync('/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3', [path.join(tmpDir, 'patch_axis.py'), draftPath]);
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: '/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '18707100,10477500'],
  requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  explicitTotalSlideCount: 1,
  fontPolicy: { basis: 'design', families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(tmpDir, 'validation-v2.json'),
});
console.log(JSON.stringify({ finalPath, result, font }));
