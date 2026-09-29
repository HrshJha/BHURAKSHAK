import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const runtimeModules = '/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtimeModules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);

const root = '/Users/harshkumarjha/mine';
const skillDir = '/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const tmpDir = path.join(root, '.codex-build/ml-impact-outcomes');
const outputDir = path.join(root, 'submit/ml_impact_slide');
const finalPath = path.join(outputDir, 'Impact_Benefits_Red_Box_ML_benefit.pptx');
const imgPath = '/var/folders/dh/_5m7l0cd7gv5nypd9dv3lyyh0000gn/T/codex-clipboard-57ba7821-990c-43f7-9650-7999f5d10d96.png';
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: 'Arial' });
const img = new Uint8Array(await fs.readFile(imgPath));
const pres = Presentation.create({ slideSize: { width: 1964, height: 1100 } });
const slide = pres.slides.add();
slide.background.fill = '#FFFFFF';
slide.images.add({ blob: img, contentType: 'image/png', alt: 'User-provided Impact and Benefits slide screenshot', fit: 'cover', position: { left: 0, top: 0, width: 1964, height: 1100 } });

// Replace only the marked empty rectangle. The previous mixed-domain bars are not comparable.
slide.shapes.add({ geometry: 'rect', name: 'cover-marked-box', position: { left: 988, top: 663, width: 976, height: 437 }, fill: '#FFFFFF', line: { fill: 'none', width: 0 } });
slide.shapes.add({ geometry: 'roundRect', name: 'impact-graph-panel', position: { left: 990, top: 675, width: 963, height: 414 }, fill: '#F4F0FC', line: { fill: '#D9C8FC', width: 3 } });

function textBox(name, text, x, y, w, h, size, color = '#27323B', bold = false, align = 'left') {
  const s = slide.shapes.add({ geometry: 'textbox', name, position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  s.text = text;
  s.text.style = { typeface: font, fontSize: size, color, bold, alignment: align, autoFit: 'shrinkText' };
  return s;
}

textBox('graph-heading', 'HOW ML COULD HELP', 1034, 690, 865, 29, 20, '#6D2926', true);
textBox('graph-subheading', 'From sensor-like data to a risk pattern for review', 1034, 720, 865, 24, 14, '#44515F');

const stages = [
  { x: 1034, title: '1  INPUT', detail: 'Simulated sensor patterns\n(current model evidence)', fill: '#E5F4E0', stroke: '#91C879' },
  { x: 1252, title: '2  ML SCREENS', detail: 'Anomaly score +\nphysics check +\nrisk classification', fill: '#EAF3F8', stroke: '#8AB6CE' },
  { x: 1470, title: '3  ALERT LOGIC', detail: 'Combines repeated\nrisk into an alert\nlevel', fill: '#F8F1E2', stroke: '#D7B36E' },
  { x: 1688, title: '4  HUMAN REVIEW', detail: 'Could help focus\nwhich patterns need\nfollow-up', fill: '#EFE9FA', stroke: '#B8A3D9' },
];
for (const [i, stage] of stages.entries()) {
  slide.shapes.add({ geometry: 'roundRect', name: `benefit-stage-${i + 1}`, position: { left: stage.x, top: 780, width: 190, height: 128 }, fill: stage.fill, line: { fill: stage.stroke, width: 2 } });
  textBox(`benefit-stage-${i + 1}-title`, stage.title, stage.x + 12, 795, 166, 23, 14, '#26353E', true, 'center');
  textBox(`benefit-stage-${i + 1}-detail`, stage.detail, stage.x + 12, 826, 166, 68, 13, '#3F4B54', false, 'center');
}
for (const [i, x] of [1220, 1438, 1656].entries()) {
  textBox(`benefit-flow-arrow-${i + 1}`, '→', x, 822, 28, 38, 26, '#318A50', true, 'center');
}

slide.shapes.add({ geometry: 'line', name: 'benefit-status-divider', position: { left: 1034, top: 930, width: 858, height: 1 }, line: { fill: '#D5C7EB', width: 1 } });
textBox('benefit-label', 'POSSIBLE BENEFIT', 1034, 948, 230, 24, 15, '#1F6F37', true);
textBox('benefit-explanation', 'Highlight changing risk patterns so teams can prioritize follow-up.', 1034, 975, 858, 28, 17, '#27323B', true);
textBox('benefit-evidence', 'Evidence today: synthetic tests only. Earlier warning and real-mine benefit are not demonstrated.', 1034, 1014, 858, 28, 14, '#6D2926');
slide.speakerNotes.textFrame.setText(`This graphic explains the intended ML benefit pathway, not a demonstrated operational outcome. The current model evaluation uses the synthetic held-out corpus (reports/final_eval.md and reports/test_lock.json, evals_run=1). Source architecture: src/pipeline.py combines model inputs including anomaly_score and physics_residual; src/risk/alert_engine.py applies alert thresholds and persistence; reports/acceptance_criteria.md says there is no early-warning success claim, no physical tabletop/real-mine validation, and the reported locked-test alert median lead is -3.33 hours. The intended user value is helping prioritize patterns for follow-up. No claim is made that the current model has reduced harm, improved early warning, or been used in field operations.`);

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const draftPath = path.join(tmpDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draftPath);
const preview = await pres.export({ slide, format: 'png', scale: 1 });
await fs.writeFile(path.join(outputDir, 'Impact_Benefits_Red_Box_ML_benefit_preview.png'), new Uint8Array(await preview.arrayBuffer()));
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: '/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '18707100,10477500'],
  requiredNativeChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: false,
  explicitTotalSlideCount: 1,
  fontPolicy: { basis: 'design', families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(tmpDir, 'validation-benefit.json'),
});
console.log(JSON.stringify({ finalPath, result, font }));
