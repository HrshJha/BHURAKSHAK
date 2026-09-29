import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root = '/Users/harshkumarjha/mine';
const skillDir = '/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const tmpDir = path.join(root, '.codex-build/feasibility-econ-ml');
const outputDir = path.join(root, 'submit/feasibility_economic_graph');
const finalPath = path.join(outputDir, 'Feasibility_ML_Economic_Graph_researched.pptx');
const imgPath = '/var/folders/dh/_5m7l0cd7gv5nypd9dv3lyyh0000gn/T/codex-clipboard-b60324f5-9268-46d7-91ef-c314d7c7462f.png';
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const img = new Uint8Array(await fs.readFile(imgPath));
const pres = Presentation.create({ slideSize: { width: 1964, height: 1100 } });
const slide = pres.slides.add();
slide.background.fill = '#FFFFFF';
slide.images.add({ blob: img, contentType: 'image/png', alt: 'User-provided feasibility and viability slide screenshot', fit: 'cover', position: { left: 0, top: 0, width: 1964, height: 1100 } });

// Replace only the screenshot's red-marked empty area with the two editable charts.
slide.shapes.add({ geometry: 'rect', name: 'white-chart-canvas', position: { left: 761, top: 725, width: 718, height: 362 }, fill: '#FFFFFF', line: { fill: 'none', width: 0 } });

const ml = slide.charts.add('bar', {
  position: { left: 774, top: 746, width: 329, height: 285 },
  title: 'Locked-test ML trade-off (%)', titlePlacement: 'aboveChart',
  titleTextStyle: { typeface: font, fontSize: 18, bold: true, fill: '#1F6F37', alignment: 'center' },
  categories: ['Tuned XGB', 'Default XGB', 'Logistic', 'Threshold'],
  series: [
    { name: 'CRITICAL recall', values: [9.33, 4.57, 76.50, 4.16], valuesFormatCode: '0.0', fill: '#2D8A4B' },
    { name: 'NORMAL false alarms', values: [15.08, 3.72, 62.63, 1.74], valuesFormatCode: '0.0', fill: '#E3A33B' },
  ],
  barOptions: { direction: 'bar', grouping: 'clustered', gapWidth: 70, overlap: 0 },
  hasLegend: true,
  legend: { position: 'bottom', overlay: false, textStyle: { typeface: font, fontSize: 10, fill: '#222222' } },
  xAxis: { min: 0, max: 80, majorUnit: 20, numberFormatCode: '0', textStyle: { typeface: font, fontSize: 10, fill: '#444444' }, line: { fill: '#777777', width: 1 }, majorGridlines: { fill: '#DADADA', width: 1 } },
  yAxis: { textStyle: { typeface: font, fontSize: 11, fill: '#222222' }, line: { fill: 'none', width: 0 }, majorGridlines: null },
  dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: font, fontSize: 9, bold: true, fill: '#1A1A1A' } },
  chartFill: '#FFFFFF', chartLine: { fill: 'none', width: 0 }, plotAreaFill: '#FFFFFF', plotAreaLine: { fill: 'none', width: 0 },
});
applyPresentationChartFont(ml, { fontFamily: font });

const econ = slide.charts.add('bar', {
  position: { left: 1106, top: 746, width: 356, height: 285 },
  title: 'Priced parts subtotal (₹000)', titlePlacement: 'aboveChart',
  titleTextStyle: { typeface: font, fontSize: 18, bold: true, fill: '#1F6F37', alignment: 'center' },
  categories: ['6 sensor nodes', 'Gateway core*', 'Parts subtotal*'],
  series: [{ name: 'Retail list price', values: [22.956, 8.398, 31.354], valuesFormatCode: '0.0', fill: '#2D8A4B' }],
  barOptions: { direction: 'bar', grouping: 'clustered', gapWidth: 85 },
  hasLegend: false,
  xAxis: { min: 0, max: 35, majorUnit: 10, numberFormatCode: '0', textStyle: { typeface: font, fontSize: 10, fill: '#444444' }, line: { fill: '#777777', width: 1 }, majorGridlines: { fill: '#DADADA', width: 1 } },
  yAxis: { textStyle: { typeface: font, fontSize: 11, fill: '#222222' }, line: { fill: 'none', width: 0 }, majorGridlines: null },
  dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: font, fontSize: 11, bold: true, fill: '#1A1A1A' } },
  chartFill: '#FFFFFF', chartLine: { fill: 'none', width: 0 }, plotAreaFill: '#FFFFFF', plotAreaLine: { fill: 'none', width: 0 },
});
applyPresentationChartFont(econ, { fontFamily: font });

function textBox(name, text, x, y, w, h, size = 13, color = '#444444') {
  const s = slide.shapes.add({ geometry: 'textbox', name, position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  s.text = text;
  s.text.style = { typeface: font, fontSize: size, color, alignment: 'center', autoFit: 'shrinkText' };
  return s;
}
textBox('ml-source-note', 'Same locked synthetic holdout; recall vs NORMAL false alarms.', 773, 1034, 333, 33, 12);
textBox('economic-source-note', '*₹3,826/node; gateway Pi 4 + LoRa board. Excludes Pi accessories,\nshipping, wiring, field fit & labour. Retail snapshot: 29 Sep 2026.', 1104, 1027, 358, 50, 11);
slide.speakerNotes.textFrame.setText(`ML: reports/final_eval.md and reports/test_lock.json. One locked synthetic holdout (seed 20260929, evals_run=1). Paired values, percent: tuned XGBoost critical recall 9.33, normal false-alarm rate 15.08; default XGBoost 4.57, 3.72; logistic regression 76.50, 62.63; threshold rule 4.16, 1.74. Synthetic corpus only; not a real-mine evaluation.\n\nCost snapshot: 29 Sep 2026, Indian retail listings. Per sensor node: Heltec Wireless Stick V3 with antenna and Li-ion charging/protection ₹2,749 (https://vihaaniotgateway.in/product/heltec-wireless-stick-v3-sx1262-esp32-lora-oled/); MPU-6050 breakout ₹173 (https://robokits.co.in/sensors/accelerometer-and-magnetometer/triple-axis-accelerometer-gyro-mpu-6050-breakout?cPath=72_213); VL53L1X ToF ₹570 (https://www.ktron.in/product/vl53l1x-laser-ranging-module-tof/); vibration switch module ₹42 (https://robokits.co.in/development-board/sensors-compatible-with-arduino); waterproof DS18B20 probe ₹55 (https://robu.in/product-category/sensor-modules/temperature-humidity-sensor/); 18650 2200mAh cell ₹110 (https://robodo.in/products/icr-18650-2200mah-2c-lithium-ion-battery); F4-2 100x68x50mm IP65 ABS box ₹127 (https://robu.in/product-category/workbench-tools-and-kits/). Total ₹3,826/node; six nodes ₹22,956.\nGateway core: Raspberry Pi 4 Model B 4GB ₹5,649 including GST (https://robu.in/product-category/microcontroller-development-board/raspberry-pi-microcontroller-development-board/official-boards-and-accessories/raspberry-pi-4/) plus Heltec Wireless Stick V3 LoRa board ₹2,749 (same product link), total ₹8,398. Combined priced-parts subtotal ₹31,354. The Pi radio-interface pairing is a costing assumption and has not been built or validated in this project. The subtotal omits Pi storage, power supply, case, cables, wiring, connectors, enclosure adaptations, shipping, labour, field installation, solar/battery backup for the gateway, and commissioning. Retail prices and stock can change; this is an indicative component basket, not an installed system quote or an ROI claim.`);

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const draftPath = path.join(tmpDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draftPath);
const preview = await pres.export({ slide, format: 'png', scale: 1 });
await fs.writeFile(path.join(outputDir, 'Feasibility_ML_Economic_Graph_v2_preview.png'), new Uint8Array(await preview.arrayBuffer()));
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
  receiptPath: path.join(tmpDir, 'validation-researched.json'),
});
console.log(JSON.stringify({ finalPath, result, font }));
