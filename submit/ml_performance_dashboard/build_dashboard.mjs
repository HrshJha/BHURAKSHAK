import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const skill = '/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const work = '/Users/harshkumarjha/mine';
const out = path.join(work, 'submit/ml_performance_dashboard');
const staging = path.join(work, '.codex-finalizer');
await fs.mkdir(out, {recursive:true});
await fs.mkdir(staging, {recursive:true});
const {resolvePresentationFont, applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont();
const pres = Presentation.create({slideSize:{width:1600,height:900}});
const slide=pres.slides.add(); slide.background.fill='#FFFFFF';
const ink='#171717', muted='#4C4C4C', line='#C9C9C9', pale='#F5F5F5';
function box(t,x,y,w,h,size=22,bold=false,color=ink,align='left'){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=t; s.text.style={typeface:font,fontSize:size,bold,color,alignment:align,autoFit:'none'};
  return s;
}
function rule(x,y,w){slide.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{style:'solid',fill:line,width:1}})}

box('ML PERFORMANCE  |  LOCKED SYNTHETIC TEST',60,38,1480,54,39,true);
box('Technical feasibility · SIH 2026',60,91,1480,28,18,false,muted);
rule(60,126,1480);

// Key evidence remains explicitly split between classifier and alert engine.
box('CLASSIFIER · TUNED XGBOOST',62,145,530,24,18,true,muted);
box('9.33%',62,169,260,54,42,true);
box('Critical recall',270,185,310,30,22);
box('ALERT ENGINE · SYSTEM LEVEL',810,145,650,24,18,true,muted);
box('−3.33 h',810,169,270,54,42,true);
box('Median lead time',1095,185,365,30,22);
rule(60,230,1480);

box('Model comparison',60,245,1480,38,28,true);
box('Classification and operational rates from the same locked synthetic test',60,278,1480,25,17,false,muted);
box('CLASSIFICATION',305,306,485,19,15,true,muted);
box('OPERATIONAL',790,306,225,19,15,true,muted);
box('COMPUTATIONAL · WORKSTATION CPU / SAMPLED RSS',1015,306,525,19,15,true,muted);

const source=JSON.parse(await fs.readFile(path.join(work,'reports/final_eval.json'),'utf8'));
const defs=[
  ['Tuned XGBoost','tuned_xgboost','0.317 ms','0.613 ms','550.625 MiB'],
  ['Default XGBoost','default_xgboost','Not measured','Not measured','Not measured'],
  ['Logistic Regression','logistic_regression','Not measured','Not measured','Not measured'],
  ['Threshold Rule','threshold_rule','Not measured','Not measured','Not measured'],
];
const pct=v=>(v*100).toFixed(2)+'%';
const frac=v=>v.toFixed(3);
const vals=[
 ['Model','Critical recall','Macro PR-AUC','Macro F1','Normal false-alarm rate','p50 latency','p95 latency','Sample memory'],
 ...defs.map(([name,key,p50,p95,mem])=>{const m=source.risk_models[key];return [name,pct(m.recall_critical),frac(m.pr_auc_macro_ovr),frac(m.f1_macro),pct(m.false_alarm_rate_normal),p50,p95,mem]})
];
const widths=[245,170,175,140,225,155,155,215];
const table=slide.tables.add({rows:5,columns:8,left:60,top:330,width:1480,height:250,columnWidths:widths,values:vals});
table.styleOptions={headerRow:false,bandedRows:false};
table.borders.assign({style:'solid',fill:line,width:0.7});
for(let r=0;r<5;r++){
  table.rows[r].height=r===0?54:49;
  for(let c=0;c<8;c++){
    let cell=table.getCell(r,c);
    cell.fill=r===0?'#ECECEC':r%2===0?pale:'#FFFFFF';
  }
}
table.cells.block({row:0,column:0,rowCount:5,columnCount:8}).assign({
  textStyle:{typeface:font,fontSize:18,color:ink},
  margins:{left:10,right:9,top:8,bottom:7},anchor:'middle'
});
table.cells.block({row:0,column:0,rowCount:1,columnCount:8}).assign({textStyle:{typeface:font,fontSize:17,color:ink,bold:true}});
table.cells.block({row:1,column:0,rowCount:4,columnCount:1}).assign({textStyle:{typeface:font,fontSize:19,color:ink,bold:true}});

box('Critical recall and normal false-alarm rate',60,594,760,38,25,true);
box('Classifier-level rates · percent of respective class',60,627,760,25,17,false,muted);
const cats=defs.map(x=>x[0]);
const recall=defs.map(x=>Number(source.risk_models[x[1]].recall_critical.toFixed(6)));
const far=defs.map(x=>Number(source.risk_models[x[1]].false_alarm_rate_normal.toFixed(6)));
const chart=slide.charts.add('bar',{
  position:{left:60,top:654,width:760,height:182},categories:cats,
  series:[{name:'Critical recall',values:recall,fill:'#222222',valuesFormatCode:'0.0%'},{name:'Normal false-alarm rate',values:far,fill:'#9A9A9A',valuesFormatCode:'0.0%'}],
  barOptions:{direction:'bar',grouping:'clustered',gapWidth:75},
  hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:15,fill:ink}},
  xAxis:{min:0,max:1,majorUnit:0.25,numberFormatCode:'0%',textStyle:{typeface:font,fontSize:14,fill:muted},majorGridlines:{style:'solid',fill:'#E4E4E4',width:0.7}},
  yAxis:{textStyle:{typeface:font,fontSize:15,fill:ink},majorGridlines:null},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
});
applyPresentationChartFont(chart,{fontFamily:font});

box('Alert engine · system level',860,594,680,38,25,true);
box('Separate from classifier-level rates',860,627,680,25,17,false,muted);
const av=source.alert_engine;
const avs=[
 ['Median lead time','−3.333 h'],
 ['P10 lead time','−11.667 h'],
 ['False alerts per normal event-day',av.false_alarms_per_normal_event_day.toFixed(3)+' episodes/day']
];
const at=slide.tables.add({rows:3,columns:2,left:860,top:658,width:680,height:143,columnWidths:[420,260],values:avs});
at.borders.assign({style:'solid',fill:line,width:0.7});
for(let r=0;r<3;r++){at.rows[r].height=47;for(let c=0;c<2;c++)at.getCell(r,c).fill=r%2===0?pale:'#FFFFFF'}
at.cells.block({row:0,column:0,rowCount:3,columnCount:2}).assign({textStyle:{typeface:font,fontSize:18,color:ink},margins:{left:11,right:10,top:8,bottom:6},anchor:'middle'});

rule(60,850,1480);
box('CALIBRATION  ·  Locked-test calibration: Not measured',60,856,735,22,16,true,muted);
box('Synthetic evaluation only. Useful early-warning performance and real-mine transfer remain unverified.',700,856,840,24,16,false,muted,'right');

slide.speakerNotes.textFrame.setText('Sources: reports/final_eval.json (locked synthetic test risk models and alert engine); reports/tuning/robustness.json (tuned XGBoost workstation CPU p50/p95 latency and sampled RSS). Other model profiling and locked-test calibration were not measured. Display values rounded only.');
const candidate=path.join(staging,'ml_dashboard_candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const png=await pres.export({slide,format:'png',scale:1});
await fs.writeFile(path.join(staging,'ml_dashboard_preview.png'),new Uint8Array(await png.arrayBuffer()));
const final=path.join(out,'ML_Performance_Dashboard_SIH_2026_v2.pptx');
const result=await finalizePresentation({workspaceDir:work,candidatePath:candidate,finalPath:final,
 pythonExecutable:'/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-heading-fit','--require-native-table-slide','1'],
 requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[1],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(staging,'ml_dashboard_validation_v2.json')});
console.log(JSON.stringify({final,preview:path.join(staging,'ml_dashboard_preview.png'),font,result},null,2));
