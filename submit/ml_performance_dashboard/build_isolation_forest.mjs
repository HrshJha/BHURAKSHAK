import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const skill='/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const work='/Users/harshkumarjha/mine';
const out=path.join(work,'submit/ml_performance_dashboard');
const staging=path.join(work,'.codex-finalizer');
await fs.mkdir(out,{recursive:true}); await fs.mkdir(staging,{recursive:true});
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const j=JSON.parse(await fs.readFile(path.join(work,'reports/final_eval.json'),'utf8'));
const d=j.anomaly_models.default,t=j.anomaly_models.tuned;
const pct=v=>(v*100).toFixed(2)+'%';
const rnd=v=>Number(v.toFixed(6));
const pres=Presentation.create({slideSize:{width:1600,height:900}});
const slide=pres.slides.add();slide.background.fill='#FFFFFF';
const ink='#171717',muted='#4C4C4C',line='#C9C9C9',pale='#F5F5F5';
function txt(value,x,y,w,h,size=20,bold=false,color=ink){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return s;
}
function rule(x,y,w){slide.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{style:'solid',fill:line,width:1}})}

txt('ISOLATION FOREST PERFORMANCE',60,38,1480,58,40,true);
txt('BHURAKSHAK · locked synthetic test',60,95,1480,26,18,false,muted);
rule(60,130,1480);

txt('TUNED MODEL · DETECTION',60,148,550,24,18,true,muted);
txt(pct(t.recall_at_development_healthy_threshold),60,172,225,56,42,true);
txt('Recall at development threshold',285,190,480,29,22);
txt('TUNED MODEL · FALSE POSITIVES',810,148,680,24,18,true,muted);
txt(pct(t.false_positive_rate_normal_anomaly_label),810,172,225,56,42,true);
txt('False-positive rate',1035,190,480,29,22);
rule(60,235,1480);

txt('Measured comparison',60,252,1480,39,28,true);
txt('Anomaly detection · default and tuned Isolation Forest',60,290,1480,25,17,false,muted);
txt('CLASSIFICATION',325,326,690,18,15,true,muted);
txt('OPERATIONAL',1015,326,525,18,15,true,muted);

const vals=[
 ['Model','Anomaly PR-AUC','Recall at development threshold','False-positive rate'],
 ['Default Isolation Forest',pct(d.pr_auc_anomaly),pct(d.recall_at_development_healthy_threshold),pct(d.false_positive_rate_normal_anomaly_label)],
 ['Tuned Isolation Forest',pct(t.pr_auc_anomaly),pct(t.recall_at_development_healthy_threshold),pct(t.false_positive_rate_normal_anomaly_label)]
];
const table=slide.tables.add({rows:3,columns:4,left:60,top:350,width:1480,height:194,columnWidths:[265,310,440,465],values:vals});
table.borders.assign({style:'solid',fill:line,width:0.7});
for(let r=0;r<3;r++){table.rows[r].height=r===0?58:68;for(let c=0;c<4;c++)table.getCell(r,c).fill=r===0?'#ECECEC':r===2?pale:'#FFFFFF'}
table.cells.block({row:0,column:0,rowCount:3,columnCount:4}).assign({textStyle:{typeface:font,fontSize:21,color:ink},margins:{left:12,right:10,top:10,bottom:8},anchor:'middle'});
table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:font,fontSize:19,color:ink,bold:true}});
table.cells.block({row:1,column:0,rowCount:2,columnCount:1}).assign({textStyle:{typeface:font,fontSize:21,color:ink,bold:true}});

txt('PR-AUC and recall',60,578,900,37,26,true);
txt('Percent · PR-AUC and recall use the same locked synthetic test',60,611,900,24,16,false,muted);
const chart1=slide.charts.add('bar',{
 position:{left:60,top:646,width:900,height:185},categories:['Default Isolation Forest','Tuned Isolation Forest'],
 series:[{name:'Anomaly PR-AUC',values:[rnd(d.pr_auc_anomaly),rnd(t.pr_auc_anomaly)],fill:'#222222',valuesFormatCode:'0.00%'},{name:'Recall',values:[rnd(d.recall_at_development_healthy_threshold),rnd(t.recall_at_development_healthy_threshold)],fill:'#9A9A9A',valuesFormatCode:'0.00%'}],
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:70},hasLegend:true,
 legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:16,fill:ink}},
 xAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0%',textStyle:{typeface:font,fontSize:15,fill:muted},majorGridlines:{style:'solid',fill:'#E2E2E2',width:.7}},
 yAxis:{textStyle:{typeface:font,fontSize:16,fill:ink},majorGridlines:null},
 chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
});applyPresentationChartFont(chart1,{fontFamily:font});

txt('False-positive rate',1000,578,540,37,26,true);
txt('Separate scale · lower is better',1000,611,540,24,16,false,muted);
const chart2=slide.charts.add('bar',{
 position:{left:1000,top:646,width:540,height:185},categories:['Default','Tuned'],
 series:[{name:'False-positive rate',values:[rnd(d.false_positive_rate_normal_anomaly_label),rnd(t.false_positive_rate_normal_anomaly_label)],fill:'#777777',valuesFormatCode:'0.00%'}],
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:95},hasLegend:false,
 xAxis:{min:0,max:.03,majorUnit:.01,numberFormatCode:'0%',textStyle:{typeface:font,fontSize:15,fill:muted},majorGridlines:{style:'solid',fill:'#E2E2E2',width:.7}},
 yAxis:{textStyle:{typeface:font,fontSize:16,fill:ink},majorGridlines:null},
 chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
});applyPresentationChartFont(chart2,{fontFamily:font});

rule(60,848,1480);
txt('Synthetic evaluation only. Recall uses a development threshold; real-mine performance remains unverified.',60,855,1480,25,17,false,muted);
slide.speakerNotes.textFrame.setText('Source: reports/final_eval.json, anomaly_models.default and anomaly_models.tuned, locked synthetic test. The user-provided bhurakshak_isolation_forest_performance.png supplies the original displayed values and caveat. Percentages are rounded to two decimal places. Recall uses the development healthy threshold; this is not a newly optimized locked-test threshold.');

const candidate=path.join(staging,'isolation_forest_candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const png=await pres.export({slide,format:'png',scale:1});
await fs.writeFile(path.join(out,'Isolation_Forest_Performance_SIH_2026_preview.png'),new Uint8Array(await png.arrayBuffer()));
const final=path.join(out,'Isolation_Forest_Performance_SIH_2026_v2.pptx');
const result=await finalizePresentation({workspaceDir:work,candidatePath:candidate,finalPath:final,
 pythonExecutable:'/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-heading-fit','--require-native-table-slide','1'],
 requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(staging,'isolation_forest_validation_v2.json')});
console.log(JSON.stringify({final,preview:path.join(out,'Isolation_Forest_Performance_SIH_2026_preview.png'),passed:result.packageIntegrity.status,chartCount:result.nativeChartValidation.detectedChartCount},null,2));
