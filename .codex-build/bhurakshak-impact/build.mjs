import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = '/Users/harshkumarjha/mine';
const skillDir = '/Users/harshkumarjha/.codex/plugins/cache/openai-primary-runtime/presentations/26.927.11222/skills/presentations';
const runtimeModules = '/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtimeModules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont();
const buildDir = path.join(root, '.codex-build/bhurakshak-impact');
const outDir = path.join(root, 'submit/bhurakshak_impact');
const candidate = path.join(buildDir, 'candidate.pptx');
const finalPath = path.join(outDir, 'BHURAKSHAK_Impact_Pathway_v3.pptx');
const previewPath = path.join(outDir, 'BHURAKSHAK_Impact_Pathway_v3_preview.png');
const W=1920,H=1080;
const C={navy:'#173746', green:'#247A48', blue:'#3B83A7', text:'#1E3039', muted:'#56707C', pale:'#F5F8F7', line:'#D9E4E0', lightGreen:'#EAF4EE', lightBlue:'#EDF4F8', amber:'#B87922', white:'#FFFFFF', gray:'#7A8B92'};
const p=Presentation.create({slideSize:{width:W,height:H}}); const s=p.slides.add(); s.background.fill=C.white;
function shape(geometry,x,y,w,h,fill='none',stroke='none',sw=0,r=0){return s.shapes.add({geometry,position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:stroke,width:sw},...(r?{borderRadius:r}:{})});}
function txt(text,x,y,w,h,size,color=C.text,bold=false,opts={}){let q=shape('textbox',x,y,w,h,'none','none',0);q.text=text;q.text.style={typeface:font,fontSize:size,bold,color,alignment:opts.align??'left',verticalAlignment:opts.valign??'middle',autoFit:'shrinkText',wrap:true};q.text.insets={left:0,right:0,top:0,bottom:0};return q;}
function line(x1,y1,x2,y2,color,width=3){
 const left=Math.min(x1,x2),top=Math.min(y1,y2),w=Math.max(Math.abs(x2-x1),0.1),h=Math.max(Math.abs(y2-y1),0.1);
 s.shapes.add({geometry:'line',position:{left,top,width:w,height:h,verticalFlip:(x2-x1)*(y2-y1)<0},fill:'none',line:{style:'solid',fill:color,width}});
}
function circ(x,y,d,fill,stroke='none',sw=0){return shape('ellipse',x,y,d,d,fill,stroke,sw);}
// Header
shape('rect',56,56,8,108,C.green,'none',0);
txt('BHURAKSHAK',88,53,640,56,24,C.green,true);
txt('Mine subsidence monitoring · intended impact pathway',88,106,1300,72,40,C.navy,true);
shape('roundRect',1460,69,400,54,C.lightBlue,'none',0,20);
txt('DESIGNED SYSTEM · NOT A FIELD RESULT',1480,71,360,50,17,C.blue,true,{align:'center'});
txt('Sensor signals move through anomaly detection and risk assessment to human review. The impact is potential until physically and operationally validated.',88,183,1744,47,21,C.muted,false);
// process cards: connected flow
const xs=[56,360,664,968,1272,1576], y=306, cw=268,ch=354;
const data=[
 {n:'01',head:'SENSOR MONITORING',color:C.green,fill:'#F8FBF9',body:'Designed for continuous tilt, vibration, crack-change and displacement readings.'},
 {n:'02',head:'DEFORMATION DETECTION',color:C.blue,fill:C.lightBlue,body:'Isolation Forest flags unusual patterns in the sensor readings.'},
 {n:'03',head:'RISK ASSESSMENT',color:C.blue,fill:'#EAF3F7',body:'XGBoost assigns a risk class; physics checks screen for implausible signals.'},
 {n:'04',head:'EARLY WARNING',color:C.amber,fill:'#FBF7EF',body:'GREEN → WATCH →\nWARNING → CRITICAL\nGuide attention; not a\ncollapse-time prediction.'},
 {n:'05',head:'HUMAN INTERVENTION',color:C.green,fill:'#F4F9F5',body:'Operators and authorities can prioritize inspections and safety measures.'},
 {n:'06',head:'POTENTIAL IMPACT',color:C.green,fill:'#F1F7F3',body:'Earlier information could help reduce exposure and limit damage.'},
];
const cards=[];
for(let i=0;i<data.length;i++){
 const d=data[i],x=xs[i];
 const card=shape('roundRect',x,y,cw,ch,d.fill,C.line,2,14);cards.push(card);
 shape('rect',x,y,6,ch,d.color,'none',0);
 // Number label
 circ(x+22,y+22,40,d.color);
 txt(d.n,x+22,y+22,40,40,15,C.white,true,{align:'center'});
 // Icon zone, separate line icons created with simple editable shapes.
 const ix=x+95, iy=y+38;
 if(i===0){ // sensor: chip + leads
  shape('roundRect',ix,iy,54,42,'none',d.color,3,5); shape('ellipse',ix+20,iy+10,14,14,'none',d.color,3);
  line(ix+10,iy-6,ix+10,iy);line(ix+44,iy-6,ix+44,iy);line(ix+10,iy+42,ix+10,iy+49);line(ix+44,iy+42,ix+44,iy+49);line(ix-6,iy+10,ix,iy+10);line(ix-6,iy+32,ix,iy+32);line(ix+54,iy+10,ix+60,iy+10);line(ix+54,iy+32,ix+60,iy+32);
 } else if(i===1){ // magnifier with signal trace
  circ(ix+4,iy+1,36,'none',d.color,3); line(ix+33,iy+31,ix+52,iy+50,d.color,4);line(ix+11,iy+21,ix+18,iy+21,d.color,2);line(ix+18,iy+21,ix+22,iy+13,d.color,2);line(ix+22,iy+13,ix+28,iy+25,d.color,2);
 } else if(i===2){ // connected model nodes
  circ(ix+2,iy+6,12,C.white,d.color,2);circ(ix+2,iy+34,12,C.white,d.color,2);circ(ix+31,iy+20,14,C.white,d.color,2);circ(ix+58,iy+20,14,d.color);
  line(ix+14,iy+12,ix+31,iy+23,d.color,2);line(ix+14,iy+40,ix+31,iy+28,d.color,2);line(ix+45,iy+27,ix+58,iy+27,d.color,2);
 } else if(i===3){ // warning triangle
  shape('triangle',ix+3,iy+1,56,50,'none',d.color,3);shape('rect',ix+29,iy+15,4,17,d.color,'none',0,2);circ(ix+29,iy+36,5,d.color);
 } else if(i===4){ // two people
  circ(ix+8,iy+1,18,'none',d.color,3);circ(ix+37,iy+7,15,'none',d.color,3);shape('arc',ix+2,iy+24,31,25,'none',d.color,3);shape('arc',ix+32,iy+27,29,22,'none',d.color,3);
 } else { // small building / shield-like impact
  shape('rect',ix+7,iy+15,48,36,'none',d.color,3);shape('triangle',ix+7,iy+1,48,31,'none',d.color,3);shape('rect',ix+26,iy+31,12,20,'none',d.color,2);shape('rect',ix+13,iy+25,8,8,d.color,'none',0);shape('rect',ix+43,iy+25,8,8,d.color,'none',0);
 }
 txt(d.head,x+18,y+110,cw-36,54,19,d.color,true,{align:'center'});
 line(x+24,y+171,x+cw-24,y+171,C.line,1.5);
 txt(d.body,x+23,y+192,cw-46,139,i===3?19:20,C.text,false,{align:'center',valign:'top'});
}
// arrows between cards, unobstructed centerline
for(let i=0;i<5;i++){
 const x=xs[i]+cw+5, end=xs[i+1]-8, cy=y+ch/2;
 line(x,cy,end-9,cy,C.gray,2.5);
 shape('triangle',end-12,cy-6,14,12,C.gray,'none',0);
}
// AI pipeline bracket/label above the two relevant cards
shape('roundRect',559,244,386,56,C.lightBlue,C.blue,1.5,18);
txt('AI PIPELINE\nIsolation Forest + XGBoost + physics checks',571,246,362,52,14,C.blue,true,{align:'center'});
// Impact outcomes
line(56,713,1864,713,C.line,2);
txt('POTENTIAL OUTCOMES',56,739,470,44,19,C.green,true);
txt('Possible benefits to evaluate in physical and field studies',534,739,1200,44,19,C.muted,false);
const outcomes=[['WORKER SAFETY','Worker exposure'],['FASTER DETECTION','Time to notice'],['TARGETED INSPECTION','Inspection priority'],['ENVIRONMENTAL PROTECTION','Land and water']];
const ox=[56,516,976,1436];
for(let i=0;i<4;i++){
 let x=ox[i]; shape('roundRect',x,808,428,102,'#FFFFFF',C.line,1.5,10);
 // small green vertical accent and simple status dot
 shape('rect',x,808,5,102,i===1?C.blue:C.green,'none',0);
 circ(x+24,843,24,i===1?C.lightBlue:C.lightGreen,'none',0);
 // small centered directional arrow glyph using text (not decorative)
 txt('→',x+24,839,24,30,17,i===1?C.blue:C.green,true,{align:'center'});
 txt(outcomes[i][0],x+65,823,337,30,17,C.navy,true);
 txt(outcomes[i][1],x+65,859,337,28,16,C.muted,false);
}
// Evidence boundary and exact required caveat
shape('roundRect',56,947,1808,57,'#F3F7F6','none',0,8);
txt('Evidence to date: held-out synthetic evaluation only; physical and mine-site transfer remain unverified.',78,951,1760,48,17,C.navy,true);
txt('Potential impact; real-world effectiveness awaits physical and field validation.',56,1020,1808,32,15,C.muted,false,{align:'center'});
s.speakerNotes.textFrame.setText('Impact chain is a proposed system pathway. Current software evaluation is on a held-out synthetic corpus only. No physical or mine-site validation is established. Alert levels are intended risk categories and do not predict exact collapse time.');
await fs.mkdir(buildDir,{recursive:true}); await fs.mkdir(outDir,{recursive:true});
await (await PresentationFile.exportPptx(p)).save(candidate);
const png=await p.export({slide:s,format:'png',scale:1}); await fs.writeFile(path.join(buildDir,'slide.png'),new Uint8Array(await png.arrayBuffer()));
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:'/Users/harshkumarjha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','18288000,10287000'],requiredNativeChartOwnerSlides:[],materializeLiteralChartWorkbooks:false,explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation-v3.json')});
await fs.copyFile(path.join(buildDir,'slide.png'),previewPath);
console.log(JSON.stringify({finalPath,previewPath,font,result},null,2));
