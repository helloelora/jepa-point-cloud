"""Build the standalone interactive demo (demo/index.html) and a body-only copy for
preview, from results/embed_drift.json. Data is slimmed (subsampled + rounded) and
inlined so the page is fully self-contained (no network, no external assets)."""
import json
import os

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "results", "embed_drift.json")
d = json.load(open(SRC))

BG_KEEP = 1400          # subsample background scatter
PTS_KEEP = 520          # subsample object point cloud


def thin(seq, k):
    n = len(seq)
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]


def r3(x):
    return round(float(x), 3)


slim = {"classes": d["classes"], "show": d["show"],
        "angles": [r3(a) for a in d["projector"]["angles"]],
        "axis": [r3(a) for a in d["projector"]["axis"]]}
for v in ("projector", "backbone", "random"):
    view = d[v]
    bidx = thin(view["bg"], BG_KEEP)
    bg = [[r3(view["bg"][i][0]), r3(view["bg"][i][1])] for i in bidx]
    by = [view["bg_y"][i] for i in bidx]
    objs = {}
    for name, o in view["objects"].items():
        pidx = thin(o["pts"], PTS_KEEP)
        objs[name] = {
            "pts": [[r3(o["pts"][i][0]), r3(o["pts"][i][1]), r3(o["pts"][i][2])] for i in pidx],
            "traj": [[r3(p[0]), r3(p[1])] for p in o["traj"]],
            "cos": [r3(c) for c in o["cos"]],
            "drift": r3(o["drift"]), "label": o["label"]}
    slim[v] = {"bg": bg, "bg_y": by, "var": r3(view["var"]), "objects": objs}

DATA_JSON = json.dumps(slim, separators=(",", ":"))

INNER = r"""<style>
:root{
  --ground:#E6EAEF; --paper:#F4F6F9; --void:#0E1722;
  --ink:#15212E; --muted:#6C7A89; --line:#C7D0DA;
  --accent:#F25C18; --accent-2:#2563A8; --pt:#AEB9C6;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
.demo-root{background:var(--ground);color:var(--ink);font-family:var(--sans);
  min-height:100%;padding:clamp(18px,4vw,48px);line-height:1.45;}
.wrap{max-width:1180px;margin:0 auto;}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--muted);margin:0 0 10px;}
.title{font-size:clamp(34px,6vw,62px);font-weight:800;letter-spacing:-.03em;margin:0;line-height:.98;}
.title em{font-style:normal;color:var(--accent);}
.sub{font-size:clamp(15px,2vw,19px);color:var(--ink);max-width:46ch;margin:12px 0 0;}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:26px 0 18px;}
.chip{font-family:var(--mono);font-size:12px;letter-spacing:.04em;padding:7px 13px;border:1px solid var(--line);
  background:var(--paper);color:var(--muted);border-radius:999px;cursor:pointer;transition:.15s;}
.chip:hover{border-color:var(--accent-2);color:var(--accent-2);}
.chip.on{background:var(--ink);color:#fff;border-color:var(--ink);}
.panels{display:grid;grid-template-columns:0.85fr 1fr 1fr;gap:14px;}
@media(max-width:860px){.panels{grid-template-columns:1fr;}}
.panel{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--paper);
  display:flex;flex-direction:column;position:relative;}
.panel.object{background:var(--void);border-color:#1d2c3c;}
.plabel{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  padding:13px 15px 0;color:var(--muted);display:flex;align-items:center;gap:8px;}
.panel.object .plabel{color:#7d93a8;}
.tag{font-size:10px;letter-spacing:.08em;padding:2px 7px;border-radius:5px;}
.tag.good{background:rgba(37,99,168,.13);color:var(--accent-2);}
.tag.bad{background:rgba(242,92,24,.13);color:var(--accent);}
canvas{display:block;width:100%;height:300px;touch-action:none;}
@media(max-width:860px){canvas{height:280px;}}
.objname{position:absolute;bottom:12px;left:15px;font-family:var(--mono);font-size:13px;
  color:#cbd6e2;letter-spacing:.05em;}
.readout{position:absolute;top:38px;right:15px;text-align:right;font-family:var(--mono);}
.readout b{display:block;font-size:30px;font-weight:600;letter-spacing:-.02em;line-height:1;}
.readout small{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.readout .drift{font-size:11px;color:var(--muted);margin-top:4px;}
.cos-hi{color:var(--accent-2);} .cos-lo{color:var(--accent);}
.seg{display:flex;gap:0;margin:auto 15px 14px;border:1px solid var(--line);border-radius:8px;overflow:hidden;width:max-content;}
.seg button{font-family:var(--mono);font-size:11px;letter-spacing:.05em;padding:6px 12px;border:0;background:var(--paper);
  color:var(--muted);cursor:pointer;}
.seg button.on{background:var(--ink);color:#fff;}
.controls{display:flex;align-items:center;gap:16px;margin:20px 0 0;padding:16px 18px;
  border:1px solid var(--line);border-radius:14px;background:var(--paper);}
.play{font-family:var(--mono);font-size:13px;padding:9px 16px;border:1px solid var(--ink);background:var(--ink);
  color:#fff;border-radius:9px;cursor:pointer;min-width:104px;}
.play:hover{background:#22323f;}
.angle{flex:1;accent-color:var(--accent);height:4px;cursor:pointer;}
.angleread{font-family:var(--mono);font-size:22px;min-width:74px;text-align:right;letter-spacing:-.01em;}
.foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:14px;align-items:center;margin-top:18px;}
.legend{font-family:var(--mono);font-size:11.5px;color:var(--muted);display:flex;align-items:center;gap:7px;flex-wrap:wrap;}
.legend .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-left:14px;}
.legend .dot:first-child{margin-left:0;}
.dot.bg{background:var(--pt);} .dot.cls{background:var(--accent-2);}
.dot.cur{background:var(--accent);box-shadow:0 0 0 2px rgba(242,92,24,.25);}
.note{font-family:var(--mono);font-size:11px;color:var(--muted);margin:0;}
.note a{color:var(--accent-2);text-decoration:none;}
.note a:hover{text-decoration:underline;}
button:focus-visible,input:focus-visible,.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
</style>

<div class="demo-root">
<div class="wrap">
  <div class="eyebrow">JEPA · ModelNet40 · self-supervised</div>
  <h1 class="title">Rotation <em>Invariance</em></h1>
  <p class="sub">Spin a shape and watch where its embedding lands. The projector keeps it
  pinned to its cluster; the backbone lets it drift across the map.</p>

  <div class="chips" id="chips"></div>

  <div class="panels">
    <section class="panel object">
      <div class="plabel">Object</div>
      <canvas id="objCanvas"></canvas>
      <div class="objname" id="objName">airplane</div>
    </section>
    <section class="panel plot">
      <div class="plabel">Projector <span class="tag good">invariant</span></div>
      <canvas id="projCanvas"></canvas>
      <div class="readout"><b id="projCos" class="cos-hi">1.000</b><small>cos(emb&#8320;, emb&#8348;)</small>
        <div class="drift" id="projDrift">drift 0.0</div></div>
    </section>
    <section class="panel plot">
      <div class="plabel"><span id="rightTitle">Backbone</span> <span class="tag bad" id="rightTag">pose-sensitive</span></div>
      <canvas id="rightCanvas"></canvas>
      <div class="readout"><b id="rightCos" class="cos-lo">0.94</b><small>cos(emb&#8320;, emb&#8348;)</small>
        <div class="drift" id="rightDrift">drift 0.0</div></div>
      <div class="seg" id="seg">
        <button data-v="backbone" class="on">Backbone</button>
        <button data-v="random">Untrained</button>
      </div>
    </section>
  </div>

  <div class="controls">
    <button id="play" class="play">&#9208; Pause</button>
    <input id="angle" class="angle" type="range" min="0" max="47" value="0" aria-label="rotation angle">
    <div class="angleread" id="angleVal">0&deg;</div>
  </div>

  <div class="foot">
    <div class="legend"><span class="dot bg"></span>all classes<span class="dot cls"></span>same class<span class="dot cur"></span>this shape, rotating</div>
    <p class="note">Fixed PCA(2) of the ModelNet40 test set &middot; <a href="https://github.com/helloelora/jepa-point-cloud">github.com/helloelora/jepa-point-cloud</a></p>
  </div>
</div>
</div>

<script>
const DATA = __DATA__;
const angles = DATA.angles, NA = angles.length, axis = DATA.axis;
const state = {obj: DATA.show[0], t: 0, right: "backbone", playing: true};

// ---- viridis-ish ramp for the object point cloud ----
const RAMP = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function ramp(u){u=Math.max(0,Math.min(1,u));const x=u*(RAMP.length-1),i=Math.floor(x),f=x-i;
  const a=RAMP[i],b=RAMP[Math.min(i+1,RAMP.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;}

function rot(ax,a){const[x,y,z]=ax,c=Math.cos(a),s=Math.sin(a),C=1-c;
  return [[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]];}
function mul(m,p){return [m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2],
  m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2], m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]];}
// fixed viewing rotation (elev 18, azim 35)
const EL=18*Math.PI/180, AZ=35*Math.PI/180;
const VIEW=(function(){const ry=[[Math.cos(AZ),0,Math.sin(AZ)],[0,1,0],[-Math.sin(AZ),0,Math.cos(AZ)]];
  const rx=[[1,0,0],[0,Math.cos(EL),-Math.sin(EL)],[0,Math.sin(EL),Math.cos(EL)]];
  return [0,1,2].map(i=>[0,1,2].map(j=>rx[i][0]*ry[0][j]+rx[i][1]*ry[1][j]+rx[i][2]*ry[2][j]));})();

function setup(c){const dpr=Math.min(window.devicePixelRatio||1,2);const r=c.getBoundingClientRect();
  c.width=r.width*dpr;c.height=r.height*dpr;const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);
  return {x,w:r.width,h:r.height};}

// ---- object canvas ----
const objC=document.getElementById("objCanvas");
function drawObject(){const{x,w,h}=setup(objC);x.clearRect(0,0,w,h);
  const pts=DATA.projector.objects[state.obj].pts, R=rot(axis,angles[state.t]);
  const sc=Math.min(w,h)*0.40, cx=w/2, cy=h/2;
  const proj=pts.map(p=>{const sp=mul(R,p);const v=mul(VIEW,sp);return {x:cx+v[0]*sc,y:cy-v[1]*sc,z:v[2],c:sp[1]};});
  proj.sort((a,b)=>a.z-b.z);
  for(const q of proj){const u=(q.c+1)/2, d=(q.z+1.4)/2.8;
    x.globalAlpha=0.35+0.6*d;x.fillStyle=ramp(u);
    x.beginPath();x.arc(q.x,q.y,1.4+1.7*d,0,7);x.fill();}
  x.globalAlpha=1;}

// ---- embedding canvas (cached background) ----
function viewBounds(v){let xs=v.bg.map(p=>p[0]),ys=v.bg.map(p=>p[1]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const px=(x1-x0)*0.09,py=(y1-y0)*0.09;return [x0-px,x1+px,y0-py,y1+py];}
function drawEmbed(canvas,vk){const{x,w,h}=setup(canvas);
  const v=DATA[vk], o=v.objects[state.obj], lab=o.label, [bx0,bx1,by0,by1]=viewBounds(v);
  const MX=Math.min(w,h)*0.06;
  const sx=p=>MX+(p[0]-bx0)/(bx1-bx0)*(w-2*MX);
  const sy=p=>h-MX-(p[1]-by0)/(by1-by0)*(h-2*MX);
  x.clearRect(0,0,w,h);
  // background scatter: all classes (muted) + same class (blue)
  for(let i=0;i<v.bg.length;i++){const p=v.bg[i],same=v.bg_y[i]===lab;
    x.fillStyle=same?"#2563A8":"#AEB9C6";x.globalAlpha=same?0.85:0.45;
    x.beginPath();x.arc(sx(p),sy(p),same?2.6:1.4,0,7);x.fill();}
  x.globalAlpha=1;
  // trail + current point
  const tr=o.traj;x.strokeStyle="#F25C18";x.lineWidth=1.6;x.globalAlpha=0.9;x.beginPath();
  for(let i=0;i<=state.t;i++){const p=[sx(tr[i]),sy(tr[i])];i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]);}x.stroke();
  const cp=tr[state.t];x.globalAlpha=0.22;x.fillStyle="#F25C18";x.beginPath();x.arc(sx(cp),sy(cp),11,0,7);x.fill();
  x.globalAlpha=1;x.fillStyle="#F25C18";x.strokeStyle="#15212E";x.lineWidth=1.2;
  x.beginPath();x.arc(sx(cp),sy(cp),5.5,0,7);x.fill();x.stroke();}

function fmtCos(c){return (c>=0?"":"-")+Math.abs(c).toFixed(3);}
function render(){
  drawObject();
  drawEmbed(document.getElementById("projCanvas"),"projector");
  drawEmbed(document.getElementById("rightCanvas"),state.right);
  const pc=DATA.projector.objects[state.obj], rc=DATA[state.right].objects[state.obj];
  document.getElementById("projCos").textContent=fmtCos(pc.cos[state.t]);
  document.getElementById("rightCos").textContent=fmtCos(rc.cos[state.t]);
  document.getElementById("projDrift").textContent="drift "+pc.drift.toFixed(1);
  document.getElementById("rightDrift").textContent="drift "+rc.drift.toFixed(1);
  document.getElementById("angleVal").textContent=Math.round(angles[state.t]*180/Math.PI)+"°";
  document.getElementById("angle").value=state.t;
}

// ---- controls ----
const chips=document.getElementById("chips");
DATA.show.forEach(name=>{const b=document.createElement("button");b.className="chip"+(name===state.obj?" on":"");
  b.textContent=name;b.onclick=()=>{state.obj=name;document.querySelectorAll(".chip").forEach(c=>c.classList.toggle("on",c===b));
    document.getElementById("objName").textContent=name;render();};chips.appendChild(b);});
document.getElementById("seg").onclick=e=>{const v=e.target.dataset.v;if(!v)return;state.right=v;
  document.querySelectorAll(".seg button").forEach(x=>x.classList.toggle("on",x.dataset.v===v));
  document.getElementById("rightTitle").textContent=v==="random"?"Untrained":"Backbone";
  document.getElementById("rightTag").textContent=v==="random"?"random init":"pose-sensitive";
  render();};
const playBtn=document.getElementById("play");
playBtn.onclick=()=>{state.playing=!state.playing;playBtn.innerHTML=state.playing?"&#9208; Pause":"&#9654; Spin";if(state.playing)loop();};
document.getElementById("angle").oninput=e=>{state.playing=false;playBtn.innerHTML="&#9654; Spin";state.t=+e.target.value;render();};

let last=0;
function loop(ts){if(!state.playing)return;if(!ts)ts=0;if(ts-last>70){state.t=(state.t+1)%NA;last=ts;render();}
  requestAnimationFrame(loop);}
const reduce=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let ro=new ResizeObserver(()=>{render();});
["objCanvas","projCanvas","rightCanvas"].forEach(id=>ro.observe(document.getElementById(id)));
if(reduce){state.playing=false;playBtn.innerHTML="&#9654; Spin";render();}else{render();requestAnimationFrame(loop);}
</script>
"""

inner = INNER.replace("__DATA__", DATA_JSON)

# body-only copy (for quick preview) + standalone page (repo / GitHub Pages)
open(os.path.join(HERE, "_body.html"), "w").write(inner)
standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
              '<title>Rotation Invariance — JEPA Point Cloud</title>\n'
              '<style>html,body{margin:0;height:100%}</style>\n</head>\n<body>\n'
              + inner + '\n</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w").write(standalone)
print("wrote demo/index.html (%d KB) and demo/_body.html" % (len(standalone) // 1024))
