'use strict';
const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
const form=$('#recipe'), chart=$('#chart'), plot=chart.parentElement;
const format=(v,n=2)=>v==null?'—':v.toLocaleString('cs-CZ',{maximumFractionDigits:n,minimumFractionDigits:n});
const percent=n=>n==null?'—':format(100*n)+' %';
const time=t=>new Date(t*1000).toLocaleString('cs-CZ',{timeZone:'UTC',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
const palette={price:'#d5e8ed',forecast:'#63e3c1',band:'#55948f',equity:'#7ce2c1'};
let state=null, selected='', desired='', report=null, source='mt5', mode='price', cursor=-1, jobSignature='',loading='',pinned=false;
function msg(message){const el=$('#notice');el.hidden=!message;el.textContent=message||'';}
async function api(path,params,body){
    if(params && Object.prototype.hasOwnProperty.call(params,'graph')){
        params={...params};body=params.engine==='tdnn'?'':params.graph;delete params.graph;
    }
    const opt=params?{method:'POST',headers:{'X-DNForecast-Key':token},body:body||''}:{};
    const r=await fetch(path+(params?'?'+new URLSearchParams(params):''),opt);
    if(!r.ok)throw Error(await r.text());return r;
}
function view(id){
    $$('.view').forEach(v=>v.hidden=v.id!==id);
    $$('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===id));
    if(id==='lab')draw();if(id==='automation')draft();
}
$$('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
function pickSource(value){
    source=value;$('#broker').hidden=value!=='mt5';$('#files').hidden=value!=='file';
    $$('[data-source]').forEach(b=>b.classList.toggle('active',b.dataset.source===value));hint();
}
$$('[data-source]').forEach(b=>b.onclick=()=>pickSource(b.dataset.source));
function params(){return Object.fromEntries(new FormData(form));}
function applyRecipe(p){
    form.elements.engine.value=p.engine||'tdnn';
    Object.entries(p).forEach(([key,value])=>{if(form.elements[key]&&value!=null)form.elements[key].value=value;});
    NetworkEditor.refresh();
}
try{applyRecipe(JSON.parse(localStorage.getItem('studio.recipe')||'{}'));}catch{}
form.addEventListener('change',()=>{try{localStorage.setItem('studio.recipe',JSON.stringify(params()));}catch{}});
function draft(){const p=params();$('#draft').textContent=p.engine.toUpperCase()+' · '+p.target+' · '+form.elements.seconds.selectedOptions[0].text+' · '+p.bars+' svíček · '+p.epochs+' epoch';}
function hint(){
    if(!state)return;
    $('#hint').textContent=source==='file'?'Vstup v CSV musí obsahovat uzavřené svíčky ve správném pořadí.':
        state.feed.online?'Připojeno k '+state.feed.broker+'. Historii načte terminál automaticky.':
        'V Navigátoru MT5 otevřete Strategie a přetáhněte DNForecast_Bridge na jeden graf přihlášeného terminálu.';
}
function fillOptions(el,values){
    if(JSON.stringify(Array.from(el.children,o=>o.value))===JSON.stringify(values))return;
    const old=el.value;el.replaceChildren(...values.map(v=>new Option(v,v)));
    if(values.includes(old))el.value=old;
}
$('#upload').onchange=async e=>{
    try {const f=e.target.files[0];if(!f)return;
        const name=await (await api('/api/import',{name:f.name},f)).text();
        await refresh();$('#dataset').value=name;msg('Načteno: '+name);
    }catch(err){msg(err.message);}finally{e.target.value='';}
};
form.onsubmit=async e=>{
    e.preventDefault();
    try{
        const p=params(),csv=source==='file';
        if(csv&&!$('#dataset').value)throw Error('Nejdříve vyberte nebo importujte historii.');
        if(csv)p.file=$('#dataset').value;
        $('#train').disabled=true;pinned=false;desired=await(await api(csv?'/api/train':'/api/run',p)).text();
        msg('Úloha byla přijata.');await refresh();
    }catch(err){msg(err.message);}finally{if(state)$('#train').disabled=state.busy;}
};
$('#schedule').onclick=async()=>{
    if(!form.reportValidity()){view('lab');return;}
    try{await api('/api/plan',{...params(),minutes:$('#interval').value});msg('Plán byl uložen.');await refresh();}
    catch(err){msg(err.message);}
};
$('#stop').onclick=async()=>{try{await api('/api/stop-plan',{});msg('Plán zastaven.');await refresh();}catch(err){msg(err.message);}};
function jobs(list){
    const key=JSON.stringify(list.map(j=>[j.id,j.status,j.log]));
    if(key===jobSignature)return;jobSignature=key;
    const el=$('#jobs');el.replaceChildren();
    if(!list.length){el.textContent='Zatím nebyla spuštěna žádná úloha.';return;}
    list.forEach(j=>{
        const row=document.createElement('div'),info=document.createElement('div'),title=document.createElement('b'),desc=document.createElement('span');
        row.className='job';title.textContent=j.contract;desc.textContent=[j.status,j.created?new Date(j.created).toLocaleString('cs-CZ'):'',j.status==='Chyba'?j.log:''].filter(Boolean).join(' · ');
        info.append(title,desc);row.append(info);
        if(j.model){const b=document.createElement('button');b.textContent='Otevřít';b.onclick=()=>{desired='';pinned=true;openJob(j);view('lab');};row.append(b);}
        el.append(row);
    });
}
function rows(body,data){
    body.replaceChildren(...data.map(vals=>{const tr=document.createElement('tr');
        vals.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});return tr;}));
}
function metric(id,v,color){const el=$(id);el.textContent=percent(v);el.className=color&&v!=null?(v<0?'negative':v>0?'positive':''):'';}
async function openJob(j){
    selected=j.id;loading=j.id;report=null;cursor=-1;
    $('#market-title').textContent=j.contract.split(' | ')[0];
    $('#contract').textContent=j.contract;
    $('#model').hidden=false;$('#model').href='/model/'+j.id;$('#model').download=j.model;
    $('#settings').hidden=!j.config;
    if(j.config){
        const c=j.config;
        $('#recipe-text').style.whiteSpace='pre-wrap';
        $('#recipe-text').textContent=[j.device||'CPU',c.epochs+' epoch',({both:'long i short',long:'long',short:'short'})[c.side],
            'práh '+c.edge+' bp','náklad '+c.cost+' bp na stranu','pozice '+c.allocation+' %'].join(' · ')+(c.graph?'\n'+c.graph:'');
        $('#reuse').onclick=()=>{applyRecipe(c);pickSource(c.target?'mt5':'file');if(!c.target)$('#dataset').value=j.dataset;msg('Parametry načteny do konfigurace.');};
    }
    const scores=(j.log||'').split(/\r?\n/).slice(1).filter(Boolean).map(line=>{
        const [h,m,base,c]=line.split(',').map(Number);return [h,percent(base>0?1-m/base:null),percent(c)];
    });
    rows($('#scores'),scores);
    try{if(j.hasReport){
        const data=await(await api('/report/'+j.id)).json();if(selected!==j.id)return;report=data;
        $('#contract').textContent=[j.broker,j.contract,data.points.length+' testovacích svíček'].filter(Boolean).join(' · ');
    }}catch(e){if(selected===j.id)msg(e.message);}
    if(selected!==j.id)return;loading='';
    metric('#net',report&&report.net,true);metric('#dd',report&&-report.drawdown,false);metric('#win',report&&report.winRate,false);
    $('#count').textContent=report?report.trades.length+' obchodů · expozice '+percent(report.exposure):'uzavřené pozice';
    rows($('#trades'),report?report.trades.slice(-100).reverse().map(t=>[time(t.entry),t.side,percent(t.gain)]):[]);
    if(report&&!report.trades.length)rows($('#trades'),[['Žádný obchod','—','—']]);draw();
}
async function refresh(){
    const s=await(await api('/api/state')).json();state=s;
    NetworkEditor.hardware(s.cuda,s.cudaMessage);
    $('#lamp').classList.toggle('on',s.feed.online);
    $('#connection').textContent=s.feed.online?s.feed.broker+' · MT5 připojeno':'MT5 odpojeno';
    fillOptions($('#symbols'),s.feed.symbols);fillOptions($('#dataset'),s.datasets);hint();jobs(s.jobs);
    $('#train').disabled=s.busy;$('#train').firstChild.textContent=s.busy?'Úloha běží ':'Spustit experiment ';
    const latest=s.jobs[0],wanted=s.jobs.find(j=>j.id===desired);
    if(s.busy&&latest)msg(latest.status+' · '+latest.contract);
    else if(wanted&&wanted.model){desired='';msg('Experiment dokončen.');await openJob(wanted);}
    else if(wanted&&wanted.status==='Chyba'){desired='';msg('Úloha selhala: '+wanted.log);}
    else if(!pinned&&!loading){const j=s.jobs.find(x=>x.model);if(j&&selected!==j.id)await openJob(j);}
    if(!s.busy && !desired && latest && latest.status==='Dokončeno' && $('#notice').textContent.endsWith(latest.contract))msg('Experiment dokončen.');
    $('#plan-title').textContent=s.plan?s.plan.recipe.target+' · '+s.plan.recipe.minutes+' min':'Vypnuto';
    $('#plan-text').textContent=s.plan?s.plan.message+' · Další kontrola '+new Date(s.plan.next).toLocaleString('cs-CZ'):'Zvolte parametry v experimentu a poté aktivujte plán.';
    $('#stop').hidden=!s.plan;
}
function legend(){
    const items=mode==='price'?[['Skutečná cena',palette.price],['Predikce na horizontu',palette.forecast],['Interval Q10–Q90',palette.band]]:[['Kapitál po nákladech · počátek 100',palette.equity]];
    const el=$('#legend');el.replaceChildren(...items.map(([s,c])=>{const span=document.createElement('span');span.style.setProperty('--swatch',c);span.textContent=s;return span;}));
}
function draw(){
    legend();if(!plot.clientWidth)return;
    const w=plot.clientWidth,h=plot.clientHeight,ratio=window.devicePixelRatio||1;
    chart.width=w*ratio;chart.height=h*ratio;const g=chart.getContext('2d');g.scale(ratio,ratio);
    const left=9,right=w-74,top=15,bottom=h-23,pts=report?report.points:[];
    const field=mode==='price'?1:5,scale=mode==='price'?1:100;
    let lo=Infinity,hi=-Infinity;
    for(const p of pts)for(const k of mode==='price'?[1,2,3,4]:[5])if(p[k]!=null){lo=Math.min(lo,p[k]*scale);hi=Math.max(hi,p[k]*scale);}
    const pad=Number.isFinite(lo)?Math.max((hi-lo)*.08,Math.abs(hi)*.0001,.000001):1;
    lo-=pad;hi+=pad;
    const X=i=>left+(right-left)*i/Math.max(1,pts.length-1),Y=v=>top+(bottom-top)*(hi-v)/(hi-lo);
    g.font='11px Consolas, monospace';g.fillStyle='#a7bdc7';g.lineWidth=1;
    for(let i=0;i<=4;i++){const y=top+(bottom-top)*i/4;g.strokeStyle='#294959';g.beginPath();g.moveTo(left,y);g.lineTo(right,y);g.stroke();
        if(pts.length)g.fillText(format(hi-i*(hi-lo)/4,hi>1000?0:hi<10?4:2),right+7,y+4);}
    $('#empty').hidden=!!pts.length;$('#tooltip').hidden=true;if(!pts.length)return;
    const series=(k,c)=>{g.beginPath();let move=true;pts.forEach((p,i)=>{if(p[k]==null){move=true;return;}const a=X(i),b=Y(p[k]*scale);if(move)g.moveTo(a,b);else g.lineTo(a,b);move=false;});
        g.strokeStyle=c;g.lineWidth=1.6;g.stroke();};
    if(mode==='price'){
        const first=pts.findIndex(p=>p[3]!=null);
        if(first>=0){g.beginPath();g.moveTo(X(first),Y(pts[first][4]));for(let i=first+1;i<pts.length;i++)g.lineTo(X(i),Y(pts[i][4]));
            for(let i=pts.length-1;i>=first;i--)g.lineTo(X(i),Y(pts[i][3]));g.closePath();g.fillStyle='#59bcab38';g.fill();}
        series(1,palette.price);series(2,palette.forecast);
    }else series(5,palette.equity);
    g.fillStyle='#a7bdc7';g.fillText(time(pts[0][0]),left,h-4);g.textAlign='right';g.fillText(time(pts[pts.length-1][0]),right,h-4);g.textAlign='left';
    if(cursor>=0&&cursor<pts.length){const p=pts[cursor],x=X(cursor);g.strokeStyle='#aecbd0';g.setLineDash([3,3]);g.beginPath();g.moveTo(x,top);g.lineTo(x,bottom);g.stroke();
        $('#tooltip').hidden=false;$('#tooltip').textContent=time(p[0])+' · '+format(p[field]*scale,4)+(mode==='price'&&p[2]!=null?' · NN '+format(p[2],4):'');}
}
$$('[data-plot]').forEach(b=>b.onclick=()=>{mode=b.dataset.plot;cursor=-1;$$('[data-plot]').forEach(x=>x.classList.toggle('active',x===b));$('#chart-title').textContent=mode==='price'?'Cena a predikce':'Vývoj kapitálu';draw();});
plot.onpointermove=e=>{if(!report)return;cursor=Math.max(0,Math.min(report.points.length-1,Math.round((e.clientX-plot.getBoundingClientRect().left-9)/(plot.clientWidth-83)*(report.points.length-1))));draw();};
plot.onpointerleave=()=>{cursor=-1;draw();};
plot.onkeydown=e=>{if(!report||!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();cursor=Math.max(0,Math.min(report.points.length-1,cursor+(e.key==='ArrowLeft'?-1:1)));draw();};
new ResizeObserver(draw).observe(plot);draw();
(async function poll(){try{await refresh();}catch(e){msg('Spojení se Studiem: '+e.message);}setTimeout(poll,2200);})();
