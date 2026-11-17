'use strict';
const NetworkEditor=(()=>{
    const el=id=>document.getElementById(id);
    const engine=el('engine'),spec=el('graph-spec'),dialog=el('graph-dialog'),list=el('layer-list');
    const kinds=['LSTM','GRU','RNN','Dense','Identity'],activations=['linear','tanh','relu','sigmoid'];
    const presets={
        stack:'LSTM 32 linear 0 concat -1\nGRU 16 linear 0 concat 0\nDense 16 relu 0 concat 1',
        residual:'Dense 24 tanh 0 concat -1\nGRU 24 linear 0.1 concat 0\nIdentity 0 linear 0 sum 0 1\nDense 16 tanh 0 concat 2',
        parallel:'LSTM 24 linear 0 concat -1\nGRU 16 linear 0 concat -1\nDense 16 relu 0 concat 0 1'
    };
    let nodes=[],gpu='',reason='Ověřuji zařízení…',rawChanged=false;
    function parse(text){
        const lines=text.trim().split(/\r?\n/).filter(line=>line.trim());
        if(!lines.length||lines.length>1024)throw Error('Zvolte 1 až 1 024 vrstev.');
        return lines.map((line,i)=>{
            const [kind,width,act,drop,merge,...parents]=line.trim().split(/\s+/);
            if(!kinds.includes(kind)||!/^\d+$/.test(width)||!activations.includes(act)||!drop||!Number.isFinite(+drop)||+drop<0||+drop>=1||
                !['sum','concat'].includes(merge)||!parents.length||parents.length>64||parents.some(p=>!/^(-1|\d+)$/.test(p)))
                throw Error('Neplatný zápis vrstvy '+(i+1)+'.');
            return {kind,width:+width,act,drop:+drop,merge,parents:parents.map(p=>+p+1).join(', ')};
        });
    }
    function serialize(){
        return nodes.map((n,i)=>{
            const inputs=n.parents.trim().split(/[\s,]+/);
            if(!n.parents.trim() || inputs.some(p=>!/^\d+$/.test(p)||+p>i))throw Error('Zkontrolujte vstupy vrstvy '+(i+1)+'.');
            if(!Number.isInteger(n.width)||n.width<(n.kind==='Identity'?0:1)||!Number.isFinite(n.drop)||n.drop<0||n.drop>=1)
                throw Error('Zkontrolujte šířku a dropout vrstvy '+(i+1)+'.');
            return [n.kind,n.width,n.kind==='Dense'?n.act:'linear',n.drop,n.merge,...inputs.map(p=>+p-1)].join(' ');
        }).join('\n');
    }
    function fields(n,i){
        const row=document.createElement('section');row.className='graph-row';
        const heading=document.createElement('div');heading.className='node-heading';row.append(heading);
        const badge=document.createElement('b');badge.textContent='Vrstva '+(i+1);heading.append(badge);
        const remove=document.createElement('button');remove.type='button';remove.textContent='Odebrat';remove.onclick=()=>{
            nodes.splice(i,1);nodes.forEach(node=>{node.parents=node.parents.split(/[\s,]+/).filter(p=>p!==''&&+p!==i+1).map(p=>+p>i+1?+p-1:p).join(', ');});render();
        };heading.append(remove);
        const controls=document.createElement('div');controls.className='node-fields';row.append(controls);
        function field(title,key,options){
            const label=document.createElement('label');label.textContent=title;controls.append(label);
            const input=document.createElement(options?'select':'input');label.append(input);
            if(options)options.forEach(v=>input.add(new Option(v,v)));
            else if(key==='width'||key==='drop'){input.type='number';input.min='0';input.step=key==='width'?'1':'any';}
            input.value=n[key];input.disabled=key==='act'&&n.kind!=='Dense';
            input.oninput=()=>{
                n[key]=(key==='width'||key==='drop')?(input.value===''?NaN:+input.value):input.value;
                if(key==='kind'){if(n.kind!=='Dense')n.act='linear';if(n.kind==='Identity')n.width=0;else if(!n.width)n.width=16;render();}
                else preview();
            };
        }
        field('Operace','kind',kinds);field('Šířka','width');field('Aktivace','act',activations);
        field('Dropout','drop');field('Spojení','merge',['concat','sum']);field('Vstupy (čísla)','parents');
        return row;
    }
    function preview(){
        try{const text=serialize();el('graph-raw').value=text;rawChanged=false;const used=new Set([nodes.length]);
            for(let i=nodes.length-1;i>=0;i--)if(used.has(i+1))nodes[i].parents.trim().split(/[\s,]+/).forEach(p=>used.add(+p));
            if(nodes.some((_,i)=>!used.has(i+1)))throw Error('Připojte všechny větve k poslední vrstvě.');
            if(!text)throw Error('Přidejte vrstvu.');
            el('graph-note').textContent=nodes.length+' vrstev · výstup z vrstvy '+nodes.length;return text;
        }catch(e){el('graph-note').textContent=e.message;return null;}
    }
    function render(){list.replaceChildren(...nodes.map(fields));preview();}
    function refresh(){
        const custom=engine.value!=='tdnn';
        el('graph-options').hidden=!custom;el('design').textContent=custom?'Upravit architekturu ↗':'Navrhnout vlastní síť ↗';
        for(const n of ['rate','batch'])document.getElementById('recipe').elements[n].disabled=!custom;
        const types=spec.value.trim().split(/\r?\n/).map(s=>s.split(' ')[0]);
        el('graph-summary').textContent=types.slice(0,5).join(' → ')+(types.length>5?' … ('+types.length+')':'');
        el('gpu-state').textContent=gpu?gpu:('CUDA · '+(reason||'nedostupná'));
    }
    el('design').onclick=()=>{try{nodes=parse(spec.value);render();el('preset').value='';dialog.showModal();}catch(e){el('gpu-state').textContent=e.message;}};
    el('preset').onchange=e=>{if(presets[e.target.value]){nodes=parse(presets[e.target.value]);render();}};
    el('add-node').onclick=()=>{nodes.push({kind:'Dense',width:16,act:'tanh',drop:0,merge:'concat',parents:String(nodes.length)});render();};
    el('dismiss').onclick=()=>dialog.close();
    el('import-graph').onclick=()=>{try{nodes=parse(el('graph-raw').value);render();}catch(e){el('graph-note').textContent=e.message;}};
    el('graph-raw').oninput=()=>{rawChanged=true;};
    el('apply-graph').onclick=()=>{if(rawChanged){try{nodes=parse(el('graph-raw').value);render();}catch(e){el('graph-note').textContent=e.message;return;}}
        const text=preview();if(!text)return;spec.value=text;
        if(engine.value==='tdnn')engine.value='cpu';refresh();spec.dispatchEvent(new Event('change',{bubbles:true}));dialog.close();};
    engine.onchange=refresh;
    refresh();return {refresh,hardware(name,problem){if(name!==gpu||problem!==reason){gpu=name;reason=problem;refresh();}}};
})();
