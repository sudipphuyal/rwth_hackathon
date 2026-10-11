'use strict';
(() => {
  const bundle = window.DASHBOARD_DATA;
  const status = document.getElementById('load-status');
  if (!bundle?.evidence || !bundle?.procurement) {
    status.hidden = false;
    status.textContent = 'The project data snapshot could not be loaded. Keep data.js beside this page and reopen it.';
    return;
  }
  const E = bundle.evidence, P = bundle.procurement;
  const colors = { actual:'#123f70', a:'#bd352c', b:'#3b70b4', baseline:'#8290a0', ink:'#132940', border:'#d9e2ec' };
  const fmt = (x,d=3) => Number(x).toLocaleString('en-GB',{minimumFractionDigits:d,maximumFractionDigits:d});
  const signed = (x,d=2) => `${x>=0?'+':'−'}${fmt(Math.abs(x),d)}%`;
  const escape = value => String(value).replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
  const dateLabel = value => new Date(value+'T00:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
  const percentile = q => `P${Number((q*100).toFixed(1))}`;
  const ratio = row => `${row.C_under}:${row.C_over}`;
  const humanMethod = method => method==='ML'?'Approach A':'Lag blend';
  const tooltip = document.getElementById('chart-tooltip');
  const svgNS = 'http://www.w3.org/2000/svg';
  const charts = new Map();
  const hiddenSeries = new Map();
  let activeTab = 'overview';
  function element(tag,attributes={},text='') {
    const node=document.createElementNS(svgNS,tag);
    Object.entries(attributes).forEach(([k,v])=>node.setAttribute(k,String(v)));
    if(text!==undefined)node.textContent=text;
    return node;
  }
  function svgBase(host,height,label) {
    const w=Math.max(280,host.clientWidth);
    const svg=element('svg',{viewBox:`0 0 ${w} ${height}`,height,role:'img','aria-label':label});
    svg.append(element('title',{},label));
    host.replaceChildren(svg);
    return {svg,w};
  }
  function showTip(event,html) {
    tooltip.innerHTML=html;tooltip.hidden=false;
    const width=tooltip.offsetWidth,height=tooltip.offsetHeight;
    tooltip.style.left=Math.max(8,Math.min(event.clientX+15,window.innerWidth-width-8))+'px';
    tooltip.style.top=Math.max(8,Math.min(event.clientY+13,window.innerHeight-height-8))+'px';
  }
  function hideTip(){tooltip.hidden=true;}
  function legend(id,series,chartId) {
    const host=document.getElementById(id);
    host.replaceChildren();
    for(const item of series){
      const button=document.createElement('button');button.type='button';button.className='legend-item';
      button.setAttribute('aria-pressed','true');
      button.innerHTML=`<span class="swatch ${item.dashed?'dashed':''}" style="color:${item.color}"></span>${escape(item.name)}`;
      button.addEventListener('click',()=>{
        const hidden=hiddenSeries.get(chartId)||new Set();
        if(hidden.has(item.key)){hidden.delete(item.key);button.setAttribute('aria-pressed','true');}
        else {if(hidden.size===series.length-1)return;hidden.add(item.key);button.setAttribute('aria-pressed','false');}
        hiddenSeries.set(chartId,hidden);charts.get(chartId)?.();hideTip();
      });
      host.append(button);
    }
  }
  function registerChart(id,draw){charts.set(id,draw);draw();}
  function lineChart(id,rows,series,{xKey='x',dates=false,yLabel='',height=350,residual=false,xMax=null,title=''}={}) {
    const draw=()=>{
      const host=document.getElementById(id);if(host.closest('[hidden]'))return;
      const {svg,w}=svgBase(host,height,title||yLabel);
      const m={l:65,r:20,t:18,b:58},bottom=height-m.b,right=w-m.r,pw=right-m.l,ph=bottom-m.t;
      const visible=series.filter(s=>!hiddenSeries.get(id)?.has(s.key));
      const xs=rows.map(r=>dates?Date.parse(r[xKey]+'T00:00:00Z'):r[xKey]);
      const vals=rows.flatMap(r=>visible.map(s=>r[s.key])).filter(v=>v!==null&&v!==undefined&&Number.isFinite(v));
      if(!vals.length){svg.append(element('text',{x:m.l,y:70,class:'chart-label'},'No numeric values are available for this selection.'));return;}
      let low=residual?Math.min(0,...vals):0,high=Math.max(0,...vals);
      const span=Math.max(1,high-low);if(residual)low-=span*.08;high+=span*.09;
      const xmin=Math.min(...xs),xmax=xMax===null?Math.max(...xs):xMax;
      const x=v=>m.l+(v-xmin)/Math.max(1,xmax-xmin)*pw,y=v=>bottom-(v-low)/Math.max(1,high-low)*ph;
      for(let i=0;i<=4;i++){
        const v=low+(high-low)*i/4,yp=y(v);
        svg.append(element('line',{x1:m.l,x2:right,y1:yp,y2:yp,class:'chart-grid'}));
        svg.append(element('text',{x:m.l-10,y:yp+5,'text-anchor':'end',class:'chart-label chart-muted'},fmt(v,high<30?1:0)));
      }
      const count=w<520?3:6;
      for(let i=0;i<=count;i++){
        const value=xmin+(xmax-xmin)*i/count;
        const label=dates?new Date(value).toLocaleDateString('en-GB',{month:'short',year:w<520?undefined:'2-digit',timeZone:'UTC'}):fmt(value,0);
        svg.append(element('text',{x:x(value),y:bottom+24,'text-anchor':i===0?'start':i===count?'end':'middle',class:'chart-label chart-muted'},label));
      }
      svg.append(element('text',{x:(m.l+right)/2,y:height-8,'text-anchor':'middle',class:'chart-axis-title'},dates?'Delivery date':'Hour of delivery day (UTC)'));
      svg.append(element('text',{transform:`translate(17 ${(m.t+bottom)/2}) rotate(-90)`,'text-anchor':'middle',class:'chart-axis-title'},yLabel));
      if(residual){svg.append(element('line',{x1:m.l,x2:right,y1:y(0),y2:y(0),stroke:colors.ink,'stroke-width':1}));}
      for(const s of visible){
        let d='',connected=false;
        rows.forEach((r,i)=>{const v=r[s.key];if(v===null||v===undefined||!Number.isFinite(v)){connected=false;return;}d+=`${connected?'L':'M'}${x(xs[i]).toFixed(2)},${y(v).toFixed(2)} `;connected=true;});
        svg.append(element('path',{d,stroke:s.color,class:'chart-line',...(s.dashed?{'stroke-dasharray':'6 4'}:{})}));
      }
      const guide=element('line',{x1:0,x2:0,y1:m.t,y2:bottom,class:'chart-guide',visibility:'hidden'});svg.append(guide);
      const marks=visible.map(s=>{const dot=element('circle',{r:4,fill:s.color,class:'chart-dot',visibility:'hidden'});svg.append(dot);return dot;});
      const overlay=element('rect',{x:m.l,y:m.t,width:pw,height:ph,fill:'transparent','aria-label':'Move across the plot to inspect measured values'});svg.append(overlay);
      function inspect(event){
        const bounds=svg.getBoundingClientRect(),cursor=(event.clientX-bounds.left)*w/bounds.width;
        const target=xmin+(Math.min(right,Math.max(m.l,cursor))-m.l)/pw*(xmax-xmin);
        let lo=0,hi=xs.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(xs[mid]<target)lo=mid+1;else hi=mid;}
        let idx=lo;if(idx>0&&Math.abs(xs[idx-1]-target)<Math.abs(xs[idx]-target))idx--;
        const r=rows[idx],xp=x(xs[idx]);guide.setAttribute('x1',xp);guide.setAttribute('x2',xp);guide.setAttribute('visibility','visible');
        const label=dates?dateLabel(r[xKey]):`${Math.floor(r[xKey])}:${String(Math.round((r[xKey]%1)*60)).padStart(2,'0')} UTC`;
        let html=`<b>${label}</b>`;
        visible.forEach((s,j)=>{const val=r[s.key];marks[j].setAttribute('visibility',val===null||val===undefined?'hidden':'visible');if(val!==null&&val!==undefined){marks[j].setAttribute('cx',xp);marks[j].setAttribute('cy',y(val));}html+=`<div class="tip-row"><span>${escape(s.name)}</span><span>${val===null||val===undefined?'No measurement':fmt(val,2)}</span></div>`;});
        html+=`<span>${escape(yLabel)}</span>`;showTip(event,html);
      }
      overlay.addEventListener('pointermove',inspect);overlay.addEventListener('pointerdown',inspect);
      overlay.addEventListener('pointerleave',()=>{guide.setAttribute('visibility','hidden');marks.forEach(dot=>dot.setAttribute('visibility','hidden'));hideTip();});
    };
    registerChart(id,draw);
  }
  function barChart(id,items,{xLabel='',height=null,title=''}={}) {
    const draw=()=>{
      const host=document.getElementById(id);if(host.closest('[hidden]'))return;
      const mobile=host.clientWidth<650,per=mobile?69:51;
      const h=height||Math.max(190,items.length*per+66),{svg,w}=svgBase(host,h,title||xLabel);
      const left=mobile?15:Math.min(330,w*.32),right=w-65,top=13,bottom=h-46;
      const maximum=Math.max(...items.map(r=>r.value))*1.12;
      const x=v=>left+v/Math.max(1,maximum)*(right-left);
      const rowHeight=(bottom-top)/items.length;
      for(let i=0;i<=4;i++){const v=maximum*i/4;const tick=maximum>100000?`${fmt(v/1000000,2)}m`:fmt(v,maximum>10000?0:1);svg.append(element('line',{x1:x(v),x2:x(v),y1:top,y2:bottom,class:'chart-grid'}));svg.append(element('text',{x:x(v),y:bottom+22,'text-anchor':'middle',class:'chart-label chart-muted'},tick));}
      items.forEach((item,i)=>{
        const y=top+i*rowHeight;
        const labelY=mobile?y+15:y+rowHeight/2+5;
        svg.append(element('text',{x:mobile?left:left-13,y:labelY,'text-anchor':mobile?'start':'end',class:'chart-label'},item.label));
        const barY=mobile?y+25:y+rowHeight/2-10;
        svg.append(element('rect',{x:left,y:barY,width:Math.max(0,x(item.value)-left),height:20,fill:item.color||colors.a}));
        svg.append(element('text',{x:x(item.value)+9,y:barY+15,class:'chart-label'},fmt(item.value,item.decimals??2)+(item.suffix||'')));
      });
      svg.append(element('text',{x:(left+right)/2,y:h-5,'text-anchor':'middle',class:'chart-axis-title'},xLabel));
    };
    registerChart(id,draw);
  }
  function openTab(id,focus=false){
    if(!document.getElementById(id))return;
    activeTab=id;
    document.querySelectorAll('[role=tabpanel]').forEach(p=>p.hidden=p.id!==id);
    document.querySelectorAll('[role=tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===id)));
    document.getElementById('tab-'+id)?.setAttribute('aria-selected','true');
    hideTip();requestAnimationFrame(()=>charts.forEach(draw=>draw()));
    if(focus)document.getElementById('tab-'+id).focus();
    history.replaceState(null,'','#'+id);
  }
  const tabs=[...document.querySelectorAll('[role=tab]')];
  tabs.forEach((button,index)=>{
    button.addEventListener('click',()=>{openTab(button.dataset.tab);document.querySelector('main').scrollIntoView({behavior:'smooth',block:'start'});});
    button.addEventListener('keydown',event=>{if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();const next=(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;openTab(tabs[next].dataset.tab,true);}});
  });
  document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',()=>{openTab(b.dataset.open);document.querySelector('main').scrollIntoView({behavior:'smooth'});}));
  const annualSeries=[{key:'actualMWh',name:'Measured demand',color:colors.actual},{key:'forecastMWh',name:'Approach A forecast',color:colors.a},{key:'lagBlendMWh',name:'Lag benchmark',color:colors.baseline,dashed:true}];
  legend('overview-legend',annualSeries,'overview-chart');
  lineChart('overview-chart',E.dailySeries.rows||E.dailySeries,annualSeries,{xKey:'date',dates:true,yLabel:'Daily energy (MWh)',height:350,title:'Annual measured demand, Approach A and lag benchmark'});
  const metricIds=['householdMae','householdRmse','portfolioWape','dailyMae'];
  document.getElementById('annual-metric-grid').innerHTML=metricIds.map(id=>{
    const m=E.annualPerformance.metrics.find(row=>row.id===id),good=m.relativeImprovementPct>=0;
    const names={householdMae:'Household MAE',householdRmse:'Household RMSE',portfolioWape:'Portfolio WAPE',dailyMae:'Daily-total MAE'};
    return `<article class="metric"><h3>${names[id]}</h3><strong class="metric-value">${fmt(m.approachA,id==='portfolioWape'?2:3)}${m.unit==='%'?'%':''}</strong><p>Benchmark: ${fmt(m.lagBlend,id==='portfolioWape'?2:3)}${m.unit==='%'?'%':''}</p><p>${m.unit==='%'?'Percentage of measured energy':escape(m.unit)}</p><p class="delta ${good?'':'worse'}">${fmt(Math.abs(m.relativeImprovementPct),2)}% ${good?'lower':'higher'} error</p></article>`;
  }).join('');
  function updateDay(){
    const date=document.getElementById('test-day').value;
    const profile=(E.dayProfiles.rows||E.dayProfiles).find(row=>row.date===date);
    const compared=E.historicalComparison.rows.find(row=>row.date===date).A;
    document.getElementById('day-summary').textContent=`${compared.homes} measured homes · Actual ${fmt(compared.actualMWh)} MWh · A forecast ${fmt(compared.forecastMWh)} MWh · Daily signed error ${signed(compared.signedErrorPct)}`;
    const rows=profile.hoursUTC.map((hour,i)=>({hour,actual:profile.actualKWh[i],forecast:profile.forecastKWh[i],residual:profile.residualKWh[i]}));
    const series=[{key:'actual',name:'Measured demand',color:colors.actual},{key:'forecast',name:'A forecast',color:colors.a}];
    legend('day-legend',series,'day-chart');hiddenSeries.delete('day-chart');
    lineChart('day-chart',rows,series,{xKey:'hour',xMax:24,yLabel:'Energy (kWh per 15 minutes)',height:320,title:dateLabel(date)+' actual and predicted interval demand'});
    lineChart('residual-chart',rows,[{key:'residual',name:'Forecast − actual',color:colors.a}],{xKey:'hour',xMax:24,yLabel:'Residual (kWh per 15 minutes)',height:200,residual:true,title:'Forecast minus actual: positive means overprediction'});
  }
  document.getElementById('test-day').addEventListener('change',updateDay);updateDay();
  document.querySelector('#historical-table tbody').innerHTML=E.historicalComparison.rows.flatMap(row=>['A','B'].map(method=>{
    const v=row[method];return `<tr><td>${dateLabel(row.date)}</td><td>Approach ${method}</td><td class="numeric">${v.homes}</td><td class="numeric">${fmt(v.actualMWh)}</td><td class="numeric">${fmt(v.forecastMWh)}</td><td class="numeric">${signed(v.signedErrorPct)}</td><td class="numeric">${fmt(v.absoluteDailyErrorMWh)}</td></tr>`;
  })).join('');
  const featureLabels=['History + calendar','Add weather','Add building','Add heating','Add recent demand','Recent + PV'];
  barChart('feature-chart',E.featureStudy.rows.map((row,i)=>({label:featureLabels[i],value:row.portfolioWapePct,color:row.selected?colors.a:colors.baseline,suffix:'%'})),{xLabel:'Validation portfolio WAPE (%)',title:'Validation feature groups, with selected recent-demand model highlighted'});
  document.getElementById('weather-grid').innerHTML=E.weather.rows.map(row=>`<article class="metric"><h3>${escape(row.variable)}</h3><strong class="metric-value">${fmt(row.mae,row.variable==='Sunshine duration'?3:2)}</strong><p>${escape(row.unit)}</p><p>${escape(row.model)}</p></article>`).join('');
  const dailyQuantities=[{label:'P10',value:E.finalForecast.A.p10MWh,color:colors.baseline,decimals:3},{label:'P50 median',value:E.finalForecast.A.p50MWh,color:colors.baseline,decimals:3},{label:'A point forecast',value:E.finalForecast.A.pointMWh,color:colors.a,decimals:3},{label:'P80 daily purchase (4:1)',value:P.dailyRows.find(row=>row.Method==='ML'&&row.Scenario==='C').SeparatelyCalibratedDailyProcurement_MWh,color:colors.b,decimals:3},{label:'P90',value:E.finalForecast.A.p90MWh,color:colors.baseline,decimals:3}];
  barChart('daily-quantiles-chart',dailyQuantities,{xLabel:'Whole-day electricity energy (MWh)',title:'Daily model quantities: percentiles and point forecast'});
  function updatePurchasing(){
    const method=document.getElementById('purchase-model').value,scenario=document.getElementById('purchase-ratio').value,level=document.getElementById('purchase-level').value;
    const row=P.dailyRows.find(r=>r.Method===method&&r.Scenario===scenario);
    const historic=P.historicalRows.find(r=>r.Method===method&&r.Scenario===scenario&&r.Level===level);
    const q=row.OptimalQuantile,label=percentile(q),daily=level==='Daily';
    const quantity=daily?row.SeparatelyCalibratedDailyProcurement_MWh:row.SumIntervalProcurement_MWh;
    const reduction=historic.RelativeCostReductionPct,better=reduction>=0;
    document.getElementById('purchase-headlines').innerHTML=`<article class="headline"><p>Model point forecast</p><strong>${fmt(row.PointDailyEnergy_MWh)} <small>MWh</small></strong><span>28 February, all 410 households</span></article><article class="headline"><p>${daily?'Selected daily purchase':'Sum of interval purchases'}</p><strong>${fmt(quantity)} <small>MWh</small></strong><span>${label}, under ${ratio(row)} penalties</span><small>${daily?'Separate net daily calibration':'96 individual interval quantities'}</small></article><article class="headline"><p>Historical simulated penalty</p><strong>${fmt(Math.abs(reduction),2)}% ${better?'less':'more'}</strong><span>Versus this method’s point forecast</span><small>${daily?'Net daily':'Quarter-hour'} test settlement, not final-day savings</small></article>`;
    const meaning=row.C_under===row.C_over?'Each missing and extra kWh receive the same assumed penalty.':row.C_under>row.C_over?`Each missing kWh costs ${row.C_under/row.C_over} times as much as each extra kWh.`:`Each extra kWh costs ${row.C_over/row.C_under} times as much as each missing kWh.`;
    document.getElementById('policy-explanation').innerHTML=`${meaning} <strong>${row.C_under} ÷ (${row.C_under} + ${row.C_over}) = ${fmt(q*100,1)}%</strong>, so select <strong>${label}</strong> from estimated demand. ${daily?'This is a single daily decision.':'Apply that percentile separately to each 15-minute interval.'}`;
    document.getElementById('purchase-chart-title').textContent=daily?'28 February: separately calibrated daily purchase':'28 February: demand forecast and interval purchase schedule';
    document.getElementById('purchase-chart-description').textContent=daily?'The daily purchase uses historical whole-day error blocks. It is separate from the sum of interval purchases.':'The purchase curve comes from validation-calibrated demand uncertainty. No actual demand is available for this day.';
    hiddenSeries.delete('purchase-chart');
    if(daily){
      document.getElementById('purchase-legend').replaceChildren();
      barChart('purchase-chart',[{label:'Point forecast',value:row.PointDailyEnergy_MWh,color:colors.a,decimals:3},{label:'P10',value:row.P10DailyEnergy_MWh,color:colors.baseline,decimals:3},{label:'P50',value:row.P50DailyEnergy_MWh,color:colors.baseline,decimals:3},{label:`${label} selected purchase`,value:quantity,color:colors.b,decimals:3},{label:'P90',value:row.P90DailyEnergy_MWh,color:colors.baseline,decimals:3}],{xLabel:'Whole-day electricity energy (MWh)',title:'Daily point, quantiles and selected purchase'});
    }else{
      const series=[{key:'point',name:`${humanMethod(method)} demand forecast`,color:colors.a},{key:'purchase',name:`${label} purchase quantity`,color:colors.b,dashed:true}];
      legend('purchase-legend',series,'purchase-chart');
      const schedule=P.schedules[method].map((s,i)=>({hour:i/4,point:s.PointForecast_kWh,purchase:s['Q_'+scenario+'_kWh']}));
      lineChart('purchase-chart',schedule,series,{xKey:'hour',xMax:24,yLabel:'Energy (kWh per 15 minutes)',height:330,title:'Forecast demand and selected cost-based interval purchase schedule'});
    }
    document.getElementById('balance-intro').innerHTML=`At <strong>${label}</strong>, the estimated probability of demand at or below the quantity is <strong>${fmt(q*100,1)}%</strong>, leaving <strong>${fmt((1-q)*100,1)}%</strong> above it. These probabilities depend on the estimated distribution.`;
    document.getElementById('probability-strip').innerHTML=`<span class="covered" style="width:${q*100}%">${fmt(q*100,1)}% at or below</span><span class="shortage" style="width:${(1-q)*100}%">${fmt((1-q)*100,1)}% above</span>`;
    document.getElementById('balance-values').innerHTML=`<div>Expected avoided shortage penalty<strong>${row.C_under} × ${fmt(1-q,3)} = ${fmt(row.C_under*(1-q),3)}</strong></div><div>Expected added surplus penalty<strong>${row.C_over} × ${fmt(q,3)} = ${fmt(row.C_over*q,3)}</strong></div>`;
    const first=P.schedules[method][0],adjustment=first['Q_'+scenario+'_kWh']-first.PointForecast_kWh;
    document.getElementById('interval-example').innerHTML=daily?`<h3>The whole-day quantity</h3><p>Point forecast ${fmt(row.PointDailyEnergy_MWh)} MWh, plus the selected daily uncertainty adjustment ${adjustmentText(quantity-row.PointDailyEnergy_MWh,'MWh')}, gives <strong>${fmt(quantity)} MWh</strong>. The adjustment comes from whole-day validation error blocks.</p>`:`<h3>A real interval example: 00:00 UTC on 28 February</h3><p>Point forecast <strong>${fmt(first.PointForecast_kWh,2)} kWh</strong> ${adjustment>=0?'+':'−'} validation-derived uncertainty adjustment <strong>${fmt(Math.abs(adjustment),2)} kWh</strong> = <strong>${fmt(first['Q_'+scenario+'_kWh'],2)} kWh</strong> at ${label}. Repeating this for all 96 intervals gives ${fmt(quantity)} MWh.</p>`;
    barChart('cost-chart',[{label:'Buy the point forecast',value:historic.PointCost,color:colors.a,decimals:0},{label:`Buy ${label}`,value:historic.QuantileCost,color:colors.b,decimals:0}],{xLabel:'Total historical simulated penalty units',height:215,title:'Point versus quantile historical simulated penalty'});
    const metric=P.testPolicyMetrics.find(r=>r.Method===method&&r.Scenario===scenario&&r.Level===level&&r.Policy==='CostOptimalQuantile');
    document.getElementById('cost-details').textContent=`${metric.N.toLocaleString('en-GB')} measured ${daily?'days':'intervals'} scored. Quantile-policy shortage: ${fmt(metric.TotalShortage_kWh/1000,2)} MWh; surplus: ${fmt(metric.TotalSurplus_kWh/1000,2)} MWh. Penalty = ${row.C_under} × shortage kWh + ${row.C_over} × surplus kWh. ${better?'Lower':'Higher'} test penalty does not guarantee future savings.`;
    document.querySelectorAll('#procurement-table tbody tr').forEach(tr=>tr.classList.toggle('selected-row',tr.dataset.method===method&&tr.dataset.scenario===scenario));
  }
  function adjustmentText(value,unit){return `${value>=0?'+':'−'}${fmt(Math.abs(value))} ${unit}`;}
  document.querySelector('#procurement-table tbody').innerHTML=P.dailyRows.map(row=>{
    const h=P.historicalRows.find(r=>r.Method===row.Method&&r.Scenario===row.Scenario&&r.Level==='Quarter-hour');
    return `<tr data-method="${row.Method}" data-scenario="${row.Scenario}"><td>${humanMethod(row.Method)}</td><td>${ratio(row)}</td><td>${percentile(row.OptimalQuantile)}</td><td class="numeric">${fmt(row.PointDailyEnergy_MWh)}</td><td class="numeric">${fmt(row.SeparatelyCalibratedDailyProcurement_MWh)}</td><td class="numeric">${fmt(row.SumIntervalProcurement_MWh)}</td><td class="numeric">${fmt(Math.abs(h.RelativeCostReductionPct),2)}% ${h.RelativeCostReductionPct>=0?'less':'more'}</td></tr>`;
  }).join('');
  ['purchase-model','purchase-ratio','purchase-level'].forEach(id=>document.getElementById(id).addEventListener('change',updatePurchasing));updatePurchasing();
  const sourceGroups=[
    ['Project scope and report',['finalReport','preprocessing','bReport']],
    ['Approach A selection and error metrics',['aModelLock','aReport','aMetrics','aImprovement','aDaily']],
    ['Reported historical comparison',['historicalComparison','comparisonProvenance','finalComparison']],
    ['Features and weather',['aFeatures','aWeather','aWeatherReport']],
    ['Uncertainty',['aUncertainty','aUncertaintyReport']],
    ['Approach B implementation',['bCode','bReport']],
  ];
  const linkFor=path=>bundle.sourceFiles[path];
  document.getElementById('source-list').innerHTML=sourceGroups.map(([label,ids])=>`<article class="source-entry"><h3>${label}</h3><ul>${ids.filter(id=>E.sources[id]).map(id=>{const s=E.sources[id],url=linkFor(s.path);return `<li>${url?`<a href="${escape(url)}" target="_blank" rel="noopener">${escape(s.path.split('/').pop())}</a>`:escape(s.path.split('/').pop())}<br><code>${escape(s.path)}</code></li>`;}).join('')}</ul></article>`).join('')+`<article class="source-entry"><h3>Procurement and P80</h3><ul>${P.sources.filter(s=>!s.relativePath.endsWith('.parquet')).map(s=>{const url=linkFor(s.relativePath);return `<li>${url?`<a href="${escape(url)}" target="_blank" rel="noopener">${escape(s.label)}</a>`:escape(s.label)}<br><code>${escape(s.relativePath)}</code></li>`;}).join('')}</ul></article>`;
  let timer;new ResizeObserver(()=>{clearTimeout(timer);timer=setTimeout(()=>charts.forEach(draw=>draw()),80);}).observe(document.querySelector('main'));
  const initial=location.hash.replace('#','');openTab(document.getElementById('tab-'+initial)?initial:'overview');
  window.Dashboard={openTab,updatePurchasing,data:bundle,get activeTab(){return activeTab;}};
})();
