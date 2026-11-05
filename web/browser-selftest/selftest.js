const el=id=>document.getElementById(id);let worker=null,timer=null,generation=0,battery=null;
function stop(message='Stopped. Test results were not sent anywhere.') {generation++;if(worker)worker.terminate();worker=null;clearTimeout(timer);el('start').disabled=!el('consent').checked;el('stop').disabled=true;el('status').textContent=message;}
el('consent').onchange=()=>{if(!el('consent').checked)stop('Consent withdrawn. Stopped.');else if(!worker)el('start').disabled=false;};el('stop').onclick=()=>stop();
document.addEventListener('visibilitychange',()=>{if(document.hidden&&worker)stop('Stopped because this page was hidden.');});window.addEventListener('pagehide',()=>stop());
el('start').onclick=async()=>{
 if(!el('consent').checked||worker)return;const current=++generation;el('start').disabled=true;el('stop').disabled=false;el('results').textContent='';
 try{
  if(document.hidden)throw Error('Keep this page visible');if(navigator.connection?.saveData)throw Error('Data Saver is enabled');
  if(navigator.getBattery&&!battery){try{battery=await navigator.getBattery();battery.addEventListener('chargingchange',()=>{if(!battery.charging&&worker)stop('Stopped because the device was unplugged.');});}catch{}}
  if(current!==generation)return;if(document.hidden||!el('consent').checked)throw Error('Keep this page visible with consent enabled');if(navigator.connection?.saveData)throw Error('Data Saver is enabled');if(battery&&!battery.charging)throw Error('Plug in your device before testing');
  const backend=el('backend').value;if(!['cpu','wasm','webgpu'].includes(backend))throw Error('Invalid backend');
  worker=new Worker('./selftest_worker.js');timer=setTimeout(()=>stop('Time limit reached. Worker stopped.'),120000);
  worker.onerror=event=>{if(current===generation){el('results').textContent=(event.message||'Worker startup failed before a result was available.')+(event.filename?'\n'+event.filename+':'+event.lineno+':'+event.colno:'');stop('FAIL: worker could not run in this browser.');}};
  worker.onmessage=({data})=>{if(current!==generation)return;
   if(data.type==='status')el('status').textContent=data.message;
   if(data.type==='pass'){el('results').textContent=JSON.stringify(data.results,null,2);stop('PASS: '+data.backend+' completed real forward/backward/AdamW parity checks for both fixtures. Stopped.');}
   if(data.type==='error'){el('results').textContent=data.error;stop('FAIL: selected backend did not pass. No automatic fallback.');}
  };
  worker.postMessage({backend});el('status').textContent='Loading local training runtime. Test is running…';
 }catch(error){if(current===generation)stop('Could not start: '+error.message);}
};
