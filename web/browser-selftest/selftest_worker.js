// TensorFlow.js 4.22 assigns this legacy binding during startup. Declaring it
// avoids its Function(...) fallback while retaining the strict page policy.
var regeneratorRuntime;
importScripts('./vendor/tf.min.js','./vendor/tf-backend-webgpu.min.js','./vendor/tf-backend-wasm.min.js');
let busy=false;
self.onmessage=async({data})=>{
 if(busy)return;busy=true;let model;
 try{
  const {createModel,FORMAT}=await import('./model.js');const backend=data.backend;
  if(!['cpu','wasm','webgpu'].includes(backend))throw Error('Invalid backend');
  if(backend==='webgpu'&&!navigator.gpu)throw Error('WebGPU is unavailable in this browser worker.');
  if(backend==='wasm'){tf.wasm.setWasmPaths('./vendor/');tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT',false);}
  if(!await tf.setBackend(backend))throw Error('Backend initialization failed');await tf.ready();
  if(tf.getBackend()!==backend)throw Error('Backend mismatch');
  const results=[];
  for(const filename of ['parity.json','parity-skewed.json']){
   postMessage({type:'status',message:'Testing '+backend+' on '+filename+'…'});
   const response=await fetch('./fixtures/'+filename,{credentials:'omit',redirect:'error',cache:'no-store',referrerPolicy:'no-referrer'});
   if(!response.ok)throw Error('Fixture download failed');const text=await response.text();if(text.length>1000000)throw Error('Fixture too large');const f=JSON.parse(text);
   const lease={format:FORMAT,lease:'browser-selftest-0001',cfg:f.cfg,block:0,steps:1,batch:1,lr:f.optimizer.base_lr,ttl:120,sigmas:[.1,1,80],data:[f.batch.data],shared:{},block_state:{}};
   for(const[k,t]of Object.entries(f.initial)){const group=k.startsWith('shared.')?'shared':'block_state';lease[group][k.slice(k.indexOf('.')+1)]={shape:t.shape,values:t.data};}
   const start=performance.now();model=createModel(tf,lease,32);
   const x=tf.tensor2d([f.batch.data],f.batch.shape,'int32'),s=tf.tensor1d(f.sigma.data),eps=tf.tensor(f.epsilon.data,f.epsilon.shape);
   const r=tf.tidy(()=>tf.variableGrads(()=>model.objective(x,s,eps).loss,Object.values(model.vars)));
   const loss=(await r.value.data())[0];let maxGradientError=0,maxDeltaError=0;
   for(const[k,t]of Object.entries(f.expected.gradients)){const key=(k.startsWith('shared.')?'shared/':'block_state/')+k.slice(k.indexOf('.')+1),actual=await r.grads[model.vars[key].name].data();
    for(let i=0;i<actual.length;i++)maxGradientError=Math.max(maxGradientError,Math.abs(actual[i]-t.data[i]));}
   r.value.dispose();Object.values(r.grads).forEach(g=>g.dispose());x.dispose();s.dispose();eps.dispose();
   const metrics=await model.trainStep(lease.data,f.sigma.data,f.epsilon.data),delta=await model.delta();
   for(const[k,t]of Object.entries(f.expected.deltas)){const group=k.startsWith('shared.')?'d_shared':'d_block',name=k.slice(k.indexOf('.')+1),actual=delta[group][name].values;
    for(let i=0;i<actual.length;i++)maxDeltaError=Math.max(maxDeltaError,Math.abs(actual[i]-t.data[i]));}
   model.dispose();model=null;
   const lossError=Math.abs(loss-f.expected.loss),remainingTensors=tf.memory().numTensors;
   const result={fixture:filename,backend:tf.getBackend(),loss,referenceLoss:f.expected.loss,lossError,maxGradientError,maxDeltaError,remainingTensors,seconds:(performance.now()-start)/1000};
   if(![lossError,maxGradientError,maxDeltaError].every(Number.isFinite)||lossError>2e-5||maxGradientError>2e-5||maxDeltaError>2e-6||remainingTensors!==0)throw Error('Numerical parity failed: '+JSON.stringify(result));
   results.push(result);await new Promise(resolve=>setTimeout(resolve,Math.max(100,(performance.now()-start)*3)));
  }
  postMessage({type:'pass',backend:tf.getBackend(),results});
 }catch(error){postMessage({type:'error',error:String(error.message||error)});}
 finally{if(model)model.dispose();busy=false;}
};
