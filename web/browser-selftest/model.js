/* Full DBMoE block training, Apache-2.0 TensorFlow.js runtime supplied locally.
 * PyTorch state_dict layout is retained at the wire boundary. No remote code.
 */
export const FORMAT = 'dbmoe-browser-f32-v1';
export const LIMITS = Object.freeze({maxParameters: 4_000_000, maxBytes: 64*1024*1024,
  maxMemoryMiB: 256, maxSteps: 4, maxBatch: 1, maxContext: 256, maxWidth: 256,
  maxExperts: 4, maxLayers: 2, maxFfn: 512});
const finite = x => typeof x === 'number' && Number.isFinite(x);
const positiveInt = x => Number.isInteger(x) && x > 0;
export function expectedShapes(c) {
  const shared = {'emb.weight':[c.vocab,c.d], pos:[c.ctx,c.d], 'ln.weight':[c.d], 'ln.bias':[c.d], scale:[]};
  const block = {'inp.weight':[c.d,c.d], 'sig.weight':[c.d,1], 'sig.bias':[c.d], 'ln.weight':[c.d], 'ln.bias':[c.d], 'out.weight':[c.d,c.d]};
  for(let i=0;i<c.layers_per_block;i++) {
    const p=`layers.${i}.`; for(const n of ['ln1','ln2']) {block[p+n+'.weight']=[c.d];block[p+n+'.bias']=[c.d];}
    block[p+'qkv.weight']=[3*c.d,c.d];block[p+'proj.weight']=[c.d,c.d];block[p+'moe.router.weight']=[c.experts,c.d];
    block[p+'moe.w1']=[c.experts,c.d,c.ffn];block[p+'moe.w2']=[c.experts,c.ffn,c.d];
  }
  return {shared,block_state:block};
}
export function validateLease(l, memoryMiB=LIMITS.maxMemoryMiB) {
  if(!l||l.format!==FORMAT||typeof l.lease!=='string'||!/^[A-Za-z0-9_-]{8,128}$/.test(l.lease)) throw Error('Unsupported lease format or identifier');
  const c=l.cfg;
  if(!c||!['d','heads','ffn','experts','ctx','layers_per_block','blocks','vocab'].every(k=>positiveInt(c[k]))||c.vocab!==256||c.d>LIMITS.maxWidth||c.ctx>LIMITS.maxContext||c.experts>LIMITS.maxExperts||c.ffn>LIMITS.maxFfn||c.layers_per_block>LIMITS.maxLayers||c.blocks>128||c.heads>16||c.d%c.heads||!finite(c.aux)||c.aux<0||c.aux>1) throw Error('Model exceeds browser capability limits');
  if(!Number.isInteger(l.block)||l.block<0||l.block>=c.blocks||!positiveInt(l.steps)||l.steps>LIMITS.maxSteps||!positiveInt(l.batch)||l.batch>LIMITS.maxBatch||!finite(l.lr)||l.lr<=0||l.lr>0.01||!finite(l.ttl)||l.ttl<=0||l.ttl>86400) throw Error('Lease exceeds browser work limits');
  if(!Array.isArray(l.sigmas)||l.sigmas.length!==c.blocks+1||!l.sigmas.every((x,i)=>finite(x)&&x>0&&x<=1000&&(!i||x>l.sigmas[i-1]))) throw Error('Invalid noise schedule');
  let count=0;
  for(const [group,shapes] of Object.entries(expectedShapes(c))) {
    const source=l[group];
    if(!source||Object.keys(source).sort().join('|')!==Object.keys(shapes).sort().join('|'))throw Error('Parameter names differ');
    for(const [name,shape] of Object.entries(shapes)) {
      const t=source[name], n=shape.reduce((a,b)=>a*b,1);count+=n;
      if(!t||!Array.isArray(t.shape)||t.shape.length!==shape.length||!t.shape.every((x,i)=>Number.isInteger(x)&&x===shape[i])||!Array.isArray(t.values)||t.values.length!==n||!t.values.every(finite))throw Error('Invalid parameter tensor: '+name);
    }
  }
  if(!Array.isArray(l.data)||l.data.length!==l.steps*l.batch||!l.data.every(row=>Array.isArray(row)&&row.length===c.ctx+1&&row.every(x=>Number.isInteger(x)&&x>=0&&x<256)))throw Error('Invalid training data');
  // Conservative planning estimate, not a hard browser/driver allocator limit.
  const estimate=count*96 + l.batch*c.ctx*(c.d*96+c.ctx*c.heads*32+c.ffn*c.experts*32)+16*1024*1024;
  if(count>LIMITS.maxParameters||!finite(memoryMiB)||memoryMiB<32||memoryMiB>LIMITS.maxMemoryMiB||estimate>memoryMiB*1024*1024)throw Error('Model exceeds selected memory budget');
  return {parameters:count, estimatedMemoryMiB:Math.ceil(estimate/1024/1024)};
}
export function createModel(tf, lease, memoryMiB) {
  validateLease(lease,memoryMiB);
  const vars={}, initial={};
  for(const group of ['shared','block_state'])for(const [k,t]of Object.entries(lease[group])){
    const name=group+'/'+k; initial[name]=new Float32Array(t.values);
    vars[name]=tf.tidy(()=>tf.variable(tf.tensor(t.values,t.shape,'float32'),true,name));
  }
  const c=lease.cfg;
  const v=(g,k)=>vars[g+'/'+k];
  const linear=(x,w,b=null)=>{const s=x.shape,out=tf.matMul(x.reshape([-1,s.at(-1)]),w,false,true).reshape([...s.slice(0,-1),w.shape[0]]);return b?out.add(b):out;};
  const ln=(x,g,k)=>{const mean=tf.mean(x,-1,true),dev=x.sub(mean);return dev.mul(tf.rsqrt(tf.mean(dev.square(),-1,true).add(1e-5))).mul(v(g,k+'.weight')).add(v(g,k+'.bias'));};
  // This custom gradient makes the discrete routing mask explicitly non-differentiable.
  const routeMask=tf.customGrad(p=>({value:tf.oneHot(tf.argMax(p,-1),c.experts),gradFunc:()=>tf.zerosLike(p)}));
  function objective(data, sigma, epsilon) {
    const B=data.shape[0],T=c.ctx,d=c.d;
    const ids=data.slice([0,0],[B,T]),target=data.slice([0,1],[B,T]);
    const raw=v('shared','emb.weight'),table=raw.mul(tf.rsqrt(tf.mean(raw.square(),-1,true).add(1e-6))).mul(0.5);
    // One-hot matmul has a portable backward pass; WASM lacks Gather's segment-sum gradient.
    const embed=indices=>tf.matMul(tf.oneHot(indices,256).reshape([B*T,256]),table).reshape([B,T,d]);
    const ctx=embed(ids).add(v('shared','pos'));
    const e=embed(target),s=sigma.reshape([B,1,1]),ss=s.square().add(.25);
    const z=e.add(epsilon.mul(s)),skip=tf.scalar(.25).div(ss),out=s.mul(.5).mul(tf.rsqrt(ss)),cin=tf.rsqrt(ss);
    let h=ctx.add(linear(z.mul(cin),v('block_state','inp.weight'))).add(linear(tf.log(s).div(4),v('block_state','sig.weight'),v('block_state','sig.bias')));
    let aux=tf.scalar(0);
    for(let i=0;i<c.layers_per_block;i++){
      const p=`layers.${i}.`,norm=ln(h,'block_state',p+'ln1');
      const qkv=tf.split(linear(norm,v('block_state',p+'qkv.weight')),3,-1);
      const heads=x=>x.reshape([B,T,c.heads,d/c.heads]).transpose([0,2,1,3]);
      const [q,k,vv]=qkv.map(heads);
      const causal=tf.tensor2d(Array.from({length:T*T},(_,j)=>j%T>Math.floor(j/T)?-1e9:0),[T,T]);
      const attention=tf.matMul(tf.softmax(tf.matMul(q,k,false,true).div(Math.sqrt(d/c.heads)).add(causal),-1),vv).transpose([0,2,1,3]).reshape([B,T,d]);
      h=h.add(linear(attention,v('block_state',p+'proj.weight')));
      const flat=ln(h,'block_state',p+'ln2').reshape([B*T,d]);
      const prob=tf.softmax(linear(flat,v('block_state',p+'moe.router.weight')),-1),mask=routeMask(prob),gate=tf.sum(prob.mul(mask),-1,true);
      let mixed=tf.zerosLike(flat);
      for(let j=0;j<c.experts;j++){
        const w1=v('block_state',p+'moe.w1').slice([j,0,0],[1,d,c.ffn]).reshape([d,c.ffn]);
        const w2=v('block_state',p+'moe.w2').slice([j,0,0],[1,c.ffn,d]).reshape([c.ffn,d]);
        const pre=tf.matMul(flat,w1),gelu=pre.mul(.5).mul(tf.erf(pre.div(Math.SQRT2)).add(1));
        mixed=mixed.add(tf.matMul(gelu,w2).mul(mask.slice([0,j],[B*T,1])).mul(gate));
      }
      aux=aux.add(tf.sum(tf.mean(mask,0).mul(tf.mean(prob,0))).mul(c.experts));h=h.add(mixed.reshape([B,T,d]));
    }
    const D=z.mul(skip).add(linear(ln(h,'block_state','ln'),v('block_state','out.weight')).mul(out));
    const logits=linear(ln(D,'shared','ln'),table).mul(v('shared','scale')).mul(1/(.5*Math.sqrt(d)));
    const ce=tf.neg(tf.sum(tf.oneHot(target,256).mul(tf.logSoftmax(logits,-1)),-1));
    const weights=tf.minimum(sigma.square().add(.25).div(sigma.square().mul(.25)),5).reshape([B,1]);
    return {loss:tf.mean(ce.mul(weights)).div(tf.mean(weights)).add(aux.mul(c.aux)),ce:tf.mean(ce)};
  }
  const optimizer={};let step=0;
  async function trainStep(dataValues,sigmaValues=null,epsilonValues=null) {
    const B=lease.batch,T=c.ctx,d=c.d,lo=lease.sigmas[lease.block],hi=lease.sigmas[lease.block+1];
    const result=tf.tidy(()=>{
      const x=tf.tensor2d(dataValues,[B,T+1],'int32');
      const sigma=sigmaValues?tf.tensor1d(sigmaValues):tf.exp(tf.randomUniform([B],Math.log(lo),Math.log(hi)));
      const eps=epsilonValues?tf.tensor(epsilonValues,[B,T,d]):tf.randomNormal([B,T,d]);
      let ce;
      const {value,grads}=tf.variableGrads(()=>{const r=objective(x,sigma,eps);ce=tf.keep(r.ce);return r.loss;},Object.values(vars));
      const norm=tf.sqrt(tf.addN(Object.values(grads).map(g=>tf.sum(g.square()))));
      const clip=tf.minimum(tf.scalar(1).div(norm.add(1e-6)),1),lr=lease.lr*Math.min(1,(step+1)/10),n=step+1;
      for(const [key,p]of Object.entries(vars)){
        const g=grads[p.name].mul(clip),old=optimizer[key];
        const m=(old?old.m.mul(.9):tf.zerosLike(p)).add(g.mul(.1));
        const vv=(old?old.v.mul(.95):tf.zerosLike(p)).add(g.square().mul(.05));
        const change=m.div(1-Math.pow(.9,n)).div(tf.sqrt(vv.div(1-Math.pow(.95,n))).add(1e-8)).mul(lr);
        p.assign(p.mul(1-lr*(p.rank>=2?.01:0)).sub(change));
        if(old){old.m.dispose();old.v.dispose();}optimizer[key]={m:tf.keep(m),v:tf.keep(vv)};
      }
      return {loss:value,ce,norm};
    });
    step++;
    const values=await Promise.all([result.loss.data(),result.ce.data(),result.norm.data()]);Object.values(result).forEach(t=>t.dispose());
    if(!values.every(a=>Number.isFinite(a[0])))throw Error('Non-finite training result; update discarded');
    return {loss:values[0][0],ce:values[1][0],gradientNorm:values[2][0],step};
  }
  async function delta() {
    const output={d_shared:{},d_block:{}};
    for(const [key,p]of Object.entries(vars)){
      const current=await p.data(),start=initial[key],values=Array.from(current,(x,i)=>x-start[i]);
      if(!values.every(finite))throw Error('Non-finite delta');
      const split=key.indexOf('/'),group=key.slice(0,split)==='shared'?'d_shared':'d_block';output[group][key.slice(split+1)]={shape:p.shape,values};
    }
    return output;
  }
  function dispose(){Object.values(vars).forEach(p=>p.dispose());Object.values(optimizer).forEach(o=>{o.m.dispose();o.v.dispose();});}
  return {vars,objective,trainStep,delta,dispose};
}
