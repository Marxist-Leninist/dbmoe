const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/web/progress/index.html','utf8');
const state=html.slice(html.indexOf('function workerState('),html.indexOf('let ownerWorkers='));
const render=html.slice(html.indexOf('function workers('),html.indexOf('let generating='));
class El{constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';}append(...xs){this.children.push(...xs)}replaceChildren(...xs){this.children=xs}}
function fixture(){const nodes={};const ctx={Date,Map,Set,Number,Object,Array,String,Infinity,localStorage:{setItem(){}},document:{createElement:t=>new El(t)},workerHistory:{},workerHistoryKey:'test',workerAliases:new Map(),initialWorkers:null,lastSuccess:Date.now(),ownerView:false,ownerWorkers:new Set(),ownerTestCompletions:{},latestWorkerSnapshot:null,latestWorkerClock:0,latestWorkerReceived:0,fmt:new Intl.NumberFormat('en-GB'),hardwareLines:()=>null,$:id=>nodes[id]||(nodes[id]=new El('div')),show:(id,v)=>{(nodes[id]||(nodes[id]=new El('div'))).textContent=v}};vm.createContext(ctx);vm.runInContext(state+'\n'+render,ctx);return {ctx,nodes}}
const id='worker-0123456789abcdef';
{
 const {ctx,nodes}=fixture();ctx.workers({workers:{},active_leases:[],net_inference:{peers:{[id]:{block:2,hops:4,ms:12}}}},Date.now());
 assert.strictEqual(nodes.workerCards.children.length,1);assert.match(nodes.workerSummary.textContent,/^1 workers · 1 online/);const c=nodes.workerCards.children[0];assert.strictEqual(c.children[1].textContent,'Online');assert.match(c.children[5].textContent,/Serving block 3/);assert.strictEqual(ctx.workerHistory[id].updates,0);
}
{
 const {ctx,nodes}=fixture();ctx.workers({workers:{[id]:{updates:7,last:new Date().toISOString()}},active_leases:[{worker:id,block:1,age_s:1}],net_inference:{peers:{[id]:{block:2,hops:4,ms:12}}}},Date.now());assert.strictEqual(nodes.workerCards.children.length,1);assert.match(nodes.workerSummary.textContent,/1 workers · 1 online · 1 training/);assert.strictEqual(nodes.workerCards.children[0].children[1].textContent,'Online · training');
}
{
 const {ctx,nodes}=fixture();ctx.workers({net_inference:{peers:{'raw-hostname':{block:0}}}},Date.now());assert.deepStrictEqual(Object.keys(ctx.workerHistory),[]);assert.strictEqual(nodes.workerCards.children[0].textContent,'No workers reported yet.');
}
{
 const {ctx,nodes}=fixture();ctx.workers({net_inference:{peers:{[id]:{block:0}}}},Date.now());ctx.workers({net_inference:{peers:{}}},Date.now()+1000);assert.strictEqual(nodes.workerCards.children.length,1);assert.strictEqual(nodes.workerCards.children[0].children[1].textContent,'Not seen lately');ctx.workers({left_workers:[id],net_inference:{peers:{}}},Date.now()+2000);assert.strictEqual(Object.keys(ctx.workerHistory).length,0);
}
console.log('4 actual-renderer worker visibility contracts passed');
