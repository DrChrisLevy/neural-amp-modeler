const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const {webcrypto} = require('node:crypto');
function setup() {
  const context = {window:{}, crypto:webcrypto, FormData};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/nam_studio/static/uploads.js'), 'utf8'), context);
  const requests=[], completed=[];
  let changes=0, unauthorized=0;
  const uploads = new context.window.NAMUploads({
    changed:()=>changes++, completed:(take)=>completed.push(take), unauthorized:()=>unauthorized++,
    request:()=>{ const r={upload:{},open(){},send(body){this.body=body;}}; requests.push(r); return r; },
  });
  const file=new File(['wav'], 'input.wav', {type:'audio/wav'});
  return {uploads, requests, completed, file, changes:()=>changes, unauthorized:()=>unauthorized};
}
test('two uploads run independently and may finish out of order',()=>{
  const {uploads, requests, completed, file}=setup();
  const a=uploads.add('A',file,file), b=uploads.add('B',file,file);
  assert.equal(requests.length,2);
  requests[0].upload.onprogress({lengthComputable:true,loaded:25,total:100});
  assert.equal(uploads.items.get(a).progress,25);
  assert.equal(uploads.items.get(b).progress,0);
  requests[1].upload.onload();
  assert.equal(uploads.items.get(b).status,'processing');
  Object.assign(requests[1],{status:200,response:{id:b,name:'B'}}); requests[1].onload();
  assert.equal(completed[0].id,b);
  assert.equal(uploads.items.size,1);
  assert.equal(uploads.active,true);
  Object.assign(requests[0],{status:200,response:{id:a,name:'A'}}); requests[0].onload();
  assert.equal(uploads.items.size,0);
  assert.equal(uploads.active,false);
});
test('failed uploads keep their files and stable identity for retry',()=>{
  const {uploads,requests,file}=setup();
  const id=uploads.add('Retry me',file,file);
  requests[0].onerror();
  assert.equal(uploads.items.get(id).status,'failed');
  uploads.start(id);
  assert.equal(requests.length,2);
  assert.equal(requests[1].body.get('upload_id'),requests[0].body.get('upload_id'));
  assert.equal(requests[1].body.get('dry').name,'input.wav');
  uploads.start(id);
  assert.equal(requests.length,2); // Cannot double-send an active upload.
  requests[1].ontimeout();
  assert.equal(uploads.items.get(id).status,'failed');
});
test('authentication and invalid audio errors stay on the affected row',()=>{
  const s=setup();
  const a=s.uploads.add('A',s.file,s.file), b=s.uploads.add('B',s.file,s.file);
  Object.assign(s.requests[0],{status:400,response:{detail:'Use 48 kHz WAV files.'}}); s.requests[0].onload();
  assert.equal(s.uploads.items.get(a).error,'Use 48 kHz WAV files.');
  assert.equal(s.uploads.items.get(b).status,'uploading');
  Object.assign(s.requests[1],{status:401,response:{detail:'Unlock your workspace.'}}); s.requests[1].onload();
  assert.equal(s.unauthorized(),1);
  assert.equal(s.uploads.items.size,2);
});
