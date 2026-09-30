#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,base64,glob,hashlib,io,json,math,os,random,socket,time
H=b'RWH1';D=b'RWD1';M=b'RWM1';W=(1<<16)-1;MAG=b'SJMETA01'
def hello(f,s,n):
 b=n.encode();return H+f.to_bytes(2,'big')+s.to_bytes(8,'big')+len(b).to_bytes(2,'big')+b
def meta(f,p):return M+f.to_bytes(2,'big')+len(p).to_bytes(4,'big')+p
def data(f,q,p):return D+f.to_bytes(2,'big')+q.to_bytes(4,'big')+p
def split(r):
 if len(r)>=16 and r[-16:-8]==MAG:
  n=int.from_bytes(r[-8:],'big')
  if 0<n<=len(r)-16:
   try:return r[:-(16+n)],json.loads(r[-(16+n):-16])
   except:pass
 return r,None
def meta_load(r,path):
 _,e=split(r)
 if e is not None:return json.dumps(e,separators=(',',':')).encode()
 p=os.path.splitext(path)[0]+'.meta.json'
 return open(p,'rb').read() if os.path.exists(p) else None
def compact_int8_meta(x):
 """Keep a complete decodable int8 control META when verbose fields exceed the link limit."""
 if x.get('storage',{}).get('format')!='swinjscc_int8_v1':raise ValueError('compact META requires int8 storage')
 required=('image_stem','input_shape','feature_shape','original_symbol_power','num_complex_symbols','dtype','storage','transport')
 if any(k not in x for k in required):raise ValueError('int8 META lacks a required decode/transport field')
 out={k:x[k] for k in required}
 for k in ('model','trainset','model_size','C','rate','snr_db','target_power','cbr'):
  if k in x:out[k]=x[k]
 if 'semantic_uep' in x:
  c=x['semantic_uep']
  keys=('scheme','metric','total_chunks','a_rep','b_rep','tier_a_bits','tier_b_bits')
  if not isinstance(c,dict) or any(k not in c for k in keys):raise ValueError('int8 META lacks complete A/B tier information')
  out['semantic_uep']={k:c[k] for k in keys}
 out['control_meta_compact']=1
 return out
def bits(ids,n):
 x=bytearray((n+7)//8)
 for i in ids:x[i//8]|=1<<(i%8)
 return base64.b64encode(x).decode()
def pace_maker(g):
 if g<=0:return lambda:None
 nxt=time.perf_counter()
 def p():
  nonlocal nxt;nxt+=g;d=nxt-time.perf_counter()
  if d>0:time.sleep(d)
  elif d<-20*g:nxt=time.perf_counter()
 return p
def scores(raw,ch):
 return energy_scores(raw,ch)[0]
def energy_scores(raw,ch,energy_dtype='auto'):
 """Score numeric NPY payload blocks; keep the historical float metric."""
 import numpy as np
 r,_=split(raw);a=np.load(io.BytesIO(r),allow_pickle=False).ravel(order='K')
 if a.dtype.kind not in 'iufc':raise ValueError(f'--importance energy requires a numeric NPY, got {a.dtype}')
 integer=a.dtype.kind in 'iu'
 detected='int' if integer else 'float'
 if energy_dtype not in ('auto','int','float'):raise ValueError(f'Invalid --energy-dtype: {energy_dtype}')
 if energy_dtype!='auto' and energy_dtype!=detected:raise ValueError(f'--energy-dtype {energy_dtype} does not match NPY dtype {a.dtype} ({detected})')
 v=a.astype(np.float64)**2 if integer else np.log1p(np.abs(a).astype(np.float64)**2)
 if not np.all(np.isfinite(v)):raise ValueError('energy NPY contains non-finite power values')
 head=len(r)-a.nbytes;n=(len(raw)+ch-1)//ch;z=[]
 for q in range(n):
  x=max(q*ch,head);y=min((q+1)*ch,len(r));i=max(0,(x-head)//a.itemsize);j=min(a.size,(y-head)//a.itemsize);z.append(float(v[i:j].mean()) if j>i else -1e30)
 metric='int8_mean_square_proxy_v1' if a.dtype==np.int8 else 'integer_mean_square_proxy_v1' if integer else 'mean_log1p_abs2_proxy'
 return z,metric
def sensitivity_scores(raw,path,ch):
 p=os.path.splitext(path)[0]+'.importance.json'
 if not os.path.exists(p):raise SystemExit(f'缺少 sensitivity sidecar: {p}; 请先用 encode_dataset.py --sensitivity-importance 编码')
 try:
  with open(p,'r',encoding='utf-8') as f:x=json.load(f)
  gs=int(x['group_symbols']);ns=int(x['num_symbols']);v=[float(q) for q in x['scores']]
 except Exception as e:raise SystemExit(f'无法读取 {p}: {e}')
 if gs<=0 or ns<=0 or len(v)<(ns+gs-1)//gs:raise SystemExit(f'importance sidecar 无效: {p}')
 body,_=split(raw);head=len(body)-ns*8;n=(len(raw)+ch-1)//ch;out=[]
 for q in range(n):
  lo=max(q*ch,head);hi=min((q+1)*ch,head+ns*8)
  if hi<=lo:out.append(-1e30);continue
  first=(lo-head)//8//gs;last=(hi-head-1)//8//gs
  out.append(sum(v[first:last+1])/(last-first+1))
 return out,x.get('scheme','decoder_gradient_importance')
def erasure_scores(raw,path,ch):
 p=os.path.splitext(path)[0]+'.erasure.json'
 try:
  with open(p,'r',encoding='utf-8') as f:x=json.load(f)
  spec=x['provenance'];v=x['scores'];n=(len(raw)+ch-1)//ch
  if x.get('complete') is not True or spec.get('scheme')!='single_chunk_erasure_delta_mse_v1':raise ValueError('incomplete or unsupported erasure cache')
  if spec.get('source_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('source NPY hash mismatch')
  if spec.get('chunk_bytes')!=ch or spec.get('total_chunks')!=n:raise ValueError('chunk size/count mismatch')
  if not isinstance(v,list) or len(v)!=n or any(isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) for q in v):raise ValueError('invalid score array')
 except Exception as e:raise ValueError(f'Cannot use erasure sidecar {p}: {e}') from e
 return v,spec['scheme']
def gradient_scores(raw,path,ch):
 p=os.path.splitext(path)[0]+'.gradient.json'
 try:
  with open(p,'r',encoding='utf-8') as f:x=json.load(f)
  spec=x['provenance'];v=x['scores'];n=(len(raw)+ch-1)//ch
  if x.get('complete') is not True or spec.get('scheme')!='int8_decoder_gradient_chunk_proxy_v1':raise ValueError('incomplete or unsupported gradient proxy')
  if spec.get('storage_format')!='swinjscc_int8_v1':raise ValueError('gradient proxy requires int8 storage')
  if spec.get('source_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('source NPY hash mismatch')
  if spec.get('chunk_bytes')!=ch or spec.get('total_chunks')!=n:raise ValueError('chunk size/count mismatch')
  if not isinstance(v,list) or len(v)!=n or any(isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or q<0 for q in v):raise ValueError('invalid gradient score array')
  mp=meta_load(raw,path)
  if not mp or json.loads(mp).get('storage',{}).get('format')!='swinjscc_int8_v1':raise ValueError('source NPY is not int8 or has no decodable META')
 except Exception as e:raise ValueError(f'Cannot use gradient sidecar {p}: {e}') from e
 return v,spec['scheme']

def predictor_scores(raw,path,ch):
 p=os.path.splitext(path)[0]+'.predictor.json'
 try:
  with open(p,'r',encoding='utf-8') as f:x=json.load(f)
  spec=x['provenance'];v=x['scores'];n=(len(raw)+ch-1)//ch
  if x.get('complete') is not True or spec.get('scheme')!='int8_erasure_rank_ridge_v1':raise ValueError('incomplete or unsupported predictor sidecar')
  if spec.get('storage_format')!='swinjscc_int8_v1':raise ValueError('predictor requires int8 storage')
  if spec.get('source_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('source NPY hash mismatch')
  if spec.get('chunk_bytes')!=ch or spec.get('total_chunks')!=n:raise ValueError('chunk size/count mismatch')
  if not isinstance(v,list) or len(v)!=n or any(isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) for q in v):raise ValueError('invalid predictor score array')
  mp=meta_load(raw,path)
  if not mp or json.loads(mp).get('storage',{}).get('format')!='swinjscc_int8_v1':raise ValueError('source NPY is not int8 or has no decodable META')
 except Exception as e:raise ValueError(f'Cannot use predictor sidecar {p}: {e}') from e
 return v,spec['scheme']

def plan(raw,ch,fid,mode,tf,mf,ar,br,importance='energy',path=None,energy_dtype='auto'):
 n=(len(raw)+ch-1)//ch;rng=random.Random((fid<<16)^0x53454d);o=list(range(n));t=o[1:];rng.shuffle(t);o=[0]+t
 _,embedded=split(raw)
 if embedded and embedded.get('storage'):
  if embedded['storage'].get('format')!='swinjscc_int8_v1':raise ValueError('Unsupported storage format')
  if mode!='baseline' and importance=='sensitivity':raise ValueError('int8 sensitivity sidecar is not byte-aligned; use --importance energy or erasure')
 if mode=='baseline':return o,[],[],None
 if importance=='sensitivity':sc,metric=sensitivity_scores(raw,path,ch)
 elif importance=='erasure':sc,metric=erasure_scores(raw,path,ch)
 elif importance=='gradient':sc,metric=gradient_scores(raw,path,ch)
 elif importance=='predictor':sc,metric=predictor_scores(raw,path,ch)
 else:sc,metric=energy_scores(raw,ch,energy_dtype)
 rk=sorted(range(n),key=lambda i:(-sc[i],i))
 if importance in ('erasure','gradient','predictor') or (importance=='energy' and metric.endswith('mean_square_proxy_v1')):rk=[0]+[i for i in rk if i!=0]
 na=max(1,round(n*tf));nb=max(0,round(n*mf));A=set(rk[:na]);B=set(rk[na:na+nb]);A.add(0)
 while len(A)>na:A.remove(min((x for x in A if x),key=lambda x:sc[x]))
 B-=A;p1=[x for x in A if ar>1]+[x for x in B if br>1];p2=list(A) if ar>2 else [];rng.shuffle(p1);rng.shuffle(p2)
 note='decoder-gradient sensitivity proxy for reconstruction loss' if importance=='sensitivity' else 'feature-energy proxy; not learned semantic importance'
 if importance=='erasure':note='measured single DATA chunk erasure delta MSE; not multi-loss optimum'
 if importance=='gradient':note='frozen-decoder gradient proxy for int8 DATA chunks; not measured erasure loss'
 if importance=='predictor':note='CPU-fitted erasure rank proxy for int8 DATA chunks; not measured erasure loss'
 if importance=='energy' and metric.endswith('mean_square_proxy_v1'):note='online mean-square integer payload proxy; not decoder-measured importance'
 c={'scheme':mode,'metric':metric,'metric_note':note,'total_chunks':n,'top_fraction':tf,'mid_fraction':mf,'a_rep':ar,'b_rep':br,'tier_a_bits':bits(A,n),'tier_b_bits':bits(B,n),'tier_a_count':len(A),'tier_b_count':len(B),'tier_c_count':n-len(A)-len(B),'header_protected':True}
 return o,p1,p2,c
def send_file(s,h,p,path,a,fid,wire_name=None):
 with open(path,'rb') as source:raw=source.read()
 n=(len(raw)+a.chunk-1)//a.chunk;mp=meta_load(raw,path);o,p1,p2,c=plan(raw,a.chunk,fid,a.mode,a.top_frac,a.mid_frac,a.a_rep,a.b_rep,a.importance,path,getattr(a,'energy_dtype','auto'))
 if c and (a.importance in ('erasure','gradient','predictor') or c['metric'].endswith('mean_square_proxy_v1')) and not mp:raise SystemExit('integer energy/erasure/gradient/predictor mode requires embedded META or a matching .meta.json')
 if c:c['channel_profile']=a.channel_profile
 name=wire_name or os.path.basename(path)
 if mp:
  try:
   x=json.loads(mp.decode())
   if not isinstance(x,dict):raise ValueError('META JSON must be an object')
   if c:x['semantic_uep']=c
   x['transport']={'version':1,'file_id':fid,'file_name':name,'file_bytes':len(raw),'chunk_bytes':a.chunk}
   mp=json.dumps(x,separators=(',',':')).encode()
   if len(mp)>a.meta_max and x.get('storage',{}).get('format')=='swinjscc_int8_v1':
    mp=json.dumps(compact_int8_meta(x),separators=(',',':')).encode()
  except Exception as e:raise SystemExit(f'cannot add transport metadata for {path}: {e}')
 if mp and len(mp)>a.meta_max:raise SystemExit(f'META for {os.path.basename(path)} is {len(mp)} bytes, above --meta-max {a.meta_max}; A/B bitsets were not sent. Increase --meta-max only if the LTE control path accepts that UDP datagram size.')
 hp=hello(fid,len(raw),name);mp=meta(fid,mp) if mp else None;pace=pace_maker(a.gap);cnt=0
 def ctl(k):
  for _ in range(k):s.sendto(hp,(h,p));pace()
  if mp:
   for _ in range(k):s.sendto(mp,(h,p));pace()
 def tx(q):
  nonlocal cnt;s.sendto(data(fid,q,raw[q*a.chunk:(q+1)*a.chunk]),(h,p));pace();cnt+=1
 t=time.time();ctl(a.meta_rep);mid=max(1,len(o)//2)
 for i,q in enumerate(o):
  tx(q)
  if a.meta_every and (i+1)%a.meta_every==0:ctl(1)
  if i+1==mid:
   for q in p1:tx(q)
   if p1:ctl(1)
 for q in p2:tx(q)
 ctl(1);print(f'[已发送] {os.path.basename(path)} chunks={n} DATA={cnt} extra={cnt-n} multiplier={cnt/max(1,n):.2f}x mode={a.mode} time={time.time()-t:.2f}s')
def warm(s,h,p,ch,g,sec):
 if sec<=0:return
 pc=pace_maker(g);e=time.time()+sec;q=0;z=b'\0'*ch
 while time.time()<e:s.sendto(data(W,q,z),(h,p));pc();q+=1
 print('[warmup]',q,'packets')
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--dir',required=True);ap.add_argument('--host',required=True)
 ap.add_argument('--port',type=int,default=50000);ap.add_argument('--chunk',type=int,default=1200)
 ap.add_argument('--gap',type=float,default=.01)
 ap.add_argument('--mode',choices=['baseline','semantic','repeat'],default='semantic')
 ap.add_argument('--importance',choices=['energy','sensitivity','erasure','gradient','predictor'],default='energy',help='energy: numeric NPY proxy; sensitivity: complex64 .importance.json; erasure: measured .erasure.json; gradient: int8 .gradient.json proxy; predictor: int8 .predictor.json rank proxy')
 ap.add_argument('--energy-dtype',choices=['auto','float','int'],default='auto',help='Optional energy input type check: auto reads NPY dtype; float includes complex; int includes signed/unsigned integers')
 ap.add_argument('--channel-profile',choices=['custom','good','medium','bad'],default='custom',help='good=1/1, medium=2/1, bad=3/2')
 ap.add_argument('--top-frac',type=float,default=.10);ap.add_argument('--mid-frac',type=float,default=.10)
 ap.add_argument('--a-rep',type=int,default=2);ap.add_argument('--b-rep',type=int,default=1)
 ap.add_argument('--meta-rep',type=int,default=3);ap.add_argument('--meta-every',type=int,default=180)
 ap.add_argument('--meta-max',type=int,default=1400,help='maximum complete META payload; tier bitsets are never silently removed')
 ap.add_argument('--warmup',type=float,default=2);ap.add_argument('--file-gap',type=float,default=0)
 ap.add_argument('--pattern',default='*.npy');ap.add_argument('--once',action='store_true');ap.add_argument('--resend',action='store_true')
 ap.add_argument('--interval',type=float,default=1)
 ap.add_argument('--rounds',type=int,default=1,help='With --once: repeat dataset using unique round-prefixed HELLO names')
 ap.add_argument('--round-gap',type=float,default=10,help='Seconds between rounds')
 ap.add_argument('--alternate',action='store_true',help='With --once --mode semantic: energy then erasure per round; --rounds counts pairs')
 a=ap.parse_args()
 if a.rounds<1 or not math.isfinite(a.round_gap) or a.round_gap<0:raise SystemExit('Invalid rounds/round-gap')
 if a.rounds>1 and not a.once:raise SystemExit('--rounds > 1 requires --once')
 if a.alternate and (not a.once or a.mode!='semantic'):raise SystemExit('--alternate requires --once --mode semantic')
 schedule=[(i,m) for i in range(1,a.rounds+1) for m in (('energy','erasure') if a.alternate else (a.importance,))]
 grouped=a.rounds>1 or a.alternate
 if a.chunk<=0 or a.chunk>1400:raise SystemExit('--chunk 1..1400')
 if a.meta_max<=0:raise SystemExit('--meta-max must be positive')
 if a.mode!='baseline' and a.chunk%8:raise SystemExit('semantic/repeat requires chunk multiple of 8')
 a.dir=os.path.abspath(a.dir)
 if not os.path.isdir(a.dir):raise SystemExit(f'[错误] 输入目录不存在或不是目录: {a.dir}')
 if a.channel_profile!='custom':
  a.a_rep,a.b_rep={'good':(1,1),'medium':(2,1),'bad':(3,2)}[a.channel_profile]
  print(f'[profile] {a.channel_profile}: A={a.a_rep}x B={a.b_rep}x C=1x')
 s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4<<20);done=set();warmup_done=False;fid=0
 try:
  while True:
   todo=[]
   for x in sorted(glob.glob(os.path.join(a.dir,a.pattern))):
    if not os.path.isfile(x):continue
    st=os.stat(x);k=(x,st.st_size,int(st.st_mtime))
    if a.resend or k not in done:todo.append((x,k))
   if a.once and not todo:raise SystemExit(f'[错误] 目录中没有匹配 {a.pattern} 的待发送文件: {a.dir}（默认不搜索子目录）')
   if todo:
    if a.mode!='baseline' and any(m=='sensitivity' for _,m in schedule):
     missing=[os.path.splitext(x)[0]+'.importance.json' for x,k in todo if not os.path.isfile(os.path.splitext(x)[0]+'.importance.json')]
     if missing:raise SystemExit(f'[错误] 缺少 {len(missing)} 个重要性文件，例如: {missing[0]}；请将同名 .npy 和 .importance.json 一起复制到输入目录')
    if a.mode!='baseline' and any(m=='erasure' for _,m in schedule):
     # Validate every source before sending any warmup or DATA packets.
     for x,k in todo:
      with open(x,'rb') as f:erasure_scores(f.read(),x,a.chunk)
    if a.mode!='baseline' and any(m=='gradient' for _,m in schedule):
     # Reject stale or mismatched sidecars before sending any packets.
     for x,k in todo:
      with open(x,'rb') as f:gradient_scores(f.read(),x,a.chunk)
    if a.mode!='baseline' and any(m=='predictor' for _,m in schedule):
     for x,k in todo:
      with open(x,'rb') as f:predictor_scores(f.read(),x,a.chunk)
    if a.mode!='baseline' and any(m=='energy' for _,m in schedule):
     for x,k in todo:
      with open(x,'rb') as f:energy_scores(f.read(),a.chunk,a.energy_dtype)
    print(f'[待发送] {len(todo)} 个文件，目录={a.dir}',flush=True)
   if grouped and len(todo)*len(schedule)>W:raise SystemExit('Too many files: round batch must not reuse file IDs or warmup ID')
   for run_index,(round_index,method) in enumerate(schedule):
    a.importance=method
    if grouped:
     print(f'[ROUND] {a.importance}_{round_index:02d} files={len(todo)}',flush=True)
     warm(s,a.host,a.port,a.chunk,a.gap,a.warmup)
    elif todo and not warmup_done:warm(s,a.host,a.port,a.chunk,a.gap,a.warmup);warmup_done=True
    for x,k in todo:
     label='baseline' if a.mode=='baseline' else a.importance
     wire_name=f'{label}_{round_index:02d}__{os.path.basename(x)}' if grouped else None
     send_file(s,a.host,a.port,x,a,fid&65535,wire_name);done.add(k);fid+=1
    if run_index+1<len(schedule):time.sleep(a.round_gap)
   if a.once:break
   time.sleep(a.interval)
 except KeyboardInterrupt:pass
 finally:s.close()
if __name__=='__main__':main()
