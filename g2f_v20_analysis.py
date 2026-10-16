#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, traceback, math, time
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

ROOT=Path('results_v20'); ROOT.mkdir(exist_ok=True)
RAW=Path('raw_v20'); RAW.mkdir(exist_ok=True)
LOG=[]
def log(x): print(x,flush=True); LOG.append(str(x))
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def dl(urls,name,min_bytes=100):
    p=RAW/name; errs=[]
    for u in urls:
        try:
            log(f'DOWNLOAD {name} <- {u}')
            req=Request(u,headers={'User-Agent':'Mozilla/5.0'})
            with urlopen(req,timeout=180) as r, open(p,'wb') as w:
                while True:
                    b=r.read(1<<20)
                    if not b: break
                    w.write(b)
            if p.stat().st_size<min_bytes: raise RuntimeError(f'too small {p.stat().st_size}')
            log(f'OK {name}: {p.stat().st_size:,} bytes sha256={sha256(p)}'); return p
        except Exception as e:
            errs.append(f'{u}: {e}')
            try:p.unlink()
            except:pass
    raise RuntimeError('download failed '+name+'\n'+'\n'.join(errs))

TRAIT_URL='https://raw.githubusercontent.com/imbaterry11/UFEED_Maize_GxE/refs/heads/main/1_Training_Trait_Data_2014_2023.csv'
GENO_URL='https://raw.githubusercontent.com/imbaterry11/UFEED_Maize_GxE/refs/heads/main/5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv'
OBS24_URLS=[
'https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Testing_data/7_Testing_Observed_Values.csv',
'https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2024/Testing_data/7_Testing_Observed_Values.csv']
EXPECTED_SHA={'trait':'f206848428c9e7ff330716b0fc46ed78590deeb6dd38433b0b1b23a0390a21fb','obs24':'8bd354c44bce50abd09a9839272112fdb34bdb96ec6e37ae644eda916aeeabcc'}

def load_genotype(path):
    log('Reading genotype matrix...'); g=pd.read_csv(path,low_memory=False)
    log(f'genotype raw shape={g.shape}, first columns={g.columns[:5].tolist()}')
    if g.shape[0]<g.shape[1]:
        mat=g.iloc[:,1:].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32).T
        ids=[str(c) for c in g.columns[1:]]; markers=g.iloc[:,0].astype(str).tolist(); orient='markers_x_hybrids_transposed'
    else:
        ids=g.iloc[:,0].astype(str).tolist(); mat=g.iloc[:,1:].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
        markers=[str(c) for c in g.columns[1:]]; orient='hybrids_x_markers'
    keep=np.isfinite(mat).any(0); mat=mat[:,keep]; markers=[m for m,k in zip(markers,keep) if k]
    vals=mat[np.isfinite(mat)]; q=np.unique(vals[:min(len(vals),200000)]) if len(vals) else np.array([])
    log(f'parsed genotype {mat.shape}, orientation={orient}, sampled values={q[:20]}')
    if np.nanmax(mat)<=1.000001: mat*=2.0
    return ids,markers,mat,orient

def marker_design(ids,M,train_ids,all_needed):
    pos={h:i for i,h in enumerate(ids)}; miss=[h for h in all_needed if h not in pos]
    if miss: raise RuntimeError(f'{len(miss)} hybrids absent from genotype matrix; examples {miss[:10]}')
    tr_unique=sorted(set(train_ids)); A=M[[pos[h] for h in tr_unique]].astype(float)
    mu=np.nanmean(A,0); mu[~np.isfinite(mu)]=0; A=np.where(np.isfinite(A),A,mu); sd=A.std(0); keep=sd>1e-8
    mu=mu[keep]; sd=sd[keep]; need=sorted(set(all_needed)); B=M[[pos[h] for h in need]][:,keep].astype(float); B=np.where(np.isfinite(B),B,mu)
    return need,(B-mu)/sd,int(keep.sum())

def safe_cat(s): return s.fillna('__NA__').astype(str)
def stage1_blue_env(g):
    g=g.copy(); g=g[np.isfinite(g.y)]; hyb=safe_cat(g.genotype); hlev=pd.Index(sorted(hyb.unique())); hmap={x:i for i,x in enumerate(hlev)}
    n=len(g); H=len(hlev); rows=[]; cols=[]; data=[]; rr=np.arange(n); hi=np.array([hmap[x] for x in hyb],int)
    rows.extend(rr); cols.extend(hi); data.extend(np.ones(n)); off=H; nuis=[]
    for col in ('rep','block'):
        s=safe_cat(g[col]); lev=pd.Index(sorted(s.unique()))
        if 1<len(lev)<max(3,int(.45*n)):
            mp={x:i for i,x in enumerate(lev[1:])}; mask=s.isin(mp); rid=np.flatnonzero(mask.to_numpy()); cid=np.array([mp[x] for x in s.iloc[rid]])+off
            rows.extend(rid); cols.extend(cid); data.extend(np.ones(len(rid))); nuis.append((col,len(lev))); off+=len(lev)-1
    X=sparse.csr_matrix((data,(rows,cols)),shape=(n,off)); y=g.y.to_numpy(float); sol=lsqr(X,y,atol=1e-10,btol=1e-10,iter_lim=max(1000,4*off))[0]
    vals=sol[:H]; vals=vals-vals.mean()+y.mean(); resid=y-X@sol; sig=float(np.sqrt(np.mean(resid**2))); cnt=hyb.value_counts()
    out=pd.DataFrame({'genotype':hlev.astype(str),'blue':vals}); out['n_plots']=out.genotype.map(cnt).astype(int); out['blue_se_proxy']=sig/np.sqrt(out.n_plots)
    return out,{'n':n,'n_hybrid':H,'nuisance':nuis,'resid_rms':sig,'status':'ok'}
def stage1_blup_env(g):
    import statsmodels.api as sm
    g=g.copy(); g=g[np.isfinite(g.y)]; g['genotype']=safe_cat(g.genotype); X=pd.DataFrame({'Intercept':np.ones(len(g))},index=g.index); nuis=[]
    for col in ('rep','block'):
        s=safe_cat(g[col]); lev=s.nunique()
        if 1<lev<max(3,int(.25*len(g))): X=pd.concat([X,pd.get_dummies(s,prefix=col,drop_first=True,dtype=float)],axis=1); nuis.append((col,int(lev)))
    try:
        fit=sm.MixedLM(g.y.to_numpy(float),X.to_numpy(float),groups=g.genotype.to_numpy(str)).fit(reml=True,method='lbfgs',maxiter=200,disp=False)
        base=float(np.asarray(fit.fe_params)[0]); re=fit.random_effects; vals=[]
        for h in sorted(g.genotype.unique()):
            rv=np.asarray(re[h]).ravel(); vals.append((h,base+(float(rv[0]) if len(rv) else 0.0)))
        out=pd.DataFrame(vals,columns=['genotype','blup']); out['n_plots']=out.genotype.map(g.genotype.value_counts()).astype(int)
        return out,{'status':'ok','converged':bool(getattr(fit,'converged',True)),'n':len(g),'n_hybrid':g.genotype.nunique(),'nuisance':nuis,'var_hybrid':float(np.asarray(fit.cov_re)[0,0]),'var_resid':float(fit.scale)}
    except Exception as e:
        b,_=stage1_blue_env(g); return b.rename(columns={'blue':'blup'})[['genotype','blup','n_plots']],{'status':'fallback_blue','error':repr(e),'n':len(g),'n_hybrid':g.genotype.nunique(),'nuisance':nuis}
def run_stage1(plot):
    B=[]; U=[]; meta=[]; envs=list(plot.env.unique()); t0=time.time()
    for j,e in enumerate(envs,1):
        g=plot[plot.env==e]; b,mb=stage1_blue_env(g); b['env']=e; B.append(b); u,mu=stage1_blup_env(g); u['env']=e; U.append(u); meta.append({'env':e,'year':int(g.year.iloc[0]),'blue':mb,'blup':mu})
        if j%20==0: log(f'stage1 {j}/{len(envs)} elapsed={time.time()-t0:.1f}s')
    return pd.concat(B,ignore_index=True),pd.concat(U,ignore_index=True),meta

def within_center(F,y,env):
    codes,uni=pd.factorize(env,sort=True); ng=len(uni); cnt=np.bincount(codes,minlength=ng).astype(float); yc=y-(np.bincount(codes,weights=y,minlength=ng)/cnt)[codes]
    Fc=F.copy()
    for j in range(F.shape[1]): Fc[:,j]-=(np.bincount(codes,weights=F[:,j],minlength=ng)/cnt)[codes]
    return Fc,yc,ng
def reml_ridge_fit(F,y,env):
    Fc,yc,ng=within_center(F.astype(float),y.astype(float),np.asarray(env)); G=Fc.T@Fc; d,V=eigh(G,check_finite=False,overwrite_a=True); b=V.T@(Fc.T@yc)
    mx=max(float(d.max()),1e-12); keep=d>1e-8*mx; dk=d[keep]; c2=(b[keep]**2)/dk; yy=float(yc@yc); rest=max(yy-float(c2.sum()),1e-12); n=max(len(y)-ng,1)
    def obj(ld):
        delta=math.exp(ld); return n*math.log(float((c2/(1+delta*dk)).sum()+rest))+float(np.log1p(delta*dk).sum())
    res=minimize_scalar(obj,bounds=(math.log(1e-6),math.log(1e4)),method='bounded',options={'xatol':1e-7}); delta=math.exp(float(res.x)); lam=1/delta
    scale=float(np.trace(G))/G.shape[0]; beta=V@(b/(d+lam)); s2e=float((c2/(1+delta*dk)).sum()+rest)/n; vg=delta*s2e*scale*len(d)/n
    return beta,{'delta':delta,'lambda':lam,'lambda_rel':lam/scale,'s2e':s2e,'h2':vg/(vg+s2e),'n':len(y),'n_env':ng,'p':F.shape[1],'success':bool(res.success)}
def fit_predict(train,target,geno_ids,M,ycol,method):
    all_needed=list(train.genotype.astype(str))+list(target.genotype.astype(str)); need,Z,p=marker_design(geno_ids,M,train.genotype.astype(str),all_needed); pos={h:i for i,h in enumerate(need)}
    Fi=Z[[pos[h] for h in train.genotype.astype(str)]]; Ft=Z[[pos[h] for h in target.genotype.astype(str)]]; log(f'Fitting {method}: n={len(train):,}, p={p:,}, env={train.env.nunique()}, target={len(target):,}')
    beta,meta=reml_ridge_fit(Fi,train[ycol].to_numpy(float),train.env.astype(str).to_numpy()); out=target[['env','year','location','genotype']].copy(); out['method']=method; out['pred']=Ft@beta; return out,meta

def top_metrics(df,ycol):
    rows=[]
    for e,g in df.groupby('env'):
        g=g.dropna(subset=[ycol,'pred']); n=len(g)
        if n<3: continue
        y=g[ycol].to_numpy(float); p=g.pred.to_numpy(float); rmse=float(np.sqrt(np.mean(((p-p.mean())-(y-y.mean()))**2))); sp=float(spearmanr(y,p).statistic)
        dec=max(1,int(round(.1*n))); oi10=set(np.argsort(-y,kind='mergesort')[:dec]); si10=set(np.argsort(-p,kind='mergesort')[:dec]); cap=len(oi10&si10)/dec
        for k in (5,10,20):
            if n<k: continue
            oi=np.argsort(-y,kind='mergesort')[:k]; si=np.argsort(-p,kind='mergesort')[:k]
            rows.append({'env':e,'year':int(g.year.iloc[0]),'n':n,'k':k,'rmse_centered':rmse,'spearman':sp,'top_decile_capture':cap,'oracle_mean':float(y[oi].mean()),'selected_mean':float(y[si].mean()),'selection_loss':float(y[oi].mean()-y[si].mean())})
    return pd.DataFrame(rows)
def summarize(E,method,outcome):
    return E.groupby(['year','k'],as_index=False).agg(n_env=('env','nunique'),selection_loss=('selection_loss','mean'),selected_mean=('selected_mean','mean'),rmse_centered=('rmse_centered','mean'),spearman=('spearman','median'),top_decile_capture=('top_decile_capture','mean')).assign(method=method,outcome=outcome)

def main():
    trait=dl([TRAIT_URL],'1_Training_Trait_Data_2014_2023.csv',10_000_000); geno=dl([GENO_URL],'5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv',10_000_000); obs24=dl(OBS24_URLS,'7_Testing_Observed_Values.csv',100_000)
    hashes={'trait':sha256(trait),'genotype_mirror':sha256(geno),'obs24':sha256(obs24)}; (ROOT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    for k in ('trait','obs24'):
        if hashes[k]!=EXPECTED_SHA[k]: log(f'WARNING source hash {k} differs from manuscript manifest: {hashes[k]} vs {EXPECTED_SHA[k]}')
    t=pd.read_csv(trait,low_memory=False); req=['Env','Year','Field_Location','Hybrid','Yield_Mg_ha','Replicate','Block','Plot']; missing=[c for c in req if c not in t.columns]
    if missing: raise RuntimeError('trait missing '+str(missing))
    plot=pd.DataFrame({'env':t.Env.astype(str),'year':pd.to_numeric(t.Year,errors='coerce'),'location':t.Field_Location.astype(str),'genotype':t.Hybrid.astype(str),'y':pd.to_numeric(t.Yield_Mg_ha,errors='coerce'),'rep':t.Replicate,'block':t.Block,'plot':t.Plot}); plot=plot[np.isfinite(plot.y)&np.isfinite(plot.year)].copy(); plot.year=plot.year.astype(int)
    geno_ids,markers,M,orient=load_genotype(geno); gset=set(geno_ids); plot=plot[plot.genotype.isin(gset)].reset_index(drop=True); log(f'plot finite genotyped={len(plot):,}, env={plot.env.nunique()}')
    rawmean=plot.groupby(['env','year','location','genotype'],as_index=False).agg(y_mean=('y','mean'),n_plots=('y','size')); target22=rawmean[rawmean.year==2022].copy()
    if len(target22)!=11802 or target22.env.nunique()!=28: raise RuntimeError(f'2022 universe mismatch {len(target22)}/{target22.env.nunique()}')
    o=pd.read_csv(obs24); o.Yield_Mg_ha=pd.to_numeric(o.Yield_Mg_ha,errors='coerce'); o=o[np.isfinite(o.Yield_Mg_ha)]; target24=pd.DataFrame({'env':o.Env.astype(str),'year':2024,'location':o.Env.astype(str).str.rsplit('_',n=1).str[0],'genotype':o.Hybrid.astype(str),'y_mean':o.Yield_Mg_ha.astype(float)}); target24=target24[target24.genotype.isin(gset)]
    if len(target24)!=9486 or target24.env.nunique()!=22: raise RuntimeError(f'2024 universe mismatch {len(target24)}/{target24.env.nunique()}')
    log('Target universe checks passed: 2022=11802/28; 2024=9486/22')
    blue,blup,smeta=run_stage1(plot); envyear=plot[['env','year','location']].drop_duplicates('env'); blue=blue.merge(envyear,on='env'); blup=blup.merge(envyear,on='env'); blue.to_csv(ROOT/'stage1_blue_all.csv',index=False); blup.to_csv(ROOT/'stage1_blup_all.csv',index=False); (ROOT/'stage1_meta.json').write_text(json.dumps(smeta,indent=1,default=str))
    sm=pd.DataFrame([{'env':x['env'],'year':x['year'],'blup_status':x['blup']['status'],'blup_converged':x['blup'].get('converged',False)} for x in smeta]); sm.to_csv(ROOT/'stage1_status.csv',index=False); log('stage1 BLUP status '+str(sm.blup_status.value_counts().to_dict()))
    target22=target22.merge(blue[blue.year==2022][['env','genotype','blue']].rename(columns={'blue':'y_blue'}),on=['env','genotype'],how='left').merge(blup[blup.year==2022][['env','genotype','blup']].rename(columns={'blup':'y_blup'}),on=['env','genotype'],how='left')
    preds=[]; metas=[]
    for Y,target in [(2022,target22),(2024,target24)]:
        trB=blue[blue.year<Y][['env','year','location','genotype','blue']].rename(columns={'blue':'y_stage1'}).dropna(); p,m=fit_predict(trB,target,geno_ids,M,'y_stage1',f'reml_gblup_blue_Y{Y}'); preds.append(p); metas.append({'target':Y,'stage1':'BLUE','max_train_year':int(trB.year.max()),'n_train':len(trB),'meta':m})
        trU=blup[blup.year<Y][['env','year','location','genotype','blup']].rename(columns={'blup':'y_stage1'}).dropna(); p,m=fit_predict(trU,target,geno_ids,M,'y_stage1',f'reml_gblup_blup_Y{Y}'); preds.append(p); metas.append({'target':Y,'stage1':'BLUP','max_train_year':int(trU.year.max()),'n_train':len(trU),'meta':m})
    P=pd.concat(preds,ignore_index=True); P.to_csv(ROOT/'gblup_predictions.csv',index=False); (ROOT/'gblup_meta.json').write_text(json.dumps(metas,indent=2,default=float))
    envs=[]; sums=[]
    for method,gp in P.groupby('method'):
        Y=int(gp.year.iloc[0]); tar=target22 if Y==2022 else target24; m=tar.merge(gp[['env','genotype','pred']],on=['env','genotype'])
        E=top_metrics(m,'y_mean'); E['method']=method; E['outcome']='mean'; envs.append(E); sums.append(summarize(E,method,'mean'))
        if Y==2022:
            for oc in ('y_blue','y_blup'):
                E=top_metrics(m,oc); E['method']=method; E['outcome']=oc; envs.append(E); sums.append(summarize(E,method,oc))
    Es=pd.concat(envs,ignore_index=True); S=pd.concat(sums,ignore_index=True); Es.to_csv(ROOT/'environment_metrics_gblup.csv',index=False); S.to_csv(ROOT/'summary_gblup.csv',index=False); target22.to_csv(ROOT/'target_2022_outcomes_stage1.csv',index=False); target24.to_csv(ROOT/'target_2024_outcomes.csv',index=False)
    audit={'genotype_orientation':orient,'genotype_shape':[len(geno_ids),len(markers)],'source_hashes':hashes,'target_counts':{'2022_pairs':len(target22),'2022_envs':int(target22.env.nunique()),'2024_pairs':len(target24),'2024_envs':int(target24.env.nunique())},'stage1_blup_status':sm.blup_status.value_counts().to_dict(),'models':metas}; (ROOT/'audit.json').write_text(json.dumps(audit,indent=2,default=float)); (ROOT/'run.log').write_text('\n'.join(LOG)+'\n'); print(S.to_string(index=False))
if __name__=='__main__':
    try: main()
    except Exception:
        (ROOT/'run.log').write_text('\n'.join(LOG)+'\n'+traceback.format_exc()); traceback.print_exc(); raise
