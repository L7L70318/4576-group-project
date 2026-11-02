"""Auditable comparisons; a benchmark check is not an independent replication."""
import numpy as np
import pandas as pd

from .data import load_reference
from .paths import PROCESSING_EXPLORATION, ROOT


def summary(panel):
    rows=[]
    for code,g in panel.groupby('paper_code'):
        r=g.set_index('date').excess_return.sort_index()
        valid=r.dropna(); sd=valid.std(ddof=1)*np.sqrt(12)
        full=r.reindex(pd.date_range(r.index.min(),r.index.max(),freq='ME'))
        slope=g.slope.dropna()
        rows.append({'paper_code': code,'n': len(valid),'start': valid.index.min(),'end': valid.index.max(),
                         'mean': valid.mean()*1200,'sd': sd*100,'sr': valid.mean()*12/sd if sd>0 else np.nan,
                         'skewness': valid.skew(),'rho1': full.autocorr(1),
                         'contango': slope.gt(0).mean() if len(slope) else np.nan,
                         'contango_including_flat': slope.ge(0).mean() if len(slope) else np.nan,
                         'slope_mean': slope.mean(),'slope_n': len(slope)})
    return pd.DataFrame(rows)


def compare_returns(independent, reference):
    both=independent.merge(reference,on=['paper_code','date'],suffixes=('_wrds','_author'),how='outer')
    rows=[]
    for code,g in both.groupby('paper_code'):
        g=g.dropna(subset=['excess_return_wrds','excess_return_author'])
        err=g.excess_return_wrds-g.excess_return_author
        rows.append({'paper_code': code,'n_common': len(g),
                         'correlation': g.excess_return_wrds.corr(g.excess_return_author) if len(g)>2 else np.nan,
                         'mean_error_bp': err.mean()*1e4,'mae_bp': err.abs().mean()*1e4,
                         'rmse_bp': np.sqrt((err**2).mean())*1e4,
                         'max_error_bp': err.abs().max()*1e4})
    return both,pd.DataFrame(rows)


def run_diagnostics(panel,label='main'):
    out=PROCESSING_EXPLORATION/'diagnostics'/label; out.mkdir(parents=True,exist_ok=True)
    ref=load_reference(); original=panel[panel.date.between('1970-01-01','2011-09-30')]
    summary(ref).to_csv(out/'author_summary.csv',index=False)
    summary(original).to_csv(out/'wrds_original_sample_summary.csv',index=False)
    summary(panel[panel.date.between('2000-01-01','2025-12-31')]).to_csv(out/'wrds_proposal_sample_summary.csv',index=False)
    both,stats=compare_returns(original,ref)
    both.to_parquet(out/'wrds_author_monthly_comparison.parquet',index=False)
    stats.to_csv(out/'wrds_author_return_errors.csv',index=False)
    # Source label alignment is unknown: export every lag, never select a best lag silently.
    lags=[]
    for lag in [-1,0,1]:
        shifted=ref[['paper_code','date','slope']].copy()
        shifted['date']=shifted.date+pd.offsets.MonthEnd(lag)
        joined=original.merge(shifted,on=['paper_code','date'],suffixes=('_wrds','_author'))
        for code,g in joined.groupby('paper_code'):
            g=g.dropna(subset=['slope_wrds','slope_author'])
            lags.append({'paper_code': code,'author_date_shift_months': lag,'n': len(g),
                             'correlation': g.slope_wrds.corr(g.slope_author) if len(g)>2 else np.nan,
                             'mae': (g.slope_wrds-g.slope_author).abs().mean()})
    pd.DataFrame(lags).to_csv(out/'slope_timing_diagnostics.csv',index=False)
    coverage=summary(ref)[['paper_code','n','start','end']].merge(
        summary(original)[['paper_code','n','start','end']],on='paper_code',how='left',suffixes=('_author','_wrds'))
    coverage.to_csv(out/'internet_I_coverage_check.csv',index=False)
    target=ROOT/'configs/internet_II_targets.csv'
    if target.exists():
        target=pd.read_csv(target)
        for source_label,d in [('author',ref),('wrds',original)]:
            checked=summary(d).merge(target,on='paper_code',suffixes=('_computed','_paper'),how='outer')
            for c in ['n','mean','sd','sr','skewness','rho1','contango','slope_mean']:
                checked[c+'_difference']=checked[c+'_computed']-checked[c+'_paper']
            checked['contango_including_flat_difference']=checked.contango_including_flat-checked.contango_paper
            checked.to_csv(out/f'internet_II_{source_label}_check.csv',index=False)
    return stats


def overlap_validation(panel):
    """Same commodity-month pairs only; returns in percent, errors in bp."""
    out=PROCESSING_EXPLORATION/'diagnostics'/'overlap_validation'
    out.mkdir(parents=True,exist_ok=True)
    ref=load_reference()
    p=panel[panel.date.between(ref.date.min(),ref.date.max())]
    d=p.merge(ref,on=['paper_code','date'],suffixes=('_wrds','_author'),validate='one_to_one')
    d=d.dropna(subset=['excess_return_wrds','excess_return_author']).copy()
    d['error_bp']=(d.excess_return_wrds-d.excess_return_author)*10000
    d['abs_error_bp']=d.error_bp.abs()
    d.to_parquet(out/'paired_observations.parquet',index=False)
    d.sort_values('abs_error_bp',ascending=False).to_csv(out/'ranked_errors.csv',index=False)
    def describe(g,code):
        rows=[]
        for source,col in [('author','excess_return_author'),('wrds','excess_return_wrds')]:
            s=g[col]*100
            rows.append({'paper_code': code,'source': source,'n': s.count(),'mean_percent': s.mean(),
                'std_percent': s.std(),'min_percent': s.min(),'q25_percent': s.quantile(.25),
                'median_percent': s.median(),'q75_percent': s.quantile(.75),'max_percent': s.max(),
                'skewness': s.skew(),'excess_kurtosis': s.kurt()})
        return rows
    pooled=pd.DataFrame(describe(d,'ALL'))
    pooled.to_csv(out/'pooled_descriptive_statistics.csv',index=False)
    rows=[]
    for code,g in d.groupby('paper_code'):
        rows.extend(describe(g,code))
    pd.DataFrame(rows).to_csv(out/'commodity_descriptive_statistics.csv',index=False)
    _,stats=compare_returns(p,ref)
    stats.to_csv(out/'commodity_errors.csv',index=False)
    timing=[]
    for shift in [-1,0,1]:
        shifted=ref.copy(); shifted['date']=shifted.date+pd.offsets.MonthEnd(shift)
        j=p.merge(shifted,on=['paper_code','date'],suffixes=('_wrds','_author')).dropna(
            subset=['excess_return_wrds','excess_return_author'])
        timing.append({'author_date_shift': shift,'n': len(j),
            'correlation': j.excess_return_wrds.corr(j.excess_return_author),
            'mae_bp': (j.excess_return_wrds-j.excess_return_author).abs().mean()*10000})
    pd.DataFrame(timing).to_csv(out/'return_date_diagnostic.csv',index=False)
    return d,pooled,stats
