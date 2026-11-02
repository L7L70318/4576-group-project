"""Pre-estimation factors and assets. Membership never depends on exit returns."""
import numpy as np
import pandas as pd

from .data import momentum_signal


def make_portfolios(panel, reference=False, skip_months=0, carry_policy='rank'):
    """Construct portfolios using formation information only.

    The author file's slope timestamp is unverified. For reference diagnostics
    only, assume it is observed at the row's calendar month end and lag one month.
    The primary Carry rule ranks the full slope cross-section and selects the
    bottom/top k commodities. ``strict`` remains available for audit comparisons.
    """
    p=panel.drop(columns=['momentum'],errors='ignore').copy()
    p=p.merge(momentum_signal(p,skip_months),on=['paper_code','date'],how='left')
    if reference:
        slope=p.pivot(index='date',columns='paper_code',values='slope')
        slope=slope.reindex(pd.date_range(slope.index.min(),slope.index.max(),freq='ME')).shift(1)
        signals=slope.rename_axis(index='date',columns='paper_code').stack().dropna().rename('formation_slope').reset_index()
        p=p.merge(signals,on=['paper_code','date'],how='left')
    else:
        p['formation_slope']=p.slope
    weights=[]
    def add(date,name,g,scale=1.):
        for code in g.paper_code:
            weights.append({'date': date,'portfolio': name,'paper_code': code,'weight': scale})
    for date,g in p.groupby('date'):
        # Universe at formation, including missing future returns; no ex-post reweighting.
        universe=g[g.mapping_status.eq('verified')] if 'mapping_status' in g else g
        if 'selected_futcode' in universe:
            universe=universe[universe.selected_futcode.notna()]
        universe=universe.sort_values('paper_code')
        if len(universe): add(date,'AVG',universe,1/len(universe))
        for sector,s in universe.groupby('sector') if 'sector' in universe else []:
            add(date,'sector_'+sector,s,1/len(s))
        carry=universe.dropna(subset=['formation_slope']).sort_values(['formation_slope','paper_code'])
        lo=carry[carry.formation_slope.lt(0)]; hi=carry[carry.formation_slope.gt(0)]
        for k in range(1,6):
            if carry_policy == 'rank':
                if len(carry) < 2*k: continue
                carry_long, carry_short = carry.head(k), carry.tail(k)
            elif carry_policy in ['strict','cash']:
                if carry_policy == 'strict' and (len(lo)<k or len(hi)<k): continue
                carry_long, carry_short = lo.head(k), hi.tail(k)
            else:
                raise ValueError(carry_policy)
            add(date,f'C{k}',carry_long,1/k)
            add(date,f'C{k}',carry_short,-1/k)
        # P1 most backwardated; P4 most contango, ranked independently per sign.
        for indices,names in [(np.array_split(np.arange(len(lo)),2),['P1','P2']),
                              (np.array_split(np.arange(len(hi)),2),['P3','P4'])]:
            source=lo if names[0]=='P1' else hi
            for idx,name in zip(indices,names):
                if len(idx): add(date,name,source.iloc[idx],1/len(idx))
        mom=universe.dropna(subset=['momentum']).sort_values(['momentum','paper_code'])
        for k in range(1,6):
            if len(mom)>=2*k:
                add(date,f'M{k}',mom.tail(k),1/k); add(date,f'M{k}',mom.head(k),-1/k)
        if len(mom)>=5:
            for j,idx in enumerate(np.array_split(np.arange(len(mom)),5),1):
                add(date,f'Q{j}',mom.iloc[idx],1/len(idx))
    w=pd.DataFrame(weights)
    joined=w.merge(p[['paper_code','date','excess_return']],on=['paper_code','date'],how='left',validate='many_to_one')
    joined['contribution']=joined.weight*joined.excess_return
    result=joined.groupby(['date','portfolio']).contribution.agg(lambda x:x.sum() if x.notna().all() else np.nan).unstack('portfolio')
    result=result.reindex(pd.date_range(p.date.min(),p.date.max(),freq='ME')).rename_axis('date')
    return result,w
