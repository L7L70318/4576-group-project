"""Public auxiliary snapshots; no undisclosed substitution for the G20 proxy."""
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .acquisition import record
from .paths import ROOT


def download_equities(conn,start='1969-01-01',end='2025-12-31'):
    from .acquisition import query
    mapping=pd.read_csv(ROOT/'configs/equity_map.csv')
    out=ROOT/'data/raw/equity_indices'; out.mkdir(parents=True,exist_ok=True)
    for row in mapping.itertuples():
        path=out/f'{row.dsindexmnem}.parquet'
        if path.exists() and path.with_suffix('.parquet.json').exists(): continue
        d=query(conn,'SELECT dsindexcode,valuedate,pi_,ri FROM tr_ds_equities.ds2indexdata '
                'WHERE dsindexcode=%s AND valuedate BETWEEN %s AND %s ORDER BY valuedate',
                (int(row.dsindexcode),start,end))
        d['dsindexcode']=pd.to_numeric(d.dsindexcode).astype('int64')
        for col in ['pi_','ri']:
            d[col]=pd.to_numeric(d[col],errors='coerce').astype('float64')
        d['valuedate']=pd.to_datetime(d.valuedate)
        d.to_parquet(path,index=False,compression='zstd')
        record(path,source='tr_ds_equities.ds2indexdata',indexcode=row.dsindexcode,
               start=start,end=end,rows=len(d))
        print(row.country,len(d),flush=True)
    return out


def prepare_equities():
    import numpy as np
    mapping=pd.read_csv(ROOT/'configs/equity_map.csv')
    raw=ROOT/'data/raw/equity_indices'
    processed=ROOT/'data/processed'; processed.mkdir(parents=True,exist_ok=True)
    interim=ROOT/'data/interim'; interim.mkdir(parents=True,exist_ok=True)
    frames=[]
    for row in mapping.itertuples():
        d=pd.read_parquet(raw/f'{row.dsindexmnem}.parquet').drop_duplicates()
        d['pi_']=pd.to_numeric(d.pi_,errors='coerce').astype('float64')
        if d.duplicated('valuedate').any(): raise ValueError(f'Conflicting equity quotes: {row.country}')
        d=d.sort_values('valuedate'); d['country']=row.country
        d['return']=d.pi_.pct_change(fill_method=None)
        d.loc[d.valuedate.diff().dt.days.gt(7) | d.pi_.le(0),'return']=np.nan
        frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    d.to_parquet(processed/'equity_country_daily.parquet',index=False)
    d['abs_return']=d['return'].abs()
    daily=d.groupby('valuedate').agg(avg_abs_return=('abs_return','mean'),
                  countries=('return','count'),equal_weight_return=('return','mean'))
    daily['month']=daily.index+pd.offsets.MonthEnd(0)
    monthly=daily.groupby('month').agg(vol_equity_g20_proxy=('avg_abs_return','mean'),
        countries_mean=('countries','mean'),countries_min=('countries','min'),days=('avg_abs_return','count'),
        vol_proposal_ew_proxy=('equal_weight_return',lambda x:np.sqrt((x**2).sum(min_count=1))))
    monthly['delta_vol_equity_g20_proxy']=monthly.vol_equity_g20_proxy.diff()
    monthly['delta_vol_proposal_ew_proxy']=monthly.vol_proposal_ew_proxy.diff()
    monthly.rename_axis('date').to_parquet(processed/'equity_volatility_monthly.parquet')
    d.groupby('country').agg(start=('valuedate','min'),end=('valuedate','max'),
        returns=('return','count')).to_csv(interim/'equity_coverage.csv')
    return monthly


def fetch(url, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.with_suffix(path.suffix+'.json').exists():
        return path
    with urlopen(Request(url,headers={'User-Agent':'Academic commodity replication'}),timeout=90) as r:
        content=r.read()
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_bytes(content); tmp.replace(path)
    record(path,source=url)
    return path


def download_public(start_year=1999,end_year=2025):
    out=ROOT/'data/raw/auxiliary'
    status=[]
    jobs=[('vix','https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',out/'vix.csv')]
    for series in ['USREC','INDPRO','DTWEXBGS','TEDRATE','TB3MS','BAA','AAA']:
        jobs.append((series,'https://fred.stlouisfed.org/graph/?'+urlencode({'g':'','id':series,'cosd':'1969-01-01','coed':'2025-12-31'})+'&mode=fred',out/f'{series}.csv'))
    # Use FRED's CSV endpoint rather than graph HTML.
    jobs=[(name, url.replace('/graph/?','/graph/fredgraph.csv?'),path) for name,url,path in jobs]
    fields=['market_and_exchange_names','report_date_as_yyyy_mm_dd','cftc_contract_market_code',
            'open_interest_all','noncomm_positions_long_all','noncomm_positions_short_all',
            'noncomm_postions_spread_all','comm_positions_long_all','comm_positions_short_all']
    for year in range(start_year,end_year+1):
        params={'$select':','.join(fields),'$where':f"report_date_as_yyyy_mm_dd >= '{year}-01-01T00:00:00' AND report_date_as_yyyy_mm_dd < '{year+1}-01-01T00:00:00'",
                '$limit':50000,'$order':'report_date_as_yyyy_mm_dd,cftc_contract_market_code'}
        jobs.append((f'cot_{year}','https://publicreporting.cftc.gov/resource/6dca-aqww.csv?'+urlencode(params),out/'cftc'/f'{year}.csv'))
    for name,url,path in jobs:
        try:
            fetch(url,path)
            d=pd.read_csv(path)
            if name.startswith('cot_') and len(d)>=50000:
                raise ValueError('Pagination cap hit; refusing truncated COT archive')
            status.append({'dataset': name,'status': 'downloaded','rows': len(d)})
            print(name,len(d),flush=True)
        # Keep independent successful snapshots when one public endpoint fails.
        except Exception as e:  # noqa: BLE001
            status.append({'dataset': name,'status': 'failed','error': f'{type(e).__name__}: {e}'})
            print(name,'failed',type(e).__name__,flush=True)
    pd.DataFrame(status).to_csv(out/'download_status.csv',index=False)
    return pd.DataFrame(status)


def prepare_public():
    raw=ROOT/'data/raw/auxiliary'; out=ROOT/'data/processed'; out.mkdir(parents=True,exist_ok=True)
    result={}
    if (raw/'vix.csv').exists():
        d=pd.read_csv(raw/'vix.csv'); d['date']=pd.to_datetime(d.DATE)
        d=d[d.date.between('1969-01-01','2025-12-31')].sort_values('date')
        d.to_parquet(out/'vix_daily.parquet',index=False)
        d['month']=d.date+pd.offsets.MonthEnd(0)
        m=d.groupby('month').tail(1)[['month','date','CLOSE']].rename(columns={'month':'date','date':'observation_date','CLOSE':'vix'})
        # Store raw level, NOT full-sample standardized z (would leak in OOS).
        m.to_parquet(out/'vix_monthly.parquet',index=False); result['vix']=m
    files=sorted((raw/'cftc').glob('*.csv'))
    if files:
        d=pd.concat([pd.read_csv(p,dtype={'cftc_contract_market_code':'string'}) for p in files],ignore_index=True)
        d['report_date']=pd.to_datetime(d.report_date_as_yyyy_mm_dd)
        d['market_code']=d.cftc_contract_market_code.str.zfill(6)
        if d.duplicated(['market_code','report_date']).any():
            raise ValueError('Duplicate CFTC market/report keys')
        d.to_parquet(out/'cftc_legacy_weekly.parquet',index=False,compression='zstd')
        d.groupby('market_code').agg(name=('market_and_exchange_names','last'),n=('report_date','size'),
                                    start=('report_date','min'),end=('report_date','max')).to_csv(raw/'cftc_market_catalog.csv')
        result['cot']=d
        mapping=pd.read_csv(ROOT/'configs/cftc_map.csv',dtype={'market_code':'string'})
        mapped=d.merge(mapping,on='market_code',validate='many_to_one')
        for col in ['open_interest_all','noncomm_positions_long_all','noncomm_positions_short_all',
                    'noncomm_postions_spread_all','comm_positions_long_all','comm_positions_short_all']:
            mapped[col]=pd.to_numeric(mapped[col],errors='coerce')
        oi=mapped.open_interest_all.where(mapped.open_interest_all.gt(0))
        ncl=mapped.noncomm_positions_long_all; ncs=mapped.noncomm_positions_short_all
        cl=mapped.comm_positions_long_all; cs=mapped.comm_positions_short_all
        denom=(cl+cs).where((cl+cs).gt(0))
        mapped['net_noncommercial_share']=(ncl-ncs)/oi
        mapped['noncommercial_share_proposal']=(ncl+ncs)/(2*oi)
        mapped['hedging_pressure']=(cs-cl)/denom
        mapped['workings_t']=1+pd.Series(__import__('numpy').where(cs.ge(cl),ncs,ncl),index=mapped.index)/denom
        mapped['month']=mapped.report_date+pd.offsets.MonthEnd(0)
        monthly=mapped.sort_values('report_date').groupby(['paper_code','month']).tail(1).copy()
        monthly['availability_status']='report_date_only_release_date_not_verified'
        monthly.to_parquet(out/'cftc_commodity_monthly_report_date.parquet',index=False)
        aggregate=monthly.groupby('month').agg(workings_t=('workings_t','mean'),
            noncommercial_share_proposal=('noncommercial_share_proposal','mean'),
            hedging_pressure=('hedging_pressure','mean'),n_commodities=('paper_code','nunique'))
        aggregate.rename_axis('date').to_parquet(out/'cftc_aggregate_monthly_report_date.parquet')
        # Release dates must be supplied from official calendars/backlog notices.
        # Do NOT assume report Tuesday + 3 days (holidays and 2025 backlog).
        release_path=ROOT/'configs/cftc_release_dates.csv'
        if release_path.exists():
            from .positioning import align_releases
            available=align_releases(mapped,pd.read_csv(release_path,parse_dates=['report_date','release_date']))
            available.to_parquet(out/'cftc_commodity_monthly_available.parquet',index=False)
    macro=[]
    for series in ['USREC','INDPRO','DTWEXBGS','TEDRATE','TB3MS','BAA','AAA']:
        path=raw/f'{series}.csv'
        if not path.exists(): continue
        d=pd.read_csv(path); d.columns=['observation_date',series]
        d.observation_date=pd.to_datetime(d.observation_date)
        d[series]=pd.to_numeric(d[series],errors='coerce')
        d['date']=d.observation_date+pd.offsets.MonthEnd(0)
        d=d.dropna(subset=[series]).sort_values('observation_date').groupby('date').tail(1)
        macro.append(d.set_index('date')[[series]])
    if macro:
        pd.concat(macro,axis=1).sort_index().to_parquet(out/'macro_monthly_revised.parquet')
    return result
