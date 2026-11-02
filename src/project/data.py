"""Monthly inputs. date labels the return month; formation_date labels entry."""
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ROOT, load_config


def load_reference():
    candidates = [ROOT / 'data/raw/paper_reference/FuturesReturn_to_post.xls',
                  ROOT / 'FuturesReturn_to_post.xls']
    path = next(p for p in candidates if p.exists())
    d = pd.read_csv(path, sep='\t')
    d['date'] = pd.to_datetime(dict(year=d.Year, month=d.Month, day=1)) + pd.offsets.MonthEnd(0)
    d = d.rename(columns={'Contract':'paper_code','Exc_Ret':'excess_return',
                          'Log_Future_Spot_Ratio':'slope'})
    assert not d.duplicated(['paper_code','date']).any()
    return d[['paper_code','date','excess_return','slope']].sort_values(['paper_code','date'])


def prepare_metadata(metadata, mapping, prefer_actual_notice=True):
    """Normalize contract metadata and derive the safe-holding cutoff.

    Contract-specific vendor FND/LTD fields take precedence.  The published
    Internet-I calendar convention is an explicit fallback, not a replacement
    for verified historical dates.  ``prefer_actual_notice=False`` reproduces
    the repository's former convention-only policy for audit comparisons.
    """
    m = metadata.copy()
    for c in ['lasttrddate','expirationdate','firstnoticedate','startdate']:
        m[c] = pd.to_datetime(m[c], errors='coerce')
    m['expiry'] = m.expirationdate.fillna(m.lasttrddate)
    m['expiry_source'] = np.select(
        [m.expirationdate.notna(), m.lasttrddate.notna()],
        ['expirationdate', 'lasttrddate'], default='missing')
    def maturity(row):
        s = str(row.contrdate)
        if len(s)!=4 or not s.isdigit() or not 1<=int(s[:2])<=12:
            return pd.NaT
        ref = row.expiry if pd.notna(row.expiry) else row.startdate
        if pd.isna(ref):
            return pd.NaT
        year = min([1900+int(s[2:]),2000+int(s[2:]),2100+int(s[2:])],
                   key=lambda y: abs(y-ref.year))
        return pd.Timestamp(year,int(s[:2]),1)
    m['maturity'] = m.apply(maturity,axis=1)
    m = m.merge(mapping.dropna(subset=['contrcode']),on='contrcode',validate='many_to_one')
    from .notice_conventions import notice_convention
    dates=[notice_convention(r.paper_code,r.maturity,'first') for r in m.itertuples()]
    upper=[notice_convention(r.paper_code,r.maturity,'last')[0] for r in m.itertuples()]
    m['convention_date'] = pd.to_datetime([d[0] for d in dates])
    m['convention_range_end'] = pd.to_datetime(upper)
    m['convention_rule'] = [d[1] for d in dates]
    missing_expiry=m.expiry.isna() & m.maturity.notna()
    m.loc[missing_expiry,'expiry']=m.loc[missing_expiry,'maturity']+pd.offsets.MonthEnd(0)
    m.loc[missing_expiry,'expiry_source']='delivery_month_end_assumption'
    cash=m.paper_code.isin(['FC','LH'])
    m['notice_cutoff'] = m.convention_date
    m['notice_source'] = 'table_I_convention_US_federal_calendar_proxy'
    if prefer_actual_notice:
        exact_fnd = ~cash & m.firstnoticedate.notna()
        m.loc[exact_fnd, 'notice_cutoff'] = m.loc[exact_fnd, 'firstnoticedate']
        m.loc[exact_fnd, 'notice_source'] = 'vendor_firstnoticedate'

        # Cash-settled contracts have no FND.  Their actual last trade date is
        # the economically relevant hold-safety boundary.
        exact_cash_ltd = cash & m.lasttrddate.notna()
        m.loc[exact_cash_ltd, 'notice_cutoff'] = m.loc[exact_cash_ltd, 'lasttrddate']
        m.loc[exact_cash_ltd, 'notice_source'] = 'vendor_lasttrddate_cash_settled'
    m['notice_is_exact'] = m.notice_source.isin(
        ['vendor_firstnoticedate', 'vendor_lasttrddate_cash_settled'])
    return m


def monthly_snapshots(folder, prefer_actual_notice=True):
    folder = Path(folder)
    mapping = pd.read_csv(ROOT / 'configs/commodity_map.csv')
    expected = {int(c) for c in mapping.contrcode.dropna()}
    completed = {int(p.stem.split('=')[1]) for p in folder.glob('contrcode=*.parquet')
                 if p.with_suffix('.parquet.json').exists()}
    if expected-completed:
        raise ValueError(f'Incomplete raw download; rerun acquisition: {sorted(expected-completed)}')
    meta = prepare_metadata(
        pd.read_parquet(folder / 'metadata.parquet'), mapping,
        prefer_actual_notice=prefer_actual_notice)
    frames, audits = [], []
    for path in sorted(folder.glob('contrcode=*.parquet')):
        code = int(path.stem.split('=')[1])
        d = pd.read_parquet(path).drop_duplicates()
        d['date'] = pd.to_datetime(d.date)
        conflict = d.duplicated(['futcode','date'],keep=False)
        audits.append(dict(contrcode=code,raw_rows=len(d),conflict_rows=int(conflict.sum()),
                           nonpositive_prices=int(d.settlement.le(0).sum()),
                           missing_prices=int(d.settlement.isna().sum())))
        d = d.loc[~conflict].copy()
        d['month'] = d.date + pd.offsets.MonthEnd(0)
        d = d[d.date.eq(d.groupby('month').date.transform('max'))]
        d = d.merge(meta[meta.contrcode.eq(code)],on='futcode',validate='many_to_one')
        d['quote_date'] = d.date
        d['date'] = d.month
        d['quote_age_days'] = (d.date-d.quote_date).dt.days
        frames.append(d)
    return pd.concat(frames,ignore_index=True), pd.DataFrame(audits), meta


def construct_monthly(snapshot, max_quote_age=None):
    """Rank before safety filtering; select rank >=2; join same futcode at t+1.

    Table Internet-I supplies notice dates. No return filling or future-return
    based source selection. A seven-calendar-day quote-age ceiling is explicit.
    """
    if max_quote_age is None:
        max_quote_age=load_config()['data_preparation']['max_quote_age_days']
    rows = []
    for (code, date), quotes in snapshot.groupby(['contrcode','date'],sort=True):
        q = quotes[quotes.maturity.notna() & quotes.expiry.gt(date) &
                   quotes.settlement.gt(0) & quotes.quote_age_days.le(max_quote_age)]
        q = q.sort_values(['maturity','futcode'])
        if q.maturity.duplicated().any():
            raise ValueError(f'Multiple contracts for delivery month: {code} {date}')
        base = quotes.iloc[0]
        row = dict(paper_code=base.paper_code,commodity=base.commodity,sector=base.sector,
                   contrcode=int(code),priority=base.priority,mapping_status=base.status,
                   formation_date=date,date=date+pd.offsets.MonthEnd(1),
                   selected_futcode=np.nan,entry_price=np.nan,slope=np.nan,
                   reason='insufficient_curve',notice_source='unresolved')
        if len(q)>=2:
            front, second = q.iloc[0], q.iloc[1]
            row.update(front_futcode=int(front.futcode), second_futcode=int(second.futcode),
                       slope=np.log(second.settlement/front.settlement))
            safe = q.iloc[1:].copy()
            safe['contract_rank'] = np.arange(2, len(safe) + 2)
            safe = safe[safe.notice_cutoff.gt(row['date']) & safe.expiry.gt(row['date'])]
            if len(safe):
                chosen = safe.iloc[0]
                row.update(selected_futcode=int(chosen.futcode),entry_price=chosen.settlement,
                           first_notice_date=chosen.firstnoticedate,notice_cutoff=chosen.notice_cutoff,
                           expiry=chosen.expiry,notice_source=chosen.notice_source,
                           notice_is_exact=bool(chosen.notice_is_exact),
                           selected_contract_rank=int(chosen.contract_rank),
                           entry_quote_date=chosen.quote_date,entry_open_interest=chosen.open_interest,
                           reason='selected')
            else:
                row['reason'] = 'no_verified_safe_contract'
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel = panel.sort_values(['paper_code','formation_date','priority'])
    panel = panel.drop_duplicates(['paper_code','formation_date'],keep='first')
    exits = snapshot[['futcode','date','settlement','quote_date','open_interest','quote_age_days']]
    exits = exits.rename(columns={'futcode':'selected_futcode','settlement':'exit_price',
                                  'quote_date':'exit_quote_date','open_interest':'exit_open_interest',
                                  'quote_age_days':'exit_quote_age'})
    panel = panel.merge(exits,on=['selected_futcode','date'],how='left',validate='many_to_one')
    panel['excess_return'] = panel.exit_price/panel.entry_price-1
    invalid = (panel.exit_price.le(0) | panel.exit_quote_age.gt(max_quote_age) |
               panel.mapping_status.ne('verified'))
    panel.loc[invalid,'excess_return'] = np.nan
    panel.loc[panel.reason.eq('selected') & panel.exit_price.isna(),'reason'] = 'missing_exit'
    panel.loc[panel.exit_quote_age.gt(max_quote_age),'reason'] = 'stale_exit'
    panel.loc[panel.exit_price.le(0),'reason'] = 'nonpositive_exit'
    panel.loc[panel.mapping_status.ne('verified'),'reason'] = 'unverified_market_identity'
    return panel[panel.date.le(snapshot.date.max())].sort_values(['paper_code','date'])


def momentum_signal(panel, skip_months=0):
    wide = panel.pivot(index='date',columns='paper_code',values='excess_return')
    wide = wide.reindex(pd.date_range(wide.index.min(),wide.index.max(),freq='ME'))
    signal = (1+wide).shift(1+skip_months).rolling(6,min_periods=6).apply(np.prod,raw=True)-1
    return signal.rename_axis(index='date',columns='paper_code').stack().dropna().rename('momentum').reset_index()


def get_data():
    from .acquisition import connect_wrds, download_futures
    conn = connect_wrds()
    try:
        return download_futures(conn)
    finally:
        conn.close()


def build_dataset(folder=None):
    settings=load_config()['sample']
    folder = folder or ROOT / 'data/raw/wrds_contracts' / f"{settings['download_start']}_{settings['proposed_wrds_end']}"
    out = ROOT / 'data/interim'; out.mkdir(parents=True,exist_ok=True)
    processed = ROOT / 'data/processed'; processed.mkdir(parents=True,exist_ok=True)
    snapshots, audit, metadata = monthly_snapshots(folder)
    snapshots.to_parquet(out/'wrds_month_end_contracts.parquet',index=False,compression='zstd')
    metadata.to_parquet(out/'wrds_metadata_clean.parquet',index=False,compression='zstd')
    audit.to_csv(out/'wrds_quote_audit.csv',index=False)
    panel = construct_monthly(snapshots)
    panel = panel.merge(momentum_signal(panel),on=['paper_code','date'],how='left',validate='one_to_one')
    panel['notice_policy'] = 'vendor_fnd_then_appendix_fallback'
    panel.to_parquet(processed/'commodity_monthly.parquet',index=False,compression='zstd')
    panel[panel.date.between('2000-01-01','2025-12-31')].to_parquet(
        processed/'commodity_monthly_2000_2025.parquet',index=False,compression='zstd')
    load_reference().to_parquet(out/'author_monthly_reference.parquet',index=False,compression='zstd')
    return panel
