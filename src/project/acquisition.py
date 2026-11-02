"""Read-only WRDS acquisition with immutable, resumable Parquet snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import ROOT, load_config

SCHEMA = 'tr_ds_fut'


def connect_wrds(pgpass=None):
    """libpq reads the password; never put credentials into a notebook output."""
    import psycopg2
    default_pgpass = Path.home() / '.pgpass'
    if not default_pgpass.exists():
        default_pgpass = Path.home() / 'AppData/Roaming/postgresql/pgpass.conf'
    path = Path(pgpass or os.environ.get('PGPASSFILE', default_pgpass))
    entries = [re.split(r'(?<!\\):', line) for line in path.read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    entry = next((x for x in entries if len(x) == 5 and
                  x[0] in ('wrds-pgdata.wharton.upenn.edu', '*') and
                  x[1] in ('9737', '*') and x[2] in ('wrds', '*')), None)
    if entry is None:
        raise ValueError('No matching WRDS entry in PGPASSFILE.')
    conn = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
                           dbname='wrds', user=entry[3], passfile=str(path),
                           sslmode='require', connect_timeout=25)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '600s'")
    return conn


def query(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, **metadata):
    """A sidecar marks a completed atomic download; no connection strings."""
    path = Path(path)
    metadata.update(file=path.name, sha256=fingerprint(path),
                    retrieved_utc=datetime.now(timezone.utc).isoformat())
    path.with_suffix(path.suffix + '.json').write_text(
        json.dumps(metadata, indent=2, default=str), encoding='utf-8')


def download_futures(conn, start=None, end=None, refresh=False):
    """One daily snapshot per contract series, including all listed maturities.

    Do not filter metadata by non-null expiry/FND: that would delete most history.
    The query projects settlement fields only; identical projections are collapsed,
    while conflicting quotes survive for explicit downstream rejection.
    """
    if start is None or end is None:
        settings = load_config()
        start = start or settings['sample']['download_start']
        end = end or settings['sample']['proposed_wrds_end']
    mapping = pd.read_csv(ROOT / 'configs/commodity_map.csv')
    codes = sorted(int(x) for x in mapping.contrcode.dropna().unique())
    out = ROOT / 'data/raw/wrds_contracts' / f'{start}_{end}'
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / 'metadata.parquet'
    if refresh or not metadata_path.exists():
        df = query(conn, f'SELECT * FROM {SCHEMA}.wrds_contract_info '
                   'WHERE contrcode = ANY(%s)', (codes,))
        for c in ['futcode', 'contrcode']:
            df[c] = pd.to_numeric(df[c]).astype('int64')
        df.to_parquet(metadata_path, index=False, compression='zstd')
        record(metadata_path, source=f'{SCHEMA}.wrds_contract_info', rows=len(df))
    for code in codes:
        path = out / f'contrcode={code}.parquet'
        sidecar = path.with_suffix('.parquet.json')
        if path.exists() and sidecar.exists() and not refresh:
            continue
        sql = f'''SELECT DISTINCT p.futcode::bigint AS futcode, p.date_ AS date,
                      p.settlement, p.volume, p.openinterest AS open_interest
                  FROM {SCHEMA}.wrds_fut_contract p
                  WHERE p.futcode IN (SELECT futcode FROM {SCHEMA}.wrds_contract_info
                                      WHERE contrcode = %s)
                    AND p.date_ BETWEEN %s AND %s
                  ORDER BY 1, 2'''
        tmp = path.with_suffix('.partial')
        writer, count = None, 0
        try:
            # fetchmany bounds local conversion size. Server-side SQL is read-only.
            with conn.cursor() as cur:
                cur.execute(sql, (code, start, end))
                while rows := cur.fetchmany(100000):
                    frame = pd.DataFrame(rows, columns=[c[0] for c in cur.description])
                    frame['date'] = pd.to_datetime(frame['date'])
                    frame['futcode'] = frame.futcode.astype('int64')
                    for c in ['settlement', 'volume', 'open_interest']:
                        frame[c] = pd.to_numeric(frame[c]).astype('float64')
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
                    writer.write_table(table)
                    count += len(frame)
            if writer is not None:
                writer.close()
                writer = None
            else:
                pd.DataFrame(columns=['futcode','date','settlement','volume',
                                      'open_interest']).to_parquet(tmp, index=False)
            tmp.replace(path)
            record(path, source=f'{SCHEMA}.wrds_fut_contract', contrcode=code,
                   start=start, end=end, rows=count, query=sql)
            print(f'contrcode {code}: {count:,} daily observations', flush=True)
        finally:
            if writer is not None:
                writer.close()
    return out
