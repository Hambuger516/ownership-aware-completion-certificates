"""Offline observed latency distributions with explicit unavailable populations."""
import math
from .service_metrics import measure


def distribution(values, *, population_size=None):
    values = sorted(values)
    return dict(observed_count=len(values),
                unavailable_count=None if population_size is None else population_size-len(values),
                seconds={label: values[math.ceil(p*len(values))-1]/1e9 if values else None
                         for label,p in (('p50',.5),('p95',.95),('p99',.99))},
                maximum_seconds=max(values)/1e9 if values else None,
                quantile='empirical nearest rank; no interpolation')


def summarize(records, *, start_ns, window_end_ns, cutoff_ns):
    records = list(records)
    metrics = measure(records, start_ns=start_ns, window_end_ns=window_end_ns,
                      cutoff_ns=cutoff_ns)
    rows = metrics['rows']
    result = dict(arrived=metrics['arrived'], outcomes=metrics['outcomes'],
                  timing_scope='offered-time cohort; observations only through cutoff',
                  no_output_at_cutoff=sum(r['output_tokens_at_cutoff']==0 for r in rows),
                  single_output_at_cutoff=sum(r['output_tokens_at_cutoff']==1 for r in rows),
                  request_distributions={})
    for field in ('ttft_ns','mean_tpot_ns','maximum_token_stall_ns','submit_lateness_ns'):
        result['request_distributions'][field] = distribution(
            [r[field] for r in rows if r[field] is not None], population_size=len(rows))
    gaps = []
    completed = []
    for r in records:
        if not start_ns <= r['offered_ns'] < window_end_ns:
            continue
        times = [t for t in r['token_times_ns'] if t<=cutoff_ns]
        gaps.extend(b-a for a,b in zip(times,times[1:]))
        if r['status']=='completed' and r['terminal_ns']<=cutoff_ns:
            completed.append(r['terminal_ns']-r['offered_ns'])
    result['completed_end_to_end'] = distribution(completed, population_size=len(rows))
    result['pooled_inter_token_gaps'] = distribution(gaps)
    result['limits'] = [
        'No-output requests have undefined TTFT, not zero; outcomes remain in all-arrival denominator.',
        'Zero/one-output requests have undefined TPOT and maximum inter-token gap, not zero.',
        'Observed quantiles do not impute censored latency or establish population tail guarantees.',
        'Pooled gaps weight longer outputs more; intervals are not independent experimental repeats.']
    return result
