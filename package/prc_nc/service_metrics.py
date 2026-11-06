"""Common offline offered-load metrics; timestamps use one client clock.

No execution-token estimates or physical safety claims are derived here.
The measured cohort is [start, window_end); completion cutoff is inclusive.
"""
import math


def _ns(seconds):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('finite nonnegative seconds required')
    return round(seconds*1_000_000_000)


def from_prc(report, *, timing_basis='production'):
    """Explicit local observation or historical production time; never infer one.

    Client mode requires full delivery history (including an empty array for
    zero-output requests). Compact reports must first use the archive exporter.
    This observation point is in-process, not an independent network client.
    """
    if timing_basis not in ('production','client'):
        raise ValueError('unknown PRC timing basis')
    for rid,r in report['records'].items():
        if rid!=r['request_id']:raise ValueError('request identity mismatch')
        if timing_basis=='client':
            times=r.get('client_token_times_ns')
            if times is None or len(times)!=r.get('client_output_token_count'):
                raise ValueError('complete client delivery history required')
            produced=r.get('token_times_ns')
            if produced is None or len(produced)!=len(times) or any(d<p for p,d in zip(produced,times)):
                raise ValueError('client delivery differs from produced output')
            terminal=r.get('client_terminal_ns')
            if r['status'] in ('completed','cancelled','rejected','timeout','failed'):
                if terminal is None or r.get('terminal_ns') is None or terminal<r['terminal_ns']:
                    raise ValueError('client terminal observation required')
        else:
            times=r['token_times_ns'];terminal=r.get('terminal_ns')
        yield dict(request_id=rid,offered_ns=r['scheduled_ns'],submitted_ns=r['arrival_ns'],
            status=r['status'],terminal_ns=terminal,cap=r['max_new_tokens'],
            token_times_ns=times)


def from_vllm(records,*,wave):
    # Each historical wave has its own clock zero. Never concatenate their
    # relative timestamps into one continuous service window.
    for r in records:
        if r['wave']!=wave:continue
        status=r['terminal'] or 'pending'
        terminal=r.get('cancelled_at_s') if status=='cancelled' else r.get('finished_at_s')
        yield dict(request_id=r['request_id'],offered_ns=_ns(r['offered_at_s']),
            submitted_ns=None if r['submitted_at_s'] is None else _ns(r['submitted_at_s']),
            status=status,terminal_ns=None if terminal is None else _ns(terminal),cap=r['cap'],
            token_times_ns=[_ns(t) for t in r['token_times_s']])


def measure(records,*,start_ns,window_end_ns,cutoff_ns,ttft_limit_ns=2_000_000_000,tpot_limit_ns=100_000_000):
    if any(type(x) is not int or x<0 for x in (start_ns,window_end_ns,cutoff_ns,ttft_limit_ns,tpot_limit_ns)) or not start_ns<window_end_ns<=cutoff_ns:
        raise ValueError('ordered integer clock boundaries required')
    terminal_states={'completed','cancelled','rejected','timeout','failed'}
    outcomes={s:0 for s in sorted(terminal_states|{'pending'})};ids=set();rows=[]
    window_completions=0;window_tokens=0;slo_count=0;window_slo_count=0
    for r in records:
        rid=r['request_id'];offered=r['offered_ns'];submitted=r['submitted_ns'];terminal=r['terminal_ns'];times=r['token_times_ns']
        if type(rid) is not str or not rid or rid in ids:raise ValueError('unique request IDs required')
        ids.add(rid)
        if type(offered) is not int or offered<0:raise ValueError('invalid offered time')
        if submitted is not None and (type(submitted) is not int or submitted<offered):raise ValueError('invalid submission time')
        if type(r['cap']) is not int or r['cap']<1 or len(times)>r['cap']:raise ValueError('output contract mismatch')
        if any(type(t) is not int or t<offered for t in times) or any(a>b for a,b in zip(times,times[1:])):
            raise ValueError('invalid output timeline')
        if times and (submitted is None or times[0]<submitted):raise ValueError('output before submission')
        status=r['status']
        if status not in terminal_states|{'pending','queued','admitted'}:raise ValueError('unknown request status')
        if status in terminal_states:
            if type(terminal) is not int or terminal<offered or (submitted is not None and terminal<submitted) or (times and terminal<times[-1]):
                raise ValueError('invalid terminal time')
        elif terminal is not None:raise ValueError('nonterminal request has terminal time')
        if status=='completed' and not times:raise ValueError('completed request has no output')
        if not start_ns<=offered<window_end_ns:continue
        observed=[t for t in times if t<=cutoff_ns]
        final=status if status in terminal_states and terminal<=cutoff_ns else 'pending'
        outcomes[final]+=1
        gaps=[b-a for a,b in zip(observed,observed[1:])]
        ttft=observed[0]-offered if observed else None
        tpot=(observed[-1]-observed[0])/(len(observed)-1) if len(observed)>1 else None
        met=final=='completed' and ttft is not None and ttft<=ttft_limit_ns and (tpot is None or tpot<=tpot_limit_ns)
        slo_count+=met
        window_slo_count+=met and terminal<window_end_ns
        window_completions+=final=='completed' and terminal<window_end_ns
        window_tokens+=sum(start_ns<=t<window_end_ns for t in observed)
        rows.append(dict(request_id=rid,status_at_cutoff=final,output_tokens_at_cutoff=len(observed),
            ttft_ns=ttft,mean_tpot_ns=tpot,maximum_token_stall_ns=max(gaps,default=None),
            completed_end_to_end_ns=terminal-offered if final=='completed' else None,
            submit_lateness_ns=None if submitted is None else submitted-offered,slo_met=met))
    n=len(rows);seconds=(window_end_ns-start_ns)/1e9
    return dict(arrived=n,outcomes=outcomes,window_completed=window_completions,window_output_tokens=window_tokens,
        window_request_throughput=window_completions/seconds,window_output_token_throughput=window_tokens/seconds,
        window_slo_completed=window_slo_count,window_slo_throughput=window_slo_count/seconds,
        cohort_slo_completed=slo_count,cohort_slo_rate=slo_count/n if n else None,
        cohort_slo_throughput=slo_count/seconds,cutoff_ns=cutoff_ns,rows=rows,
        scope='offered-time cohort; all arrivals in denominator; SLO includes tail up to cutoff; single-token TPOT not applicable')


def latency_summary(measurement):
    """Request-weighted tails, with explicit undefined/excluded counts.

    Nearest-rank empirical percentiles; no interpolation or independence claim.
    These summarize the supplied observation clock, not inferred GPU timing.
    """
    rows=measurement['rows']
    if len(rows)!=measurement['arrived']:raise ValueError('arrival denominator mismatch')
    def summarize(key):
        values=sorted(r[key] for r in rows if r[key] is not None)
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('invalid latency sample')
        result=dict(observed_requests=len(values),undefined_requests=len(rows)-len(values))
        result.update({f'p{p}_ns':values[max(0,math.ceil(p/100*len(values))-1)] if values else None for p in (50,95,99)})
        result['max_ns']=values[-1] if values else None
        return result
    return dict(percentile_method='nearest_rank',weighting='one value per request; not independent run repeats',
        arrived=len(rows),zero_output_requests=sum(r['output_tokens_at_cutoff']==0 for r in rows),
        single_output_requests=sum(r['output_tokens_at_cutoff']==1 for r in rows),
        single_token_slo_rule='completed requests use TTFT only; TPOT remains undefined',
        ttft=summarize('ttft_ns'),mean_tpot=summarize('mean_tpot_ns'),
        maximum_observed_token_stall=summarize('maximum_token_stall_ns'),
        completed_end_to_end=summarize('completed_end_to_end_ns'),
        submission_lateness=summarize('submit_lateness_ns'),
        scope='request-level observed tails; zero-output failures remain in arrival denominator; pooled ITL not inferred')
