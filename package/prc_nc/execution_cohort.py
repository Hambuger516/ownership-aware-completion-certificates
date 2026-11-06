"""Select measured vLLM requests by returned IDs, never guessed ID suffixes."""


def select_execution_cohort(raw, records):
    if raw.get('request_selection') != 'all_internal_including_warmup' or raw['pending_batches']:
        raise ValueError('complete all-internal execution observation required')
    selected = {}
    used = set()
    for rid, record in records.items():
        internal = record.get('engine_request_id')
        if record['status'] == 'rejected':
            if internal is not None:
                raise ValueError('rejected request has engine identity')
            continue
        if record['status'] != 'completed' or not isinstance(internal, str) or not internal or internal in used:
            raise ValueError('unique completed engine request identity required')
        used.add(internal)
        values = raw['requests'][internal]
        if not 0 <= values['reexecuted_tokens'] <= values['executed_tokens']:
            raise ValueError('invalid execution counts')
        selected[rid] = dict(engine_request_id=internal, **values)
    return dict(executed_tokens=sum(x['executed_tokens'] for x in selected.values()),
                reexecuted_tokens=sum(x['reexecuted_tokens'] for x in selected.values()),
                requests=selected, excluded_internal_requests=sorted(set(raw['requests'])-used),
                scope='completed client cohort via exact engine-returned IDs; warmup excluded; rejected requests have no execution')
