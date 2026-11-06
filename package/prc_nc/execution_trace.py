"""Offline accounting of successful PRC model events, not GPU kernel sensing."""
from prc_nc.execution_meter import ExecutionTokenMeter


def replay_execution(trace):
    if trace.get('status') != 'PASS':
        raise ValueError('complete successful trace required; failed compute is unknown')
    meter = ExecutionTokenMeter()
    pending = {}
    serial = 0
    padding = 0
    shared = 0
    for event in trace['events']:
        kind = event['kind']
        if kind == 'model_batch':
            if pending:
                raise ValueError('incomplete preceding model batch')
            ids, starts, lengths = (event[k] for k in ('request_ids', 'start_positions', 'input_lengths'))
            if not len(ids) == len(starts) == len(lengths) or not ids:
                raise ValueError('incomplete batch geometry')
            if event['actual_input_tokens'] != sum(lengths) or event['padded_query_tokens'] != len(ids)*max(lengths):
                raise ValueError('inconsistent physical batch geometry')
            spans = list(zip(ids, starts, lengths))
            meter.prepare(serial, spans)
            meter.confirm(serial, ids)
            serial += 1
            padding += event['padded_query_tokens'] - sum(lengths)
            pending = {rid: (event['batch_id'], start, count) for rid, start, count in spans}
        elif kind in ('token', 'prefill_chunk'):
            rid = event['request_id']
            if 'batch_id' in event:
                expected = (event['batch_id'], event['start_position'], event['input_tokens'])
                if pending.pop(rid, None) != expected:
                    raise ValueError('batch row missing, duplicated or inconsistent')
            else:
                if pending:
                    raise ValueError('single execution inside incomplete batch')
                meter.prepare(serial, [(rid, event['start_position'], event['input_tokens'])])
                meter.confirm(serial, [rid])
                serial += 1
        elif kind == 'prefix_attachment':
            shared += event['plan']['tokens']
    if pending:
        raise ValueError('unconfirmed batch rows at end of trace')
    if set(meter.executed) - set(trace['records']):
        raise ValueError('execution outside arrival cohort')
    result = meter.report(request_ids=trace['records'])
    result.update(padded_query_positions=padding, attached_prefix_positions=shared,
                  measurement_source='successful service execution events; excludes failed/partial kernels; not GPU instruction counting')
    return result
