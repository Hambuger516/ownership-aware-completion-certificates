"""Separate client observation from internal token production timestamps."""


def record_token_delivery(record, produced_ns, delivered_ns, *, retain_history):
    if delivered_ns < produced_ns or delivered_ns < record.get('client_last_token_ns', 0):
        raise ValueError('client delivery precedes production or prior delivery')
    previous = record.get('client_last_token_ns')
    if previous is None:
        record['client_first_token_ns'] = delivered_ns
        record['client_maximum_token_stall_ns'] = 0
    else:
        record['client_maximum_token_stall_ns'] = max(
            record['client_maximum_token_stall_ns'], delivered_ns-previous)
    record['client_last_token_ns'] = delivered_ns
    record['client_output_token_count'] = record.get('client_output_token_count', 0)+1
    if retain_history:
        record.setdefault('client_token_times_ns', []).append(delivered_ns)


def record_terminal_delivery(record, produced_ns, delivered_ns):
    if delivered_ns < max(produced_ns, record.get('client_last_token_ns', 0)):
        raise ValueError('terminal delivery precedes production or last output')
    record['client_terminal_ns'] = delivered_ns
