# Payouts

A payout moves your available balance to a connected bank account.

- Default schedule is daily, with a 2 business day rolling delay from the time
  funds become available.
- Minimum payout amount is 1.00 in the account's settlement currency. Balances
  under the minimum roll into the next payout.
- Manual payouts are available on request via `POST /v1/payouts` but only if
  the account's schedule is set to `manual`.
- A payout in state `pending` can be cancelled; `in_transit` and `paid` cannot.

Failed payouts (state `failed`) carry a `failure_code`, most commonly
`account_closed`, `no_account`, or `debit_not_authorized`. The funds return to
your Acme balance within 1 business day.
