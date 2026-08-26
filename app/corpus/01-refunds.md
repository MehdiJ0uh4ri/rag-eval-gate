# Refunds

A refund returns funds from a captured payment back to the customer's original
payment method. Refunds are created against a `payment_id`, never against a
customer directly.

## Rules

- A payment must be in state `captured` before it can be refunded. Refunding an
  `authorized` payment returns `409 payment_not_captured`; void it instead.
- Partial refunds are allowed. The sum of all refunds on a payment may never
  exceed the captured amount; the API rejects the excess with
  `422 refund_exceeds_captured_amount`.
- A payment may carry at most 20 refunds. The 21st returns `429 refund_limit`.
- Refunds are irreversible. There is no "un-refund" endpoint.

## Timing

Card refunds settle in 5-10 business days. SEPA refunds settle in 2 business
days. Wallet refunds (Apple Pay, Google Pay) follow the underlying card timing.

Refunds older than 180 days from the capture date cannot be processed through
the API and must be handled as a separate payout.
