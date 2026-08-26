# Webhooks

Acme Payments delivers events over HTTPS POST to endpoints you register in the
dashboard or via `POST /v1/webhook_endpoints`.

## Signature verification

Every delivery carries an `Acme-Signature` header of the form
`t=<unix_ts>,v1=<hex_hmac>`. The signed payload is `"{t}.{raw_body}"` hashed
with HMAC-SHA256 using your endpoint's signing secret (prefix `whsec_`).

Reject a delivery if the timestamp is more than 300 seconds old — this is the
replay-protection window. Always compare signatures in constant time.

## Retries

Failed deliveries (non-2xx, timeout > 10s, or connection error) are retried
with exponential backoff for up to 3 days: after 5m, 30m, 2h, 5h, 10h, then
every 12h. After 3 days the endpoint is disabled and an
`endpoint.disabled` event is sent to your account owner by email.

Deliveries are at-least-once. Consumers must deduplicate on the event `id`,
which is stable across retries.
