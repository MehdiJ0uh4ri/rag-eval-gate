# Idempotency

All `POST` endpoints accept an `Idempotency-Key` header. Keys are scoped to the
API key and the endpoint path.

- Keys are stored for 24 hours. After that the same key is treated as new.
- Replaying a key with an identical request body returns the original response,
  including the original status code, plus the header `Acme-Idempotent-Replay: true`.
- Replaying a key with a *different* body returns `422 idempotency_key_reuse`.
- Keys must be at most 255 characters. UUIDv4 is recommended.

Idempotency does not apply to `GET`, `DELETE`, or to the Files API.
