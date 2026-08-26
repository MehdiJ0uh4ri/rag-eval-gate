# Rate limits

Limits are applied per API key using a sliding window.

| Surface        | Limit                |
|----------------|----------------------|
| Read (`GET`)   | 200 requests / second |
| Write (`POST`) | 50 requests / second  |
| Search         | 20 requests / second  |
| Reporting      | 10 requests / minute  |

Exceeding a limit returns `429 rate_limit_exceeded` with a `Retry-After` header
in seconds. Every response carries `Acme-RateLimit-Remaining` and
`Acme-RateLimit-Reset`.

Rate limits are not raised on request in the sandbox environment. Production
limits can be raised by contacting support with a traffic forecast.
