# Disputes

A dispute (chargeback) is raised by the cardholder's issuing bank.

- The disputed amount plus a 15.00 dispute fee are debited from your balance
  immediately when the dispute is created.
- You have 7 days from dispute creation to submit evidence via
  `POST /v1/disputes/{id}/evidence`. The issuing bank's own deadline may be
  longer, but Acme's submission window closes at 7 days.
- Evidence can be submitted once. Use `POST .../evidence` with
  `submit: false` to save a draft; `submit: true` is final.
- If you win, the amount and the fee are both returned to your balance.
- Disputes cannot be refunded. Attempting to refund a disputed payment returns
  `409 payment_disputed`.
