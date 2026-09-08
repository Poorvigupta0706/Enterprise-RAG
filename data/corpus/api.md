# API Limits and Authentication
## Authentication
All API calls require an `Authorization: Bearer nimbus_...` header. Keys are created by owners or admins. Keys are shown once at creation. Rotation creates a new key and keeps the previous key valid for 24 hours.
## Rate limits
Starter is limited to 10 requests per second (RPS) per workspace. Pro is limited to 100 RPS. Enterprise default is 500 RPS and can be raised with a support ticket. Burst tokens equal 2x the sustained RPS for 10 seconds. HTTP 429 responses include `Retry-After` in seconds.
## Idempotency
POST and PATCH endpoints accept an `Idempotency-Key` header. Keys are remembered for 24 hours. Replays with the same key and the same body return the original response. A different body with the same key returns HTTP 409.
## Versioning
The current stable API is `v1`. Deprecated `v0` will be shut down on 2027-03-01. Sunset headers are sent on `v0` responses.