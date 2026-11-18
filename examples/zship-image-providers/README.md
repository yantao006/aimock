# ZShip Image Providers Example

This example configures `aimock` to serve local mocks for the three image providers used by ZShip Provider1: SeeAPI, KIE, and fal.ai.

## Starting the Mock Server

Run from this directory:

```bash
cd examples/zship-image-providers
npx @copilotkit/aimock --config aimock.json
```

Or from the repository root after building:

```bash
cd examples/zship-image-providers
node ../../dist/aimock-cli.js --config aimock.json
```

## Pointing Provider1 `api_base_url` at `aimock`

Set each provider's `api_base_url` in Provider1's database to the matching prefix on `http://127.0.0.1:4010`:

| Provider | Provider1 `api_base_url`           | Resulting Submit Route                      |
| -------- | ---------------------------------- | ------------------------------------------- |
| SeeAPI   | `http://127.0.0.1:4010/seeapi/v1`  | `POST /seeapi/v1/generations`               |
| KIE      | `http://127.0.0.1:4010/kie/api/v1` | `POST /kie/api/v1/jobs/createTask`          |
| fal.ai   | `http://127.0.0.1:4010/fal`        | `POST /fal/openai/gpt-image-2.6/flare/edit` |

Each provider uses a distinct path prefix (`/seeapi`, `/kie`, `/fal`) so they never collide on the same port.
SeeAPI enforces `Idempotency-Key` and strict input field validation (`strictInputFields: true`).
KIE returns `resultJson` as a JSON-encoded string and delivers `callBackUrl` payloads on completion.
fal.ai runs with `vendorPaths: true`, `webhook: true`, and `falQueue: { pollsBeforeInProgress: 1, pollsBeforeCompleted: 2 }`, so Provider1 calls work without `x-fal-target-host`, receive HTTP `400` on the first result poll before returning `200` on the second (set `pollsBeforeCompleted: 3` for two `400` polls), and receive `?fal_webhook=` callbacks.
