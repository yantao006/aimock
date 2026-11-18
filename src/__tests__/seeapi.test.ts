import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, test, expect, afterEach } from "vitest";
import { LLMock } from "../llmock.js";
import { startFromConfig } from "../config-loader.js";
import { SeeapiJobMap, SEEAPI_MAX_ENTRIES } from "../seeapi.js";

// ─── SeeapiJobMap unit surface ────────────────────────────────────────────────

describe("SeeapiJobMap", () => {
  test("is exported with the bounded-entries constant and clear resets generation", () => {
    expect(SEEAPI_MAX_ENTRIES).toBe(10_000);
    const map = new SeeapiJobMap();
    expect(map.size).toBe(0);
    expect(map.generation).toBe(0);
    map.setIdempotency("default:k1", "gen-1");
    expect(map.getIdempotency("default:k1")).toBe("gen-1");
    map.clear();
    expect(map.generation).toBe(1);
    expect(map.getIdempotency("default:k1")).toBeUndefined();
  });
});

// ─── POST /seeapi/v1/generations & GET /seeapi/v1/generations/{id} ───────────

describe("SeeAPI mock (/seeapi/v1/...)", () => {
  let mock: LLMock | undefined;
  let tmpDirs: string[] = [];

  afterEach(async () => {
    await mock?.stop();
    mock = undefined;
    for (const dir of tmpDirs) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
    tmpDirs = [];
  });

  test("submit returns 202 { generation } and polls to succeeded with assets", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage("nano-banana-2-1", {
      urls: ["https://assets.seeapi.com/out/cat.png"],
      consumedCredits: 12,
    });
    await mock.start();

    const submit = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-seeapi-key",
        "Idempotency-Key": "idem-1",
      },
      body: JSON.stringify({
        model: "nano-banana-2-1",
        endpoint: "text-to-image",
        provider: "seeapi",
        input: {
          prompt: "a cute cat",
          aspect_ratio: "1:1",
          resolution: "1K",
          output_format: "png",
        },
      }),
    });
    expect(submit.status).toBe(202);
    const submitJson = await submit.json();
    expect(submitJson.generation.status).toBe("queued");
    expect(submitJson.generation.consumed_credits).toBe(12);
    const id: string = submitJson.generation.id;
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);

    const poll = await fetch(`${mock.url}/seeapi/v1/generations/${id}`, {
      headers: { Authorization: "Bearer test-seeapi-key" },
    });
    expect(poll.status).toBe(200);
    const pollJson = await poll.json();
    expect(pollJson.generation.id).toBe(id);
    expect(pollJson.generation.status).toBe("succeeded");
    expect(pollJson.generation.result).toEqual({
      assets: [{ url: "https://assets.seeapi.com/out/cat.png" }],
    });
  });

  test("does not clash with /v1/generations and GET /seeapi/v1/generations returns 404", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage("nano-banana-2-1", { urls: ["https://assets.seeapi.com/a.png"] });
    await mock.start();

    const wrongPrefix = await fetch(`${mock.url}/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k" },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "hi" } }),
    });
    expect(wrongPrefix.status).toBe(404);

    const listRes = await fetch(`${mock.url}/seeapi/v1/generations`);
    expect(listRes.status).toBe(404);
  });

  test("missing Idempotency-Key returns HTTP 400", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage("nano-banana-2-1", { urls: ["https://assets.seeapi.com/a.png"] });
    await mock.start();

    const res = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: JSON.stringify({
        model: "nano-banana-2-1",
        input: { prompt: "hello" },
      }),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.message).toContain("Idempotency-Key");
  });

  test("same Idempotency-Key returns the same generation without creating a second job", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { urls: ["https://assets.seeapi.com/first.png"] },
      { sequenceIndex: 0 },
    );
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { urls: ["https://assets.seeapi.com/second.png"] },
      { sequenceIndex: 1 },
    );
    await mock.start();

    const makeReq = (idemKey: string, testId = "t1") =>
      fetch(`${mock!.url}/seeapi/v1/generations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": idemKey,
          "X-Test-Id": testId,
        },
        body: JSON.stringify({
          model: "nano-banana-2-1",
          input: { prompt: "cat" },
        }),
      });

    const r1 = await (await makeReq("same-key")).json();
    const r2 = await (await makeReq("same-key")).json();
    expect(r2.generation.id).toBe(r1.generation.id);

    // Different Idempotency-Key consumes sequenceIndex 1 (proving r2 did not create a second job)
    const r3 = await (await makeReq("other-key")).json();
    expect(r3.generation.id).not.toBe(r1.generation.id);

    const poll3 = await (
      await fetch(`${mock.url}/seeapi/v1/generations/${r3.generation.id}`, {
        headers: { "X-Test-Id": "t1" },
      })
    ).json();
    expect(poll3.generation.result.assets[0].url).toBe("https://assets.seeapi.com/second.png");
  });

  test("rejects unknown input field with HTTP 400 (app_key regression guard)", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage("nano-banana-2-1", { urls: ["https://assets.seeapi.com/a.png"] });
    await mock.start();

    const res = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k-bad-field" },
      body: JSON.stringify({
        model: "nano-banana-2-1",
        endpoint: "text-to-image",
        provider: "seeapi",
        input: {
          prompt: "a portrait",
          app_key: "leaked_internal_field",
        },
      }),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.message).toContain("invalid input: unknown input field app_key");
  });

  test("allowedInputFieldsByModel and strictInputFields: false are configurable", async () => {
    mock = new LLMock({
      port: 0,
      seeapi: {
        allowedInputFieldsByModel: {
          "custom-model": ["prompt", "style_preset"],
        },
      },
    });
    mock.onSeeapiImage("custom-model", { urls: ["https://assets.seeapi.com/custom.png"] });
    await mock.start();

    const okRes = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k-custom" },
      body: JSON.stringify({
        model: "custom-model",
        input: { prompt: "hi", style_preset: "anime" },
      }),
    });
    expect(okRes.status).toBe(202);

    const badRes = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k-custom-2" },
      body: JSON.stringify({
        model: "custom-model",
        input: { prompt: "hi", resolution: "1K" },
      }),
    });
    expect(badRes.status).toBe(400);
    expect((await badRes.json()).error.message).toContain("unknown input field resolution");
    await mock.stop();

    mock = new LLMock({ port: 0, seeapi: { strictInputFields: false } });
    mock.onSeeapiImage("nano-banana-2-1", { urls: ["https://assets.seeapi.com/permissive.png"] });
    await mock.start();

    const permissiveRes = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k-perm" },
      body: JSON.stringify({
        model: "nano-banana-2-1",
        input: { prompt: "hi", app_key: "allowed-when-non-strict" },
      }),
    });
    expect(permissiveRes.status).toBe(202);
  });

  test("poll progression transitions queued -> processing -> succeeded", async () => {
    mock = new LLMock({
      port: 0,
      seeapi: { pollsBeforeInProgress: 1, pollsBeforeCompleted: 2 },
    });
    mock.onSeeapiImage(/sunset/, { urls: ["https://assets.seeapi.com/sunset.jpg"] });
    await mock.start();

    const submit = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "k-prog" },
      body: JSON.stringify({
        model: "nano-banana-2-1",
        input: { prompt: "a bright sunset" },
      }),
    });
    expect(submit.status).toBe(202);
    const { generation } = await submit.json();
    expect(generation.status).toBe("queued");

    const poll1 = await (await fetch(`${mock.url}/seeapi/v1/generations/${generation.id}`)).json();
    expect(poll1.generation.status).toBe("processing");
    expect(poll1.generation.result).toBeUndefined();

    const poll2 = await (await fetch(`${mock.url}/seeapi/v1/generations/${generation.id}`)).json();
    expect(poll2.generation.status).toBe("succeeded");
    expect(poll2.generation.result.assets).toEqual([
      { url: "https://assets.seeapi.com/sunset.jpg" },
    ]);
  });

  test("failure fixture polls to failed (and supports canceled/refunded/expired)", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { error: "safety policy violation" },
      { prompt: "unsafe" },
    );
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { error: "credits refunded", status: "refunded" },
      { prompt: "refund" },
    );
    await mock.start();

    const s1 = await (
      await fetch(`${mock.url}/seeapi/v1/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Idempotency-Key": "fail-1" },
        body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "unsafe" } }),
      })
    ).json();
    const p1 = await (await fetch(`${mock.url}/seeapi/v1/generations/${s1.generation.id}`)).json();
    expect(p1.generation.status).toBe("failed");
    expect(p1.generation.error_message).toBe("safety policy violation");

    const s2 = await (
      await fetch(`${mock.url}/seeapi/v1/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Idempotency-Key": "fail-2" },
        body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "refund" } }),
      })
    ).json();
    const p2 = await (await fetch(`${mock.url}/seeapi/v1/generations/${s2.generation.id}`)).json();
    expect(p2.generation.status).toBe("refunded");
    expect(p2.generation.error_message).toBe("credits refunded");
  });

  test("matches on input_images presence and exact count", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { urls: ["https://assets.seeapi.com/txt2img.png"] },
      { hasInputImages: false },
    );
    mock.onSeeapiImage(
      "nano-banana-2-1",
      { urls: ["https://assets.seeapi.com/two-refs.png"] },
      { inputImageCount: 2 },
    );
    await mock.start();

    const s1 = await (
      await fetch(`${mock.url}/seeapi/v1/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Idempotency-Key": "img-0" },
        body: JSON.stringify({
          model: "nano-banana-2-1",
          endpoint: "text-to-image",
          input: { prompt: "dog" },
        }),
      })
    ).json();
    const p1 = await (await fetch(`${mock.url}/seeapi/v1/generations/${s1.generation.id}`)).json();
    expect(p1.generation.result.assets[0].url).toBe("https://assets.seeapi.com/txt2img.png");

    const s2 = await (
      await fetch(`${mock.url}/seeapi/v1/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Idempotency-Key": "img-2" },
        body: JSON.stringify({
          model: "nano-banana-2-1",
          endpoint: "image-to-image",
          input: {
            prompt: "merge",
            input_images: ["https://example.com/a.png", "https://example.com/b.png"],
          },
        }),
      })
    ).json();
    const p2 = await (await fetch(`${mock.url}/seeapi/v1/generations/${s2.generation.id}`)).json();
    expect(p2.generation.result.assets[0].url).toBe("https://assets.seeapi.com/two-refs.png");
  });

  test("GET /seeapi/v1/models returns static configurable catalog", async () => {
    mock = new LLMock({
      port: 0,
      seeapi: {
        models: [{ id: "seeapi-pro", provider: "seeapi", endpoints: ["text-to-image"] }],
      },
    });
    await mock.start();

    const res = await fetch(`${mock.url}/seeapi/v1/models`);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.models).toEqual([
      { id: "seeapi-pro", provider: "seeapi", endpoints: ["text-to-image"] },
    ]);
  });

  test("journals every request with service='seeapi' and supports chaos injection", async () => {
    mock = new LLMock({ port: 0 });
    mock.onSeeapiImage("nano-banana-2-1", { urls: ["https://assets.seeapi.com/j.png"] });
    await mock.start();

    const ok = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "j-1" },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "journal" } }),
    });
    expect(ok.status).toBe(202);
    const { generation } = await ok.json();
    await fetch(`${mock.url}/seeapi/v1/generations/${generation.id}`);

    const chaosDrop = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": "j-chaos",
        "X-AIMock-Chaos-Drop": "1",
      },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "journal" } }),
    });
    expect(chaosDrop.status).toBe(500);

    const journalRes = await fetch(`${mock.url}/__aimock/journal?service=seeapi`);
    const journalBody = await journalRes.json();
    expect(Array.isArray(journalBody)).toBe(true);
    expect(journalBody.length).toBe(3);
    expect(journalBody.every((e: { service?: string }) => e.service === "seeapi")).toBe(true);

    mock.reset();
    const afterReset = await fetch(`${mock.url}/seeapi/v1/generations/${generation.id}`);
    expect(afterReset.status).toBe(404);
  });

  test("strict 503 vs non-strict 404, malformed JSON 400, multipart 400, unknown id 404", async () => {
    mock = new LLMock({ port: 0, strict: true });
    await mock.start();

    const strictMiss = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "s1" },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "miss" } }),
    });
    expect(strictMiss.status).toBe(503);
    await mock.stop();

    mock = new LLMock({ port: 0 });
    await mock.start();

    const looseMiss = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "s2" },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "miss" } }),
    });
    expect(looseMiss.status).toBe(404);

    const badJson = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "s3" },
      body: "{bad",
    });
    expect(badJson.status).toBe(400);

    const multipart = await fetch(`${mock.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: {
        "Content-Type": "multipart/form-data; boundary=abc",
        "Idempotency-Key": "s4",
      },
      body: "--abc--",
    });
    expect(multipart.status).toBe(400);

    const unknownPoll = await fetch(`${mock.url}/seeapi/v1/generations/missing-id`);
    expect(unknownPoll.status).toBe(404);
  });

  test("loads from config file and JSON fixture file and honors delayMs / latency", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "aimock-seeapi-cfg-"));
    tmpDirs.push(dir);
    const fixturePath = path.join(dir, "fixtures.json");
    fs.writeFileSync(
      fixturePath,
      JSON.stringify({
        fixtures: [
          {
            match: { endpoint: "seeapi", model: "nano-banana-2-1", userMessage: "config test" },
            response: { urls: ["https://assets.seeapi.com/from-file.png"] },
            pollsBeforeInProgress: 1,
            pollsBeforeCompleted: 2,
            latency: 15,
          },
        ],
      }),
    );

    const started = await startFromConfig({
      port: 0,
      host: "127.0.0.1",
      llm: { fixtures: fixturePath },
      seeapi: { strictInputFields: true },
    });
    mock = started.llmock;

    const t0 = Date.now();
    const submit = await fetch(`${started.url}/seeapi/v1/generations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": "cfg-1" },
      body: JSON.stringify({ model: "nano-banana-2-1", input: { prompt: "config test" } }),
    });
    expect(submit.status).toBe(202);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(10);
    const { generation } = await submit.json();

    const p1 = await (await fetch(`${started.url}/seeapi/v1/generations/${generation.id}`)).json();
    expect(p1.generation.status).toBe("processing");
    const p2 = await (await fetch(`${started.url}/seeapi/v1/generations/${generation.id}`)).json();
    expect(p2.generation.status).toBe("succeeded");
    expect(p2.generation.result.assets[0].url).toBe("https://assets.seeapi.com/from-file.png");
  });
});
