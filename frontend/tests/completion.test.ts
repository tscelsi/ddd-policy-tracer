import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
  answerBody,
  answerParts,
  completionErrorMessage,
  parseCompletionResponse,
} from "../lib/completion";

const citation = {
  number: 1,
  chunk_id: "chunk-1",
  source_document_id: "source-1",
  passage: "The supporting passage.",
  speaker: "Alex Example",
  date: "2025-02-12",
  house: "House of Representatives",
  chamber: "Main",
};

describe("completion response", () => {
  it("accepts the API response shape with nullable metadata", () => {
    assert.deepEqual(
      parseCompletionResponse({
        answer: "A supported answer. [1]\n\nSources:\n[1] Alex Example",
        citations: [{ ...citation, speaker: null }],
        abstained: false,
        abstention_reason: null,
      }),
      {
        answer: "A supported answer. [1]\n\nSources:\n[1] Alex Example",
        citations: [{ ...citation, speaker: null }],
        abstained: false,
        abstention_reason: null,
      },
    );
  });

  it("rejects malformed success payloads", () => {
    assert.throws(
      () =>
        parseCompletionResponse({
        answer: "An answer",
        citations: [{ ...citation, number: "1" }],
        abstained: false,
        abstention_reason: null,
        }),
      /invalid response/,
    );
  });

  it("separates the structured citations from the answer's text source list", () => {
    assert.equal(
      answerBody("An answer. [1]\n\nSources:\n[1] Alex Example | source id"),
      "An answer. [1]",
    );
  });

  it("turns only known inline citation markers into interactive parts", () => {
    assert.deepEqual(answerParts("See [1], not [2].", [citation]), [
      { kind: "text", value: "See " },
      { kind: "citation", number: 1 },
      { kind: "text", value: ", not [2]." },
    ]);
  });

  it("surfaces API validation details", () => {
    assert.equal(
      completionErrorMessage(
        {
          detail: [
            { msg: "Field required", loc: ["body", "question"] },
            { msg: "Too long", loc: ["body", "question"] },
          ],
        },
        422,
      ),
      "Field required Too long",
    );
  });
});
