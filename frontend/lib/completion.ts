export interface CompletionCitation {
  number: number;
  chunk_id: string;
  source_document_id: string;
  passage: string;
  speaker: string | null;
  date: string | null;
  house: string | null;
  chamber: string | null;
}

export interface CompletionResponse {
  answer: string;
  citations: CompletionCitation[];
  abstained: boolean;
  abstention_reason: string | null;
}

export type AnswerPart =
  | { kind: "text"; value: string }
  | { kind: "citation"; number: number };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNullableString(value: unknown): value is string | null {
  return typeof value === "string" || value === null;
}

function isCitation(value: unknown): value is CompletionCitation {
  return (
    isRecord(value) &&
    typeof value.number === "number" &&
    Number.isInteger(value.number) &&
    typeof value.chunk_id === "string" &&
    typeof value.source_document_id === "string" &&
    typeof value.passage === "string" &&
    isNullableString(value.speaker) &&
    isNullableString(value.date) &&
    isNullableString(value.house) &&
    isNullableString(value.chamber)
  );
}

export function parseCompletionResponse(value: unknown): CompletionResponse {
  if (
    !isRecord(value) ||
    typeof value.answer !== "string" ||
    !Array.isArray(value.citations) ||
    !value.citations.every(isCitation) ||
    typeof value.abstained !== "boolean" ||
    !isNullableString(value.abstention_reason)
  ) {
    throw new Error("The completion service returned an invalid response.");
  }

  return {
    answer: value.answer,
    citations: value.citations,
    abstained: value.abstained,
    abstention_reason: value.abstention_reason,
  };
}

export function answerBody(answer: string): string {
  const sourcesStart = answer.lastIndexOf("\n\nSources:\n");
  return sourcesStart === -1 ? answer : answer.slice(0, sourcesStart);
}

export function answerParts(
  answer: string,
  citations: CompletionCitation[],
): AnswerPart[] {
  const citationNumbers = new Set(citations.map((citation) => citation.number));
  const parts: AnswerPart[] = [];
  const marker = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = marker.exec(answer)) !== null) {
    const number = Number(match[1]);
    if (!citationNumbers.has(number)) {
      continue;
    }

    if (match.index > lastIndex) {
      parts.push({ kind: "text", value: answer.slice(lastIndex, match.index) });
    }

    parts.push({ kind: "citation", number });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < answer.length) {
    parts.push({ kind: "text", value: answer.slice(lastIndex) });
  }

  return parts;
}

export function completionErrorMessage(
  body: unknown,
  status: number,
): string {
  if (isRecord(body) && typeof body.detail === "string") {
    return body.detail;
  }

  if (isRecord(body) && Array.isArray(body.detail)) {
    const messages = body.detail.flatMap((item) =>
      isRecord(item) && typeof item.msg === "string" ? [item.msg] : [],
    );
    if (messages.length > 0) {
      return messages.join(" ");
    }
  }

  return `The completion request failed (HTTP ${status}).`;
}
