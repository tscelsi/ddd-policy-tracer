"use client";

import { FormEvent, KeyboardEvent, useState } from "react";
import {
  answerBody,
  answerParts,
  CompletionCitation,
  CompletionResponse,
  completionErrorMessage,
  parseCompletionResponse,
} from "@/lib/completion";

type Message =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      result: CompletionResponse;
    };

const suggestedQuestions = [
  "What did X talk about today in parliament?",
  "What bills were talked about today?",
  "Which members spoke about housing?",
];

function citationLabel(citation: CompletionCitation): string {
  return [citation.speaker, citation.date, citation.house, citation.chamber]
    .filter((value): value is string => Boolean(value))
    .join(" · ");
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openCitation, setOpenCitation] = useState<string | null>(null);

  async function submitQuestion(rawQuestion: string) {
    const submittedQuestion = rawQuestion.trim();
    if (!submittedQuestion || isLoading) {
      return;
    }

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: submittedQuestion,
      },
    ]);
    setQuestion("");
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch("/api/completion", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: submittedQuestion }),
      });

      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        throw new Error("The completion service returned an unreadable response.");
      }

      if (!response.ok) {
        throw new Error(completionErrorMessage(payload, response.status));
      }

      const result = parseCompletionResponse(payload);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          result,
        },
      ]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The request failed. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(question);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(question);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            P
          </span>
          <span>PARLIAMENTARY RECORD</span>
        </div>
        <nav className="top-nav" aria-label="Main navigation">
          <span className="nav-item nav-item-active">Ask</span>
          <span className="nav-item">Speeches</span>
        </nav>
      </header>

      <section className="chat-frame" aria-label="Parliament question and answer">
        <div className="chat-heading">
          <div>
            <p className="eyebrow">COMPLETION</p>
            <h1>Ask Parliament</h1>
            <p className="intro">
              Explore what was said and which bills were discussed.
            </p>
          </div>
          <span className="session-note">Each question is a fresh search</span>
        </div>

        <div className="conversation" aria-live="polite">
          {messages.length === 0 && !isLoading ? (
            <div className="welcome">
              <span className="welcome-symbol" aria-hidden="true">
                ✳
              </span>
              <h2>What would you like to know?</h2>
              <p>
                Ask a question about parliamentary speeches. Answers include
                source excerpts so you can check the evidence.
              </p>
              <div className="suggestions" aria-label="Suggested questions">
                {suggestedQuestions.map((suggestion) => (
                  <button
                    className="suggestion"
                    key={suggestion}
                    onClick={() => void submitQuestion(suggestion)}
                    type="button"
                  >
                    {suggestion}
                    <span aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) =>
                message.role === "user" ? (
                  <article className="message message-user" key={message.id}>
                    <div className="message-label">YOU</div>
                    <p>{message.text}</p>
                  </article>
                ) : (
                  <AssistantMessage
                    key={message.id}
                    messageId={message.id}
                    result={message.result}
                    openCitation={openCitation}
                    onToggleCitation={(citationKey) =>
                      setOpenCitation((current) =>
                        current === citationKey ? null : citationKey,
                      )
                    }
                  />
                ),
              )}
              {isLoading && (
                <div className="message message-assistant loading-message">
                  <div className="message-label">PARLIAMENTARY RECORD</div>
                  <p>
                    <span className="loading-dot" aria-hidden="true" />
                    Searching speeches and preparing an answer…
                  </p>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <strong>We couldn’t complete that request.</strong>
              <span>{error}</span>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label className="visually-hidden" htmlFor="question">
            Ask a question
          </label>
          <textarea
            id="question"
            name="question"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about a speech, member, or bill…"
            rows={2}
            value={question}
          />
          <div className="composer-footer">
            <span>Enter to send · Shift + Enter for a new line</span>
            <button
              className="send-button"
              disabled={!question.trim() || isLoading}
              type="submit"
            >
              {isLoading ? "Searching…" : "Ask"}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
        </form>
        <p className="privacy-note">
          Answers are based on available parliamentary records and may not cover
          every debate.
        </p>
      </section>
    </main>
  );
}

function AssistantMessage({
  messageId,
  result,
  openCitation,
  onToggleCitation,
}: {
  messageId: string;
  result: CompletionResponse;
  openCitation: string | null;
  onToggleCitation: (citationKey: string) => void;
}) {
  const content = answerBody(result.answer);
  const citations = result.citations;
  const parts = answerParts(content, citations);

  return (
    <article className="message message-assistant">
      <div className="message-label">PARLIAMENTARY RECORD</div>
      {result.abstained && (
        <div className="abstention-banner">
          <strong>Not enough evidence to answer confidently</strong>
          {result.abstention_reason && <span>{result.abstention_reason}</span>}
        </div>
      )}
      <div className="answer-text">
        {parts.map((part, index) =>
          part.kind === "citation" ? (
            <button
              aria-label={`Show source excerpt ${part.number}`}
              className="inline-citation"
              key={`${part.number}-${index}`}
              onClick={() =>
                onToggleCitation(`${messageId}-${part.number}`)
              }
              type="button"
            >
              [{part.number}]
            </button>
          ) : (
            <span key={`text-${index}`}>{part.value}</span>
          ),
        )}
      </div>

      {citations.length > 0 && (
        <section className="sources" aria-label="Sources">
          <div className="sources-heading">
            <span>SUPPORTING EXCERPTS</span>
            <span>{citations.length}</span>
          </div>
          {citations.map((citation) => {
            const citationKey = `${messageId}-${citation.number}`;
            const isOpen = openCitation === citationKey;
            const label = citationLabel(citation);

            return (
              <article className="source-card" key={citation.number}>
                <button
                  aria-controls={`excerpt-${citationKey}`}
                  aria-expanded={isOpen}
                  className="source-toggle"
                  onClick={() => onToggleCitation(citationKey)}
                  type="button"
                >
                  <span className="source-number">[{citation.number}]</span>
                  <span className="source-summary">
                    {label || "Source excerpt"}
                  </span>
                  <span className="source-action">
                    {isOpen ? "Hide excerpt" : "Read excerpt"}
                  </span>
                </button>
                <div
                  className="source-detail"
                  hidden={!isOpen}
                  id={`excerpt-${citationKey}`}
                >
                  <blockquote>{citation.passage}</blockquote>
                  <dl className="source-metadata">
                    <MetadataItem label="Speaker" value={citation.speaker} />
                    <MetadataItem label="Date" value={citation.date} />
                    <MetadataItem label="House" value={citation.house} />
                    <MetadataItem label="Chamber" value={citation.chamber} />
                  </dl>
                </div>
              </article>
            );
          })}
        </section>
      )}
    </article>
  );
}

function MetadataItem({ label, value }: { label: string; value: string | null }) {
  if (!value) {
    return null;
  }

  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
