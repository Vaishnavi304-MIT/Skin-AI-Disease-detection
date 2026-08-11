"use client";

import { useEffect, useRef, useState } from "react";
import { sendChatMessage } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  predictedLabel: string | null;
  suggestedQuestions: string[];
  sessionId: string | null;
}

export default function ChatPanel({ predictedLabel, suggestedQuestions, sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevLabel = useRef<string | null>(null);

  // Reset conversation whenever a new diagnosis comes in
  useEffect(() => {
    if (predictedLabel && predictedLabel !== prevLabel.current) {
      setMessages([]);
      setError(null);
      prevLabel.current = predictedLabel;
    }
  }, [predictedLabel]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const submit = async (text: string) => {
    if (!text.trim() || !predictedLabel || !sessionId || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const res = await sendChatMessage(text, predictedLabel, sessionId);
      setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate a response.");
    } finally {
      setSending(false);
    }
  };

  const disabled = !predictedLabel || !sessionId;

  return (
    <section className="flex h-full flex-col rounded-lg border border-slate-200 bg-white shadow-panel">
      <header className="border-b border-slate-100 px-5 py-4">
        <h2 className="font-display text-lg font-semibold text-ink">Guideline chatbot</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          {predictedLabel ? (
            <>
              Grounded in reference material for{" "}
              <span className="font-medium text-clinic-teal">{predictedLabel}</span>
            </>
          ) : (
            "Classify a lesion to activate this chat"
          )}
        </p>
      </header>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {disabled && (
          <div className="flex h-full items-center justify-center text-center text-sm text-slate-400">
            Upload and classify an image on the left to start a guideline-grounded conversation.
          </div>
        )}

        {!disabled && messages.length === 0 && suggestedQuestions.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
              Suggested questions
            </p>
            <div className="flex flex-col gap-2">
              {suggestedQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => submit(q)}
                  className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-left text-sm text-slate-700 transition-colors hover:border-clinic-teal hover:bg-clinic-teal/5"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-md px-3.5 py-2.5 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-clinic-teal text-white"
                  : "border border-slate-200 bg-slate-50 text-ink"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}

        {sending && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-3.5 py-2.5">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-md border border-clinic-red/30 bg-clinic-red/5 px-3 py-2 text-sm text-clinic-red">
            {error}
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(input);
        }}
        className="flex items-center gap-2 border-t border-slate-100 px-4 py-3"
      >
        <input
          type="text"
          value={input}
          disabled={disabled || sending}
          onChange={(e) => setInput(e.target.value)}
          placeholder={disabled ? "Waiting for classification…" : "Ask a clinical question…"}
          className="flex-1 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-ink placeholder:text-slate-400 focus:border-clinic-teal focus:bg-white focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={disabled || sending || !input.trim()}
          className="rounded-md bg-clinic-teal px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-clinic-tealDark disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </section>
  );
}
