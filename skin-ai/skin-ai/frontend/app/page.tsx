"use client";

import { useEffect, useState } from "react";
import ImageUpload from "@/components/ImageUpload";
import ChatPanel from "@/components/ChatPanel";
import { createSession, type PredictResponse } from "@/lib/api";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch(() => setSessionId(null));
  }, []);

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-6 py-10">
      <header className="mb-8 flex items-baseline justify-between border-b border-slate-200 pb-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-clinic-teal">
            Clinical decision support
          </p>
          <h1 className="mt-1 font-display text-3xl font-semibold text-ink">Skin AI</h1>
        </div>
        <p className="max-w-xs text-right text-xs leading-relaxed text-slate-400">
          For use by licensed clinicians. Not a substitute for histopathological diagnosis.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <ImageUpload
            onPredicted={(result) => {
              setPrediction(result);
            }}
          />
        </div>

        <div className="lg:col-span-3" style={{ minHeight: 560 }}>
          <ChatPanel
            predictedLabel={prediction?.predicted_class ?? null}
            suggestedQuestions={prediction?.suggested_questions ?? []}
            sessionId={sessionId}
          />
        </div>
      </div>
    </main>
  );
}
