"use client";

import { useCallback, useRef, useState } from "react";
import { predictImage, type PredictResponse } from "@/lib/api";

interface Props {
  onPredicted: (result: PredictResponse, imageUrl: string) => void;
}

export default function ImageUpload({ onPredicted }: Props) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
        setError("Upload a JPG or PNG image.");
        setStatus("error");
        return;
      }

      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setStatus("loading");
      setError(null);
      setResult(null);

      try {
        const prediction = await predictImage(file);
        setResult(prediction);
        setStatus("idle");
        onPredicted(prediction, url);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Classification failed.");
        setStatus("error");
      }
    },
    [onPredicted]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-panel">
      <header className="border-b border-slate-100 px-5 py-4">
        <h2 className="font-display text-lg font-semibold text-ink">Lesion classification</h2>
        <p className="mt-0.5 text-sm text-slate-500">DINOv2 fine-tuned image classifier</p>
      </header>

      <div className="p-5">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-6 py-10 text-center transition-colors ${
            dragActive ? "border-clinic-teal bg-clinic-teal/5" : "border-slate-300 hover:border-slate-400"
          }`}
        >
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Uploaded dermoscopic lesion"
              className="mb-4 max-h-56 rounded-md border border-slate-200 object-contain"
            />
          ) : null}
          <p className="text-sm font-medium text-slate-700">
            {imageUrl ? "Replace image" : "Drop a dermoscopic image here"}
          </p>
          <p className="mt-1 text-xs text-slate-400">JPG or PNG · or click to browse</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </div>

        {status === "loading" && (
          <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
            <span className="h-3 w-3 animate-pulse rounded-full bg-clinic-teal" />
            Classifying lesion…
          </div>
        )}

        {status === "error" && error && (
          <div className="mt-4 rounded-md border border-clinic-red/30 bg-clinic-red/5 px-3 py-2 text-sm text-clinic-red">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-5 rounded-md border border-slate-200 bg-slate-50 px-4 py-4">
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Predicted diagnosis
              </span>
              <span className="font-mono text-xs text-slate-500">{result.confidence.toFixed(1)}%</span>
            </div>
            <p className="mt-1 font-display text-xl font-semibold text-ink">{result.predicted_class}</p>

            {/* Confidence readout — instrument-style tick rail */}
            <div className="tick-rail relative mt-4 h-2 rounded-full">
              <div
                className="absolute -top-1 h-4 w-[3px] rounded-full bg-clinic-teal transition-all"
                style={{ left: `calc(${Math.min(Math.max(result.confidence, 0), 100)}% - 1.5px)` }}
              />
            </div>
            <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-400">
              <span>0</span>
              <span>50</span>
              <span>100</span>
            </div>

            <p className="mt-3 text-xs leading-relaxed text-slate-500">
              Model output is a decision-support estimate, not a diagnosis. Confirm findings clinically.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
