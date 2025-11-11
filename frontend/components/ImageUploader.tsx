import React, { useState, useRef } from "react";

type Props = {
  onFileSelected?: (file: File) => void;
};

const apiBase = process.env.NEXT_PUBLIC_API_URL;

function base64ToUint8(b64: string): Uint8Array {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function toPngDataUrlFromMatrix(matrix: number[][]): string {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  if (!rows || !cols) return "";

  // compute min/max for normalization
  let min = Infinity,
    max = -Infinity;
  for (let r = 0; r < rows; r++) {
    const row = matrix[r];
    for (let c = 0; c < cols; c++) {
      const v = row[c];
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }
  const norm = (v: number) => {
    if (!Number.isFinite(v)) return 0;
    if (max === min) return 0;
    const t = Math.max(0, Math.min(1, (v - min) / (max - min)));
    return Math.round(t * 255);
  };

  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(cols, rows);
  const data = imageData.data;

  let idx = 0;
  for (let r = 0; r < rows; r++) {
    const row = matrix[r];
    for (let c = 0; c < cols; c++) {
      const g = norm(row[c]);
      // data[idx++] = g; // R
      // data[idx++] = g; // G
      // data[idx++] = g; // B
      const r = 255;
      const gb = 255 - g; // g=0 => white (255,255,255); g=255 => red (255,0,0)
      data[idx++] = r; // R
      data[idx++] = gb; // G
      data[idx++] = gb; // B

      data[idx++] = 255; // A
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function toPngDataUrlFromTyped(
  typed: Float32Array | Float64Array,
  rows: number,
  cols: number
): string {
  if (typed.length !== rows * cols) return "";

  // min/max
  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < typed.length; i++) {
    const v = typed[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }
  const norm = (v: number) => {
    if (!Number.isFinite(v)) return 0;
    if (max === min) return 0;
    const t = Math.max(0, Math.min(1, (v - min) / (max - min)));
    return Math.round(t * 255);
  };

  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(cols, rows);
  const data = imageData.data;

  let idx = 0;
  for (let i = 0; i < typed.length; i++) {
    const g = norm(typed[i]);
    // data[idx++] = g; // R
    // data[idx++] = g; // G
    // data[idx++] = g; // B
    const r = 255;
    const gb = 255 - g; // g=0 => white (255,255,255); g=255 => red (255,0,0)
    data[idx++] = r; // R
    data[idx++] = gb; // G
    data[idx++] = gb; // B

    data[idx++] = 255; // A
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

const ImageUploader: React.FC<Props> = ({ onFileSelected }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [attributionsUrl, setAttributionsUrl] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement | null>(null);

  console.log("NEXT_PUBLIC_API_URL", apiBase);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    setError(null);
    setScore(null);
    setAttributionsUrl(null);

    if (file && file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      onFileSelected?.(file);
      await uploadImage(file);
    }
  };

  const uploadImage = async (file: File) => {
    if (!apiBase) {
      setError(
        "API URL not configured. Set NEXT_PUBLIC_API_URL in .env.local."
      );
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", file, file.name);

      const resp = await fetch(`${apiBase}/post-img/`, {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json" },
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);

      const ct = resp.headers.get("content-type") || "";
      let value: number | null = null;

      if (ct.includes("application/json")) {
        const json = await resp.json();

        // Parse scalar first
        if (typeof json === "number") {
          value = json;
        } else if (
          json &&
          (typeof json.value !== "undefined" ||
            typeof json.score !== "undefined")
        ) {
          const raw = json.value ?? json.score;
          value = typeof raw === "number" ? raw : parseFloat(String(raw));
        } else {
          const raw = Array.isArray(json) ? json[0] : Object.values(json)[0];
          value = typeof raw === "number" ? raw : parseFloat(String(raw));
        }

        // Build attributions image URL only if lengths match exactly
        let attributionsUrlLocal: string | null = null;
        if (
          json &&
          typeof json.array_b64 === "string" &&
          Array.isArray(json.shape)
        ) {
          const bytes = base64ToUint8(json.array_b64 as string);
          const [rows, cols] = json.shape as [number, number];
          const dtype = String(json.dtype || "float32").toLowerCase();

          const elemSize = dtype === "float64" ? 8 : 4;
          const expectedBytes = rows * cols * elemSize;
          if (bytes.byteLength !== expectedBytes) {
            throw new Error(
              `Payload size mismatch: got ${bytes.byteLength} bytes; expected ${expectedBytes} (${rows}×${cols}×${elemSize}).`
            );
          }

          // Construct typed array with exact length
          let typed: Float32Array | Float64Array;
          if (dtype === "float64") {
            typed = new Float64Array(bytes.buffer, 0, rows * cols);
          } else {
            typed = new Float32Array(bytes.buffer, 0, rows * cols);
          }

          attributionsUrlLocal = toPngDataUrlFromTyped(typed, rows, cols);
        } else if (json && Array.isArray(json.my_array)) {
          attributionsUrlLocal = toPngDataUrlFromMatrix(
            json.my_array as number[][]
          );
        }

        if (attributionsUrlLocal) setAttributionsUrl(attributionsUrlLocal);
      } else {
        const text = await resp.text();
        value = parseFloat(text.trim());
      }

      if (value === null || Number.isNaN(value)) {
        throw new Error("API returned an unparseable value.");
      }
      setScore(value);
    } catch (err: any) {
      setError(err?.message ?? "Request failed.");
      setAttributionsUrl(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPreview(null);
    setScore(null);
    setError(null);
    setAttributionsUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="uploader">
      <input
        ref={inputRef}
        id="image-input"
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden-input"
      />
      <label htmlFor="image-input" className="dropzone">
        <span className="dz-title">Upload an image</span>
        <span className="dz-sub">JPEG or PNG</span>
      </label>

      {preview && (
        <div className="preview">
          <div
            className="img-row"
            style={{
              display: "grid",
              gridTemplateColumns: attributionsUrl ? "1fr 1fr" : "1fr",
              gap: ".75rem",
              justifyItems: "center",
              alignItems: "start",
            }}
          >
            <figure style={{ margin: 0 }}>
              <img src={preview} alt="Input" />
              <figcaption style={{ color: "#666", marginTop: ".25rem" }}>
                Input
              </figcaption>
            </figure>

            {attributionsUrl && (
              <figure style={{ margin: 0 }}>
                <img src={attributionsUrl} alt="Attributions" />
                <figcaption style={{ color: "#666", marginTop: ".25rem" }}>
                  Attributions
                </figcaption>
              </figure>
            )}
          </div>

          <button className="clear" onClick={handleClear} disabled={loading}>
            {loading ? "Processing…" : "Clear"}
          </button>

          {loading && <div className="result">Processing…</div>}
          {error && (
            <div className="result" style={{ color: "#b00020" }}>
              {error}
            </div>
          )}
          {score !== null && !loading && !error && (
            <div className="result">Estimated Fear: {score.toFixed(0)}/100</div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
