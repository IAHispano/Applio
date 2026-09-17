"use client";

import { useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import RadioRow from "../../components/RadioRow";
import { apiGet, errMsg, fetchModels, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { useSpeakers } from "../../lib/useSpeakers";

interface Voice {
  shortName: string;
  friendlyName: string;
  gender: string;
  locale: string;
}

export default function TtsPage() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const { t } = useI18n();
  const [filter, setFilter] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [rate, setRate] = useState(0);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [embedderModelCustom, setEmbedderModelCustom] = useState("");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [f0AutotuneStrength, setF0AutotuneStrength] = useState(1);
  const [proposedPitch, setProposedPitch] = useState(false);
  const [proposedPitchThreshold, setProposedPitchThreshold] = useState(155);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.5);
  const [sid, setSid] = useState(0);
  const [terms, setTerms] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const speakers = useSpeakers(pthPath);

  useEffect(() => {
    if (!speakers.includes(sid)) setSid(0);
  }, [speakers, sid]);

  useEffect(() => {
    apiGet<{ voices: Voice[] }>("/api/tts/voices")
      .then((v) => {
        setVoices(v.voices);
        if (v.voices[0]) setVoice(v.voices[0].shortName);
      })
      .catch((e) => setError(String(e?.message || e)));
    fetchModels()
      .then((m) => {
        setModels(m.models);
        if (m.models[0]) setPthPath(m.models[0]);
      })
      .catch(() => {});
  }, []);

  const shown = voices.filter(
    (v) =>
      !filter ||
      v.shortName.toLowerCase().includes(filter.toLowerCase()) ||
      v.friendlyName.toLowerCase().includes(filter.toLowerCase()),
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!terms) {
      setError(t("You must agree to the Terms of Use to proceed."));
      return;
    }
    if (!text && !file) {
      setError(t("Enter text or upload a .txt file."));
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("ttsText", text);
      if (file) fd.append("txt_file", file);
      fd.append("ttsVoice", voice);
      fd.append("ttsRate", String(rate));
      fd.append("pthPath", pthPath);
      fd.append("indexPath", indexPath);
      fd.append("pitch", String(pitch));
      fd.append("indexRate", String(indexRate));
      fd.append("volumeEnvelope", String(volumeEnvelope));
      fd.append("protect", String(protect));
      fd.append("f0Method", f0Method);
      fd.append("embedderModel", embedderModel);
      if (embedderModel === "custom" && embedderModelCustom) {
        fd.append("embedderModelCustom", embedderModelCustom);
      }
      fd.append("exportFormat", exportFormat);
      fd.append("splitAudio", String(splitAudio));
      fd.append("f0Autotune", String(f0Autotune));
      fd.append("f0AutotuneStrength", String(f0AutotuneStrength));
      fd.append("proposedPitch", String(proposedPitch));
      fd.append("proposedPitchThreshold", String(proposedPitchThreshold));
      fd.append("cleanAudio", String(cleanAudio));
      fd.append("cleanStrength", String(cleanStrength));
      fd.append("sid", String(sid));
      const { jobId: id } = await postForm<{ jobId: string }>("/api/tts", fd);
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || t("Submit failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <PageHeader
        title={t("TTS")}
        description={t("Synthesize speech from text and convert it to your selected target voice.")}
      />
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <form onSubmit={onSubmit}>
        <div className="card">
          <h2>{t("Text")}</h2>
          <label>{t("Text to Synthesize")}</label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t("Hello, this is Applio.")}
          />
          <label>{t("Upload a .txt file")}</label>
          <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <div className="grid2">
            <div>
              <label>{t("Voice filter")}</label>
              <input
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder={t("e.g. en-US, Aria, Guy…")}
              />
            </div>
            <div>
              <label>
                Voice ({shown.length} of {voices.length})
              </label>
              <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                {shown.slice(0, 400).map((v) => (
                  <option key={v.shortName} value={v.shortName}>
                    {v.friendlyName} ({v.gender})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>Speaking rate: {rate} (−100…100)</label>
              <input
                type="range"
                min={-100}
                max={100}
                step={1}
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
              />
            </div>
          </div>
        </div>
        <div className="card">
          <h2>{t("Voice Model")}</h2>
          <div className="grid2">
            <div>
              <label>{t("Voice Model")}</label>
              <input
                type="text"
                list="tmodels"
                value={pthPath}
                onChange={(e) => setPthPath(e.target.value)}
              />
              <datalist id="tmodels">
                {models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </div>
            <div>
              <label>{t("Index (optional)")}</label>
              <input type="text" value={indexPath} onChange={(e) => setIndexPath(e.target.value)} />
            </div>
            <div>
              <label>Pitch: {pitch}</label>
              <input
                type="range"
                min={-24}
                max={24}
                step={1}
                value={pitch}
                onChange={(e) => setPitch(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Search Feature Ratio: {indexRate}</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={indexRate}
                onChange={(e) => setIndexRate(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Volume Envelope: {volumeEnvelope} (default 1)</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volumeEnvelope}
                onChange={(e) => setVolumeEnvelope(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Protect Voiceless Consonants: {protect} (default 0.5)</label>
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={protect}
                onChange={(e) => setProtect(Number(e.target.value))}
              />
            </div>
            <RadioRow
              label={t("Pitch extraction algorithm")}
              name="f0-tts"
              options={["crepe", "crepe-tiny", "rmvpe", "fcpe"]}
              value={f0Method}
              onChange={setF0Method}
            />
            <RadioRow
              label={t("Embedder Model")}
              name="embedder-tts"
              options={[
                "contentvec",
                "spin",
                "spin-v2",
                "chinese-hubert-base",
                "japanese-hubert-base",
                "korean-hubert-base",
                "custom",
              ]}
              value={embedderModel}
              onChange={setEmbedderModel}
            />
            {embedderModel === "custom" && (
              <div>
                <label>{t("Custom embedder path")}</label>
                <input
                  type="text"
                  value={embedderModelCustom}
                  onChange={(e) => setEmbedderModelCustom(e.target.value)}
                  placeholder="rvc/models/embedders/embedders_custom/my-embedder"
                />
              </div>
            )}
            <div>
              <label>{t("Speaker ID")}</label>
              <select value={sid} onChange={(e) => setSid(Number(e.target.value))}>
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>{t("Export Format")}</label>
              <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
                {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <details>
            <summary>{t("Advanced Settings")}</summary>
            <div className="row">
              <label>
                <input
                  type="checkbox"
                  checked={splitAudio}
                  onChange={(e) => setSplitAudio(e.target.checked)}
                />{" "}
                {t("Split Audio")}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={f0Autotune}
                  onChange={(e) => setF0Autotune(e.target.checked)}
                />{" "}
                {t("Autotune")}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={proposedPitch}
                  onChange={(e) => setProposedPitch(e.target.checked)}
                />{" "}
                {t("Proposed Pitch")}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={cleanAudio}
                  onChange={(e) => setCleanAudio(e.target.checked)}
                />{" "}
                {t("Clean Audio")}
              </label>
            </div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <div>
                <label>Autotune Strength: {f0AutotuneStrength} (default 1)</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={f0AutotuneStrength}
                  onChange={(e) => setF0AutotuneStrength(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Proposed Pitch Threshold: {proposedPitchThreshold} (default 155)</label>
                <input
                  type="range"
                  min={50}
                  max={1200}
                  step={1}
                  value={proposedPitchThreshold}
                  onChange={(e) => setProposedPitchThreshold(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Clean Strength: {cleanStrength} (default 0.5)</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={cleanStrength}
                  onChange={(e) => setCleanStrength(Number(e.target.value))}
                />
              </div>
            </div>
          </details>
          <label className="terms">
            <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
            <span>{t("I agree to the terms of use")}</span>
          </label>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="submit" className="cta" disabled={busy}>
              {busy ? t("Submitting…") : t("Convert")}
            </button>
          </div>
        </div>
      </form>
      <JobPanel jobId={jobId} />
    </div>
  );
}
