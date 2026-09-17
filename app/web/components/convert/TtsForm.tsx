"use client";

import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchModels, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { useSpeakers } from "../../lib/useSpeakers";
import JobPanel from "../JobPanel";
import RadioRow from "../RadioRow";
import SliderField from "../ui/SliderField";

interface Voice {
  shortName: string;
  friendlyName: string;
  gender: string;
  locale: string;
}

export default function TtsForm() {
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
      .catch((e) => setError(errMsg(e)));
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
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mb-4 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
        >
          {error}
        </div>
      )}
      <form onSubmit={onSubmit}>
        <div className="card">
          <h2>{t("Text")}</h2>
          <label htmlFor="tts-text-input">{t("Text to Synthesize")}</label>
          <input
            id="tts-text-input"
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t("Hello, this is Applio.")}
          />
          <label htmlFor="tts-file-input">{t("Upload a .txt file")}</label>
          <input
            id="tts-file-input"
            type="file"
            accept=".txt"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <div className="grid2">
            <div>
              <label htmlFor="tts-voice-filter">{t("Voice filter")}</label>
              <input
                id="tts-voice-filter"
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder={t("e.g. en-US, Aria, Guy…")}
              />
            </div>
            <div>
              <label htmlFor="tts-voice-select">
                {t("Voice")} ({shown.length} of {voices.length})
              </label>
              <select id="tts-voice-select" value={voice} onChange={(e) => setVoice(e.target.value)}>
                {shown.slice(0, 400).map((v) => (
                  <option key={v.shortName} value={v.shortName}>
                    {v.friendlyName} ({v.gender})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <SliderField
                id="tts-speaking-rate"
                label={t("Speaking rate")}
                value={rate}
                min={-100}
                max={100}
                step={1}
                unit="%"
                onChange={setRate}
              />
            </div>
          </div>
        </div>
        <div className="card">
          <h2>{t("Voice Model")}</h2>
          <div className="grid2">
            <div>
              <label htmlFor="tts-model-input">{t("Voice Model")}</label>
              <input
                id="tts-model-input"
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
              <label htmlFor="tts-index-input">{t("Index (optional)")}</label>
              <input
                id="tts-index-input"
                type="text"
                value={indexPath}
                onChange={(e) => setIndexPath(e.target.value)}
              />
            </div>
            <div>
              <SliderField
                id="tts-pitch"
                label={t("Pitch")}
                value={pitch}
                min={-24}
                max={24}
                step={1}
                unit="st"
                onChange={setPitch}
              />
            </div>
            <div>
              <SliderField
                id="tts-index-rate"
                label={t("Search Feature Ratio")}
                value={indexRate}
                min={0}
                max={1}
                step={0.05}
                onChange={setIndexRate}
              />
            </div>
            <div>
              <SliderField
                id="tts-volume-envelope"
                label={t("Volume Envelope")}
                value={volumeEnvelope}
                min={0}
                max={1}
                step={0.05}
                onChange={setVolumeEnvelope}
              />
            </div>
            <div>
              <SliderField
                id="tts-protect"
                label={t("Protect Voiceless Consonants")}
                value={protect}
                min={0}
                max={0.5}
                step={0.01}
                onChange={setProtect}
              />
            </div>
            <RadioRow
              label={t("Pitch extraction algorithm")}
              name="f0-convert"
              options={[
                "rmvpe",
                "fcpe",
                "crepe",
                "crepe-tiny",
                "hybrid[crepe+rmvpe]",
                "hybrid[crepe+fcpe]",
                "hybrid[rmvpe+fcpe]",
              ]}
              value={f0Method}
              onChange={setF0Method}
            />
            <RadioRow
              label={t("Embedder Model")}
              name="embedder-convert"
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
                <label htmlFor="tts-custom-embedder">{t("Custom embedder path")}</label>
                <input
                  id="tts-custom-embedder"
                  type="text"
                  value={embedderModelCustom}
                  onChange={(e) => setEmbedderModelCustom(e.target.value)}
                  placeholder="rvc/models/embedders/embedders_custom/my-embedder"
                />
              </div>
            )}
            <div>
              <label htmlFor="tts-speaker-id">{t("Speaker ID")}</label>
              <select id="tts-speaker-id" value={sid} onChange={(e) => setSid(Number(e.target.value))}>
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="tts-export-format">{t("Export Format")}</label>
              <select
                id="tts-export-format"
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value)}
              >
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
              <label htmlFor="tts-split-audio" className="flex items-center gap-2 cursor-pointer">
                <input
                  id="tts-split-audio"
                  type="checkbox"
                  checked={splitAudio}
                  onChange={(e) => setSplitAudio(e.target.checked)}
                />{" "}
                {t("Split Audio")}
              </label>
              <label htmlFor="tts-f0-autotune" className="flex items-center gap-2 cursor-pointer">
                <input
                  id="tts-f0-autotune"
                  type="checkbox"
                  checked={f0Autotune}
                  onChange={(e) => setF0Autotune(e.target.checked)}
                />{" "}
                {t("Autotune")}
              </label>
              <label htmlFor="tts-proposed-pitch" className="flex items-center gap-2 cursor-pointer">
                <input
                  id="tts-proposed-pitch"
                  type="checkbox"
                  checked={proposedPitch}
                  onChange={(e) => setProposedPitch(e.target.checked)}
                />{" "}
                {t("Proposed Pitch")}
              </label>
              <label htmlFor="tts-clean-audio" className="flex items-center gap-2 cursor-pointer">
                <input
                  id="tts-clean-audio"
                  type="checkbox"
                  checked={cleanAudio}
                  onChange={(e) => setCleanAudio(e.target.checked)}
                />{" "}
                {t("Clean Audio")}
              </label>
            </div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <div>
                <SliderField
                  id="tts-autotune-strength"
                  label={t("Autotune Strength")}
                  value={f0AutotuneStrength}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setF0AutotuneStrength}
                />
              </div>
              <div>
                <SliderField
                  id="tts-proposed-threshold"
                  label={t("Proposed Pitch Threshold")}
                  value={proposedPitchThreshold}
                  min={50}
                  max={1200}
                  step={1}
                  unit="Hz"
                  onChange={setProposedPitchThreshold}
                />
              </div>
              <div>
                <SliderField
                  id="tts-clean-strength"
                  label={t("Clean Strength")}
                  value={cleanStrength}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setCleanStrength}
                />
              </div>
            </div>
          </details>
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
