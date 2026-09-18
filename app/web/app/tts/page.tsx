"use client";

import TtsForm from "../../components/convert/TtsForm";
import PageHeader from "../../components/layout/PageHeader";
import { useI18n } from "../../lib/i18n";

export default function TtsPage() {
  const { t } = useI18n();
  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("TTS")}
        description={t("Synthesize speech from text and convert it to your selected target voice.")}
      />
      <TtsForm />
    </div>
  );
}
