"use client";

import PageHeader from "../../components/layout/PageHeader";
import BlenderPanel from "../../components/models/BlenderPanel";
import { useI18n } from "../../lib/i18n";

export default function VoiceBlenderPage() {
  const { t } = useI18n();
  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Voice Blender")}
        description={t("Fuse two voice models together with a configurable interpolation ratio.")}
      />
      <BlenderPanel />
    </div>
  );
}
