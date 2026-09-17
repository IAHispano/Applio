"use client";

import PageHeader from "../../components/layout/PageHeader";
import DownloadPanel from "../../components/models/DownloadPanel";
import { useI18n } from "../../lib/i18n";

export default function DownloadPage() {
  const { t } = useI18n();
  return (
    <div>
      <PageHeader
        title={t("Download Models")}
        description={t("Download models from direct links or upload local checkpoint and index files.")}
      />
      <DownloadPanel />
    </div>
  );
}
