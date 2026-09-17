import { type Request, type Response, Router } from "express";
import { getJob, listJobs } from "../jobs";

const router = Router();

router.get("/", (_req: Request, res: Response) => {
  res.json({ jobs: listJobs().map((j) => ({ ...j, logs: j.logs.slice(-20) })) });
});

router.get("/:id", (req: Request, res: Response) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json({ job });
});

export default router;
