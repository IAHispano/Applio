import { type Request, type Response, Router } from "express";
import { findPython, getStatus, startInstall, startPrerequisites } from "../setup";

const router = Router();

router.get("/status", async (req: Request, res: Response) => {
  try {
    res.json(await getStatus(req.query.refresh === "1"));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  }
});

router.post("/install", (_req: Request, res: Response) => {
  try {
    const job = startInstall();
    res.status(202).json({ jobId: job.id });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  }
});

router.post("/prerequisites", async (_req: Request, res: Response) => {
  try {
    const py = await findPython();
    res.status(202).json({ jobId: startPrerequisites(py ? py.cmd : null).id });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  }
});

export default router;
