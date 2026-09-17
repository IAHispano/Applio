export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
