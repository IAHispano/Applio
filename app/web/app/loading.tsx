// Shown while a tab's chunk is still loading, so navigation gives immediate
// feedback instead of leaving the previous page on screen.
export default function Loading() {
  return (
    <div className="h-full w-full flex items-center justify-center">
      <div className="h-6 w-6 rounded-full border-2 border-white/15 border-t-white/70 animate-spin" />
    </div>
  );
}
