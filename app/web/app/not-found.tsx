import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center p-6 space-y-4">
      <h2 className="text-xl font-bold text-white">Page Not Found</h2>
      <p className="text-sm text-neutral-400">The requested page could not be found.</p>
      <Link href="/" className="cta px-4 py-2 text-xs font-semibold">
        Return Home
      </Link>
    </div>
  );
}
