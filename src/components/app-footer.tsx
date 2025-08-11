import Link from 'next/link';

export function AppFooter() {
  return (
    <footer className="mt-auto border-t bg-background">
      <div className="container mx-auto flex max-w-7xl items-center justify-between py-4 px-4 sm:px-6 lg:px-8">
        <p className="text-sm text-muted-foreground">
          &copy; 2025 University of Florida
        </p>
        <Link
          href="https://www.ufl.edu"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-muted-foreground hover:text-foreground"
          aria-label="University of Florida website (opens in a new tab)"
        >
          www.ufl.edu
        </Link>
      </div>
    </footer>
  );
}
