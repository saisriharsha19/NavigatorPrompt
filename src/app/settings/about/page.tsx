import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Image from 'next/image';
import Link from 'next/link';

export default function AboutPage() {
  return (
    <Card>
      <CardHeader className="items-center text-center">
          <Image
              src="/NavGAI-19.png"
              width={60}
              height={60}
              alt="Navigator Logo"
          />
        <CardTitle className="text-2xl">Navigator Prompt</CardTitle>
        <CardDescription>Version 1.0.0</CardDescription>
      </CardHeader>
      <CardContent className="text-center text-sm text-muted-foreground">
        <p>
          This application is designed to facilitate the creation, evaluation, and iteration of AI prompts.
        </p>
        <p className="mt-4">
          &copy; 2025 University of Florida. All rights reserved.
        </p>
         <p className="mt-2">
          Visit our website at{' '}
          <Link href="https://www.ufl.edu" target="_blank" rel="noopener noreferrer" className="font-medium text-primary underline-offset-4 hover:underline">
            www.ufl.edu
          </Link>
        </p>
      </CardContent>
    </Card>
  );
}
