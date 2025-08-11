import { config } from 'dotenv';
config();

import '@/ai/flows/generate-initial-prompt.ts';
import '@/ai/flows/evaluate-and-iterate-prompt.ts';
import '@/ai/flows/optimize-prompt-with-context.ts';
import '@/ai/flows/iterate-on-prompt.ts';
import '@/ai/flows/get-prompt-suggestions.ts';
import '@/ai/flows/generate-prompt-tags.ts';
import '@/ai/flows/scrape-url-flow.ts';
