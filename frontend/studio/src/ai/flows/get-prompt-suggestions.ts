
'use server';

/**
 * @fileOverview Defines the data structures for the prompt suggestion generation task.
 * This file contains the Zod schemas for validating the result of the task.
 */

import {z} from 'genkit';

const SuggestionResponseSchema = z.object({
  category: z.string(),
  title: z.string(),
  description: z.string(),
  priority_score: z.number(),
});

const GeneratePromptSuggestionsOutputSchema = z.array(SuggestionResponseSchema);

export type SuggestionResponse = z.infer<typeof SuggestionResponseSchema>;
export type GeneratePromptSuggestionsOutput = z.infer<typeof GeneratePromptSuggestionsOutputSchema>;

export type GeneratePromptSuggestionsInput = {
    currentPrompt: string;
    userComments?: string;
    targetedContext?: string;
};
