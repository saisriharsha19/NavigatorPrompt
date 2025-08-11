
'use server';

/**
 * @fileOverview Defines the data structures for the prompt suggestion generation task.
 * This file contains the Zod schemas for validating the result of the task.
 */

import {z} from 'genkit';

// Note: The main logic is now handled by the Celery worker in the Python backend.
// These schemas are used by the frontend to validate the final result from the /tasks/{task_id} endpoint.

const SuggestionResponseSchema = z.object({
  category: z.string(),
  title: z.string(),
  description: z.string(),
  priority_score: z.number(),
});

const GeneratePromptSuggestionsOutputSchema = z.array(SuggestionResponseSchema);

export type SuggestionResponse = z.infer<typeof SuggestionResponseSchema>;
export type GeneratePromptSuggestionsOutput = z.infer<typeof GeneratePromptSuggestionsOutputSchema>;


// Input types are now defined directly in the action/component that calls the API
export type GeneratePromptSuggestionsInput = {
    currentPrompt: string;
    userComments?: string;
    targetedContext?: string;
};
