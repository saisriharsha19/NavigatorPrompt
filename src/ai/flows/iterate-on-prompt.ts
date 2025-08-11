
'use server';

/**
 * @fileOverview Defines the data structure for the prompt iteration task.
 * This file contains the Zod schema for validating the result of the task.
 * NOTE: The new backend doesn't have a dedicated "iterate" task. This flow is now deprecated
 * and will be handled by combining suggestion generation and manual editing on the client.
 * For now, we will create a schema that matches the old structure to avoid breaking the UI completely,
 * but this should be refactored.
 */

import {z} from 'genkit';

// This is a placeholder schema to avoid breaking imports.
// The new backend logic doesn't map directly to this flow anymore.
// Iteration is now a combination of getting suggestions and the user applying them.
const IterateOnPromptOutputSchema = z.object({
  newPrompt: z.string().describe('The newly generated, refined system prompt.'),
});

export type IterateOnPromptOutput = z.infer<typeof IterateOnPromptOutputSchema>;

// Input types are now defined directly in the action/component that calls the API
export type IterateOnPromptInput = {
  currentPrompt: string;
  userComments: string;
  selectedSuggestions: string[];
};
