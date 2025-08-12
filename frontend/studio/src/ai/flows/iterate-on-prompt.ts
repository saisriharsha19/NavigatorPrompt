
'use server';

/**
 * @fileOverview Defines the data structure for the prompt iteration task.
 * NOTE: This flow is deprecated and the new backend logic does not map to this.
 */

import {z} from 'genkit';

const IterateOnPromptOutputSchema = z.object({
  newPrompt: z.string().describe('The newly generated, refined system prompt.'),
});

export type IterateOnPromptOutput = z.infer<typeof IterateOnPromptOutputSchema>;

export type IterateOnPromptInput = {
  currentPrompt: string;
  userComments: string;
  selectedSuggestions: string[];
};
