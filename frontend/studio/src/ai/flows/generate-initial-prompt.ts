
'use server';

/**
 * @fileOverview Defines the data structure for the initial prompt generation task.
 * This file contains the Zod schema for validating the result of the task.
 */

import {z} from 'genkit';

const GeneratedPromptResponseSchema = z.object({
  user_needs: z.string(),
  initial_prompt: z.string(),
});

export type GeneratedPromptResponse = z.infer<typeof GeneratedPromptResponseSchema>;

export const GenerateInitialPromptOutputSchema = GeneratedPromptResponseSchema;
export type GenerateInitialPromptOutput = GeneratedPromptResponse;

export type GenerateInitialPromptInput = {
    userNeeds: string;
    deepevalContext?: string;
};
