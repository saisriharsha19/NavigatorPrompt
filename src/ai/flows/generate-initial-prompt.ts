
'use server';

/**
 * @fileOverview Defines the data structure for the initial prompt generation task.
 * This file contains the Zod schema for validating the result of the task.
 */

import {z} from 'genkit';

// Note: The main logic is now handled by the Celery worker in the Python backend.
// These schemas are used by the frontend to validate the final result from the /tasks/{task_id} endpoint.

const GeneratedPromptResponseSchema = z.object({
  user_needs: z.string(),
  initial_prompt: z.string(),
});

export type GeneratedPromptResponse = z.infer<typeof GeneratedPromptResponseSchema>;

// This remains for compatibility with the action's expected naming, but the core type is GeneratedPromptResponse.
export const GenerateInitialPromptOutputSchema = GeneratedPromptResponseSchema;
export type GenerateInitialPromptOutput = GeneratedPromptResponse;

// Input types are now defined directly in the action/component that calls the API
export type GenerateInitialPromptInput = {
    userNeeds: string;
    deepevalContext?: string;
};
