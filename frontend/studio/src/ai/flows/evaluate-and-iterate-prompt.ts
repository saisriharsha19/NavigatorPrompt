
'use server';

/**
 * @fileOverview Defines the data structures for evaluating a prompt.
 * This file contains the Zod schemas that validate the shape of the data
 * returned by the backend's prompt evaluation task.
 */

import {z} from 'genkit';

// Note: The main logic is now handled by the Celery worker in the Python backend.
// These schemas are used by the frontend to validate the final result from the /tasks/{task_id} endpoint.

const EvaluationResponseSchema = z.object({
  original_prompt: z.string(),
  improved_prompt: z.string(),
  improvement_summary: z.string(),
  bias_score: z.number(),
  toxicity_score: z.number(),
  alignment_score: z.number(),
});

export type EvaluationResponse = z.infer<typeof EvaluationResponseSchema>;

// This remains for compatibility with the action's expected naming, but the core type is EvaluationResponse.
export const EvaluateAndIteratePromptOutputSchema = EvaluationResponseSchema;
export type EvaluateAndIteratePromptOutput = EvaluationResponse;

// Input types are now defined directly in the action/component that calls the API
export type EvaluateAndIteratePromptInput = {
    prompt: string;
    userNeeds: string;
    retrievedContent?: string;
    groundTruths?: string;
};
