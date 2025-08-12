
'use server';

/**
 * @fileOverview Defines the data structures for evaluating a prompt.
 * This file contains the Zod schemas that validate the shape of the data
 * returned by the backend's prompt evaluation task.
 */

import {z} from 'genkit';

const EvaluationResponseSchema = z.object({
  original_prompt: z.string(),
  improved_prompt: z.string(),
  improvement_summary: z.string(),
  bias_score: z.number(),
  toxicity_score: z.number(),
  alignment_score: z.number(),
});

export type EvaluationResponse = z.infer<typeof EvaluationResponseSchema>;

export const EvaluateAndIteratePromptOutputSchema = EvaluationResponseSchema;
export type EvaluateAndIteratePromptOutput = EvaluationResponse;

export type EvaluateAndIteratePromptInput = {
    prompt: string;
    userNeeds: string;
    retrievedContent?: string;
    groundTruths?: string;
};
