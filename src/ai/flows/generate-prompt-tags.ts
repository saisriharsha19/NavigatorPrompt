
'use server';

/**
 * @fileOverview Defines the data structure for the prompt analysis and tagging task.
 * This file contains the Zod schema for validating the result of the task.
 */

import {z} from 'genkit';

// Note: The main logic is now handled by the Celery worker in the Python backend.
// These schemas are used by the frontend to validate the final result from the /tasks/{task_id} endpoint.

const AnalysisResponseSchema = z.object({
    summary: z.string(),
    tags: z.array(z.string()),
    quality_indicators: z.record(z.any()), // Simplified for client-side use
    category_analysis: z.record(z.any()), // Simplified for client-side use
});

export type AnalysisResponse = z.infer<typeof AnalysisResponseSchema>;

// This remains for compatibility with the action's expected naming, but the core type is AnalysisResponse.
export const GeneratePromptMetadataOutputSchema = AnalysisResponseSchema;
export type GeneratePromptMetadataOutput = AnalysisResponse;

// Input types are now defined directly in the action/component that calls the API
export type GeneratePromptMetadataInput = {
    promptText: string;
    targetedContext?: string;
};
