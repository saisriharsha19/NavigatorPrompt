
'use server';

/**
 * @fileOverview Defines the data structure for the prompt analysis and tagging task.
 * This file contains the Zod schema for validating the result of the task.
 */

import {z} from 'genkit';

const AnalysisResponseSchema = z.object({
    summary: z.string(),
    tags: z.array(z.string()),
    quality_indicators: z.record(z.any()),
    category_analysis: z.record(z.any()),
});

export type AnalysisResponse = z.infer<typeof AnalysisResponseSchema>;

export const GeneratePromptMetadataOutputSchema = AnalysisResponseSchema;
export type GeneratePromptMetadataOutput = AnalysisResponse;

export type GeneratePromptMetadataInput = {
    promptText: string;
    targetedContext?: string;
};
