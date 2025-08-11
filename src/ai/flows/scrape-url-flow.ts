
'use server';
/**
 * @fileOverview A flow to scrape and extract text content from a given URL by calling an external crawler API.
 * 
 * - scrapeUrl - A function that handles the web scraping process.
 * - ScrapeUrlInput - The input type for the scrapeUrl function.
 * - ScrapeUrlOutput - The return type for the scrapeUrl function.
 */
import { ai } from '@/ai/genkit';
import { z } from 'genkit';

const ScrapeUrlInputSchema = z.object({
  url: z.string().url('Please provide a valid URL.'),
  sitemapUrl: z.string().url().optional().describe('Direct URL to a sitemap.xml file.'),
  includeSubdomains: z.boolean().optional().default(false).describe('Whether to automatically discover and scrape subdomains'),
  maxSubdomains: z.number().int().min(1).max(50).optional().default(10).describe('Maximum number of subdomains to scrape (1-50)'),
  maxPages: z.number().int().min(1).max(1000).optional().default(100).describe('Maximum pages per domain (1-1000)'),
  maxConcurrent: z.number().int().min(1).max(50).optional().default(10).describe('Maximum concurrent requests (1-50)'),
  delayBetweenRequests: z.number().min(0.1).max(10.0).optional().default(1.0).describe('Delay between requests in seconds (0.1-10.0)'),
  maxMemoryMb: z.number().int().min(100).max(2000).optional().default(500).describe('Maximum memory usage in MB (100-2000)'),
  preferSitemap: z.boolean().optional().default(true).describe('Prefer sitemap over manual crawling'),
  includeContent: z.boolean().optional().default(true).describe('Include full content in response'),
  includeLinks: z.boolean().optional().default(true).describe('Include extracted links in response'),
  includeMetadata: z.boolean().optional().default(true).describe('Include metadata in response'),
});

export type ScrapeUrlInput = z.infer<typeof ScrapeUrlInputSchema>;

const SubdomainResultSchema = z.object({
  subdomain: z.string(),
  pages_crawled: z.number(),
  content_preview: z.string(),
  summary: z.string().optional(),
  error: z.string().optional(),
});

const CrawlResultSchema = z.object({
  url: z.string(),
  title: z.string(),
  content: z.string().optional(),
  summary: z.string().optional(),
  metadata: z.record(z.any()).optional(),
  crawl_time: z.string(),
  content_hash: z.string(),
  word_count: z.number(),
  links: z.array(z.string()).optional(),
  error: z.string().nullable().optional(),
});

const CrawlMetadataSchema = z.object({
  crawl_time: z.string(),
  base_url: z.string(),
  total_pages: z.number(),
  discovered_subdomains: z.array(z.string()),
  sitemap_urls_found: z.number(),
  crawl_method: z.string(),
  resource_stats: z.record(z.any()),
  processing_time_seconds: z.number().optional(),
});

const ScrapeUrlOutputSchema = z.object({
  success: z.boolean(),
  message: z.string(),
  metadata: CrawlMetadataSchema.optional(),
  results: z.array(CrawlResultSchema).optional(),
  subdomains_results: z.array(SubdomainResultSchema).optional(),
  sitemap_urls: z.array(z.string()).optional(),
  // Legacy fields for backward compatibility
  content: z.string().optional().describe('Combined content from all results'),
  summary: z.string().optional().describe('Combined summary from all results'),
  subdomains: z.array(SubdomainResultSchema).optional().describe('Alias for subdomains_results'),
});

export type ScrapeUrlOutput = z.infer<typeof ScrapeUrlOutputSchema>;

export async function scrapeUrl(input: ScrapeUrlInput): Promise<ScrapeUrlOutput> {
  return scrapeUrlFlow(input);
}

const scrapeUrlFlow = ai.defineFlow(
  {
    name: 'scrapeUrlFlow',
    inputSchema: ScrapeUrlInputSchema,
    outputSchema: ScrapeUrlOutputSchema,
  },
  async ({ 
    url, 
    includeSubdomains = false, 
    maxSubdomains = 10, 
    sitemapUrl,
    maxPages = 100,
    maxConcurrent = 10,
    delayBetweenRequests = 1.0,
    maxMemoryMb = 500,
    preferSitemap = true,
    includeContent = true,
    includeLinks = true,
    includeMetadata = true,
  }) => {
    const crawlerApiUrl = process.env.CRAWLER_API_URL;
    if (!crawlerApiUrl) {
        throw new Error('Crawler API URL is not configured. Please set CRAWLER_API_URL in your environment variables.');
    }
    
    try {
        // Map client parameters to API parameters
        const payload = {
            base_url: url,
            max_subdomains: includeSubdomains ? maxSubdomains : 0,
            max_pages: maxPages,
            max_concurrent: maxConcurrent,
            delay_between_requests: delayBetweenRequests,
            max_memory_mb: maxMemoryMb,
            prefer_sitemap: preferSitemap,
            sitemap_override: sitemapUrl,
            include_content: includeContent,
            include_links: includeLinks,
            include_metadata: includeMetadata,
        };

        const response = await fetch(crawlerApiUrl+'/scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(180000), // 3-minute timeout for the crawl
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`Crawler API request failed with status ${response.status}: ${errorBody}`);
        }

        const result = await response.json();
        
        // Validate that we have the expected structure
        if (!result.success) {
            throw new Error(`Crawler API returned error: ${result.message || 'Unknown error'}`);
        }

        const resultsArray = result.results || [];
        
        // Create legacy content field for backward compatibility
        const combinedContent = resultsArray
            .filter((r: any) => r.content)
            .map((r: any) => `URL: ${r.url}\nTitle: ${r.title}\n\n${r.content}`)
            .join('\n\n---\n\n');

        // Create legacy summary field for backward compatibility  
        const combinedSummary = resultsArray
            .filter((r: any) => r.summary)
            .map((r: any) => r.summary)
            .join('\n\n');
        
        const subdomainsResults = result.subdomains_results || [];

        // Return the full API response with legacy fields added
        return {
            success: result.success,
            message: result.message,
            metadata: result.metadata,
            results: resultsArray,
            subdomains_results: subdomainsResults,
            sitemap_urls: result.sitemap_urls || [],
            // Legacy fields for backward compatibility
            content: combinedContent,
            summary: combinedSummary,
            subdomains: subdomainsResults, // Alias for backward compatibility
        };

    } catch (error: any) {
        console.error(`Error calling crawler API for ${url}:`, error);
        
        if (error.name === 'TimeoutError') {
            throw new Error('The request to the crawler service timed out after 3 minutes.');
        }
        
        if (error.name === 'AbortError') {
            throw new Error('The request to the crawler service was aborted.');
        }
        
        throw new Error(`Failed to process URL via crawler service. Reason: ${error.message}`);
    }
  }
);
