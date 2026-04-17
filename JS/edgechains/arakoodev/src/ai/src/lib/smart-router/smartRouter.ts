import { OpenAI } from "../openai/openai.js";
import { GeminiAI } from "../gemini/gemini.js";
import { LlamaAI } from "../llama/llama.js";
import { ChatModel, role } from "../../types/index";

interface Provider {
    chat(options: any): Promise<any>;
}

interface SmartRouterConstructionOptions {
    openaiApiKey?: string;
    openaiOrgId?: string;
    geminiApiKey?: string;
    llamaApiKey?: string;
    priority?: string[];
    customProviders?: { [key: string]: Provider };
}

interface SmartRouterChatOptions {
    model: string;
    prompt?: string;
    messages?: Array<{ role: role; content: string; name?: string }>;
    max_tokens?: number;
    temperature?: number;
    provider?: string;
}

interface SmartRouterResponse {
    content: string;
    provider: string;
    model: string;
}

interface ProviderPattern {
    name: string;
    models: RegExp[];
    provider: Provider;
}

interface DetectionResult {
    provider: string;
    model: string;
    confidence: number;
}

export class SmartRouter {
    private providers: Provider[];
    private providerPatterns: ProviderPattern[];
    private priority: string[];
    private customProviders: { [key: string]: Provider };

    constructor(options: SmartRouterConstructionOptions = {}) {
        this.providers = [];
        this.providerPatterns = [];
        this.priority = options.priority || ['openai', 'gemini', 'llama'];
        this.customProviders = options.customProviders || {};

        // Initialize providers with API keys
        if (options.openaiApiKey) {
            const openai = new OpenAI({ apiKey: options.openaiApiKey, orgId: options.openaiOrgId });
            this.registerProvider('openai', [
                /^gpt-.*$/,
                /^o[13]-.*$/,
                /^chatgpt-.*$/,
                /^text-embedding-.*$/,
                /^dall-e-.*$/
            ], openai);
        }

        if (options.geminiApiKey) {
            const gemini = new GeminiAI({ apiKey: options.geminiApiKey });
            this.registerProvider('gemini', [
                /^gemini-.*$/,
                /^palm-.*$/,
                /^bison-.*$/
            ], gemini);
        }

        if (options.llamaApiKey) {
            const llama = new LlamaAI({ apiKey: options.llamaApiKey });
            this.registerProvider('llama', [
                /^llama-.*$/,
                /^meta-llama\/.*$/,
                /^mixtral-.*$/,
                /^mistral-.*$/,
                /^qwen-.*$/,
                /^deepseek-.*$/
            ], llama);
        }

        // Register custom providers
        for (const [name, provider] of Object.entries(this.customProviders)) {
            this.registerProvider(name, [], provider);
        }
    }

    private registerProvider(name: string, patterns: RegExp[], provider: Provider): void {
        this.providers.push(provider);
        this.providerPatterns.push({ name, models: patterns, provider });
    }

    private detectProvider(model: string): DetectionResult {
        // Check for exact provider specification first
        if (model.includes('://')) {
            const [provider] = model.split('://');
            return { provider, model, confidence: 1.0 };
        }

        // Match against patterns
        for (const { name, models } of this.providerPatterns) {
            for (const pattern of models) {
                if (pattern.test(model)) {
                    return { provider: name, model, confidence: 0.9 };
                }
            }
        }

        // Default fallback
        return { provider: this.priority[0] || 'openai', model, confidence: 0.5 };
    }

    private async tryProvider(
        provider: Provider, 
        options: SmartRouterChatOptions, 
        providerName: string
    ): Promise<SmartRouterResponse | null> {
        try {
            let chatOptions: any = {
                model: options.model,
                prompt: options.prompt,
                messages: options.messages,
                max_tokens: options.max_tokens,
                temperature: options.temperature
            };

            // Map to provider-specific options if needed
            if (providerName === 'openai') {
                chatOptions.role = options.messages?.[0]?.role || 'user';
            }

            const response = await provider.chat(chatOptions);
            
            // Normalize response to standard format
            return {
                content: typeof response === 'string' ? response : response.content || response.text || '',
                provider: providerName,
                model: options.model
            };
        } catch (error) {
            console.warn(`Provider ${providerName} failed:`, error);
            return null;
        }
    }

    async chat(options: SmartRouterChatOptions): Promise<SmartRouterResponse> {
        if (!options.model) {
            throw new Error('Model is required');
        }

        // If specific provider requested, use it directly
        if (options.provider) {
            const providerPattern = this.providerPatterns.find(p => p.name === options.provider);
            if (providerPattern) {
                const result = await this.tryProvider(
                    providerPattern.provider, 
                    options, 
                    options.provider
                );
                if (result) {
                    return result;
                }
                throw new Error(`Provider ${options.provider} failed and no fallback available`);
            }
            throw new Error(`Unknown provider: ${options.provider}`);
        }

        // Auto-detect provider and try in priority order
        const detection = this.detectProvider(options.model);
        const providersToTry = this.priority.map(p => 
            this.providerPatterns.find(pp => pp.name === p)?.provider
        ).filter(Boolean) as Provider[];

        for (const provider of providersToTry) {
            const providerName = this.providerPatterns.find(p => p.provider === provider)?.name;
            if (providerName) {
                const result = await this.tryProvider(provider, options, providerName);
                if (result) {
                    return result;
                }
            }
        }

        throw new Error(`All providers failed for model: ${options.model}`);
    }

    listProviders(): string[] {
        return [...new Set(this.providerPatterns.map(p => p.name))];
    }

    isModelSupported(model: string): boolean {
        const detection = this.detectProvider(model);
        return detection.confidence > 0;
    }

    detectProviderFromModel(model: string): string {
        return this.detectProvider(model).provider;
    }
}