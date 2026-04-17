import { vi, describe, it, expect, beforeEach } from 'vitest';
import { SmartRouter } from '../../../../dist/ai/src/lib/smart-router/smartRouter.js';

// Mock provider classes
class MockProvider {
    constructor(public name: string, public shouldFail = false) {}
    
    async chat(options: any) {
        if (this.shouldFail) {
            throw new Error(`Provider ${this.name} failed`);
        }
        
        return {
            content: `Response from ${this.name} for model ${options.model}`,
            provider: this.name
        };
    }
}

describe('SmartRouter', () => {
    let smartRouter: SmartRouter;
    let mockOpenAI: MockProvider;
    let mockGemini: MockProvider;
    let mockLlama: MockProvider;

    beforeEach(() => {
        mockOpenAI = new MockProvider('openai');
        mockGemini = new MockProvider('gemini');
        mockLlama = new MockProvider('llama');

        smartRouter = new SmartRouter({
            openaiApiKey: 'test-openai-key',
            geminiApiKey: 'test-gemini-key',
            llamaApiKey: 'test-llama-key',
            priority: ['openai', 'gemini', 'llama']
        });

        // Replace providers with mocks
        const router = (smartRouter as any);
        router.providers = [mockOpenAI, mockGemini, mockLlama];
        router.providerPatterns = [
            { name: 'openai', models: [/^gpt-.*/], provider: mockOpenAI },
            { name: 'gemini', models: [/^gemini-.*/], provider: mockGemini },
            { name: 'llama', models: [/^llama-.*/], provider: mockLlama }
        ];
    });

    describe('Provider Detection', () => {
        it('should detect OpenAI models', () => {
            const detection = (smartRouter as any).detectProvider('gpt-4o');
            expect(detection.provider).toBe('openai');
            expect(detection.confidence).toBe(0.9);
        });

        it('should detect Gemini models', () => {
            const detection = (smartRouter as any).detectProvider('gemini-pro');
            expect(detection.provider).toBe('gemini');
            expect(detection.confidence).toBe(0.9);
        });

        it('should detect Llama models', () => {
            const detection = (smartRouter as any).detectProvider('llama-2');
            expect(detection.provider).toBe('llama');
            expect(detection.confidence).toBe(0.9);
        });

        it('should fallback to first priority for unknown models', () => {
            const detection = (smartRouter as any).detectProvider('unknown-model');
            expect(detection.provider).toBe('openai');
            expect(detection.confidence).toBe(0.5);
        });

        it('should detect provider from explicit model specification', () => {
            const detection = (smartRouter as any).detectProvider('openai://gpt-4');
            expect(detection.provider).toBe('openai');
            expect(detection.confidence).toBe(1.0);
        });
    });

    describe('Chat Functionality', () => {
        it('should route to correct provider based on model name', async () => {
            const response = await smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world'
            });

            expect(response.provider).toBe('openai');
            expect(response.content).toContain('Response from openai for model gpt-4o');
        });

        it('should use explicit provider when specified', async () => {
            const response = await smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world',
                provider: 'gemini'
            });

            expect(response.provider).toBe('gemini');
            expect(response.content).toContain('Response from gemini for model gpt-4o');
        });

        it('should fallback to next provider when primary fails', async () => {
            mockOpenAI.shouldFail = true;
            
            const response = await smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world'
            });

            expect(response.provider).toBe('gemini');
            expect(response.content).toContain('Response from gemini for model gpt-4o');
        });

        it('should throw error when all providers fail', async () => {
            mockOpenAI.shouldFail = true;
            mockGemini.shouldFail = true;
            mockLlama.shouldFail = true;

            await expect(smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world'
            })).rejects.toThrow('All providers failed for model: gpt-4o');
        });

        it('should handle messages array', async () => {
            const response = await smartRouter.chat({
                model: 'gpt-4o',
                messages: [
                    { role: 'user' as const, content: 'Hello' },
                    { role: 'assistant' as const, content: 'Hi there!' }
                ]
            });

            expect(response.provider).toBe('openai');
            expect(response.content).toContain('Response from openai for model gpt-4o');
        });

        it('should pass through additional options', async () => {
            const response = await smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world',
                max_tokens: 100,
                temperature: 0.5
            });

            expect(response.provider).toBe('openai');
            expect(response.content).toContain('Response from openai for model gpt-4o');
        });
    });

    describe('Utility Methods', () => {
        it('should list all available providers', () => {
            const providers = smartRouter.listProviders();
            expect(providers).toContain('openai');
            expect(providers).toContain('gemini');
            expect(providers).toContain('llama');
        });

        it('should check if model is supported', () => {
            expect(smartRouter.isModelSupported('gpt-4o')).toBe(true);
            expect(smartRouter.isModelSupported('gemini-pro')).toBe(true);
            expect(smartRouter.isModelSupported('llama-2')).toBe(true);
            expect(smartRouter.isModelSupported('unknown-model')).toBe(true); // Falls back to openai
        });

        it('should detect provider from model', () => {
            expect(smartRouter.detectProviderFromModel('gpt-4o')).toBe('openai');
            expect(smartRouter.detectProviderFromModel('gemini-pro')).toBe('gemini');
            expect(smartRouter.detectProviderFromModel('llama-2')).toBe('llama');
        });
    });

    describe('Error Handling', () => {
        it('should throw error when no model provided', async () => {
            await expect(smartRouter.chat({
                prompt: 'Hello world'
            })).rejects.toThrow('Model is required');
        });

        it('should throw error for unknown explicit provider', async () => {
            await expect(smartRouter.chat({
                model: 'gpt-4o',
                prompt: 'Hello world',
                provider: 'unknown-provider'
            })).rejects.toThrow('Unknown provider: unknown-provider');
        });
    });
});