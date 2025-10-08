/**
 * ============================================================================
 * Commitlint Configuration for AG News Text Classification
 * ============================================================================
 * Project: AG News Text Classification (ag-news-text-classification)
 * Description: Enforce conventional commit message format
 * Author: Võ Hải Dũng
 * License: MIT
 * ============================================================================
 * 
 * This configuration enforces commit messages following:
 * - Conventional Commits specification v1.0.0
 * - Semantic Versioning 2.0.0
 * - Custom rules for ML/Research workflows
 * 
 * Commit Message Format:
 * <type>(<scope>): <subject>
 * 
 * <body>
 * 
 * <footer>
 * 
 * Type:
 *   - feat: New feature (MINOR version bump)
 *   - fix: Bug fix (PATCH version bump)
 *   - docs: Documentation changes
 *   - style: Code formatting, whitespace
 *   - refactor: Code restructuring
 *   - perf: Performance improvements
 *   - test: Testing changes
 *   - build: Build system changes
 *   - ci: CI/CD configuration
 *   - chore: Maintenance tasks
 *   - revert: Revert previous commit
 *   - model: Model architecture changes (ML-specific)
 *   - data: Dataset/pipeline changes (ML-specific)
 *   - train: Training strategy changes (ML-specific)
 *   - eval: Evaluation changes (ML-specific)
 *   - exp: Experiment tracking (Research-specific)
 *   - paper: Paper implementation (Research-specific)
 * 
 * Scope (optional):
 *   Component or module affected (e.g., models, data, api, ui)
 * 
 * Subject:
 *   Brief description in imperative mood (e.g., "add feature" not "added")
 * 
 * Body (optional):
 *   Detailed explanation of changes, motivation, implementation details
 * 
 * Footer (optional):
 *   Breaking changes, issue references
 * 
 * Examples:
 *   feat(models): add DeBERTa-v3-xlarge with LoRA fine-tuning
 *   fix(data): correct label mapping for AG News classes
 *   docs(readme): update installation instructions for CUDA 12.1
 *   model(ensemble): implement weighted voting with confidence scores
 *   train(lora): optimize rank selection for memory efficiency
 *   exp(sota): achieve 97.8% accuracy with ensemble approach
 * 
 * Breaking Changes:
 *   feat(api)!: change prediction response format to include probabilities
 *   
 *   BREAKING CHANGE: Response format changed from array to object
 * 
 * References:
 * - Conventional Commits: https://www.conventionalcommits.org/
 * - Semantic Versioning: https://semver.org/
 * - commitlint: https://commitlint.js.org/
 * - Angular Commit Guidelines: https://github.com/angular/angular/blob/main/CONTRIBUTING.md
 * ============================================================================
 */

module.exports = {
  // Extend conventional commits configuration
  extends: ['@commitlint/config-conventional'],
  
  // Custom rules for AG News Text Classification project
  rules: {
    // ========================================================================
    // Type Rules
    // ========================================================================
    
    /**
     * Allowed commit types
     * 
     * Severity: error (2)
     * Condition: always
     * Value: array of allowed types
     */
    'type-enum': [
      2,
      'always',
      [
        // Standard Conventional Commits types
        'feat',       // New feature for users
        'fix',        // Bug fix for users
        'docs',       // Documentation only changes
        'style',      // Formatting, missing semicolons, etc.
        'refactor',   // Code change that neither fixes bug nor adds feature
        'perf',       // Code change that improves performance
        'test',       // Adding or updating tests
        'build',      // Changes to build system or dependencies
        'ci',         // Changes to CI configuration files and scripts
        'chore',      // Other changes that do not modify src or test files
        'revert',     // Reverts a previous commit
        
        // Machine Learning specific types
        'model',      // Model architecture or configuration changes
        'data',       // Dataset or data pipeline changes
        'train',      // Training configuration or strategy changes
        'eval',       // Evaluation metrics or methods changes
        'opt',        // Optimization changes (hyperparameters, etc.)
        
        // Research specific types
        'exp',        // Experiment tracking and results
        'paper',      // Paper implementation or reference
        'bench',      // Benchmark results
        'ablation',   // Ablation study results
        
        // Project specific types
        'config',     // Configuration changes
        'deploy',     // Deployment changes
        'api',        // API changes
        'ui',         // UI/Frontend changes
        'db',         // Database schema or migration
        'security',   // Security improvements
      ],
    ],
    
    /**
     * Type must be lowercase
     */
    'type-case': [2, 'always', 'lower-case'],
    
    /**
     * Type cannot be empty
     */
    'type-empty': [2, 'never'],
    
    // ========================================================================
    // Scope Rules
    // ========================================================================
    
    /**
     * Allowed commit scopes
     * 
     * Scopes represent the affected component or module
     */
    'scope-enum': [
      2,
      'always',
      [
        // Core modules
        'core',
        'utils',
        'cli',
        'api',
        'app',
        'config',
        
        // Data modules
        'data',
        'datasets',
        'preprocessing',
        'augmentation',
        'loaders',
        'sampling',
        'selection',
        'validation',
        
        // Model modules
        'models',
        'transformers',
        'deberta',
        'roberta',
        'electra',
        'xlnet',
        'llm',
        'llama',
        'mistral',
        'ensemble',
        'voting',
        'stacking',
        'efficient',
        'lora',
        'qlora',
        'adapters',
        'prompt',
        
        // Training modules
        'training',
        'trainers',
        'strategies',
        'objectives',
        'losses',
        'optimizers',
        'schedulers',
        'callbacks',
        'regularization',
        'distillation',
        
        // Evaluation modules
        'evaluation',
        'metrics',
        'analysis',
        'visualization',
        'interpretability',
        
        // Overfitting prevention
        'overfitting',
        'validators',
        'monitors',
        'constraints',
        'guards',
        
        // Experiments
        'experiments',
        'baselines',
        'benchmarks',
        'ablation',
        'hyperopt',
        'sota',
        
        // Infrastructure
        'docker',
        'kubernetes',
        'k8s',
        'ci',
        'cd',
        'github',
        'actions',
        
        // Configuration
        'configs',
        'env',
        'deps',
        'requirements',
        
        // Documentation
        'readme',
        'docs',
        'guides',
        'tutorials',
        'examples',
        'notebooks',
        
        // Testing
        'tests',
        'unit',
        'integration',
        'e2e',
        'performance',
        'regression',
        
        // Deployment
        'deployment',
        'serving',
        'inference',
        'monitoring',
        
        // UI
        'streamlit',
        'gradio',
        'dashboard',
        
        // Release
        'release',
        'version',
        'changelog',
        
        // Security
        'security',
        'auth',
        
        // Misc
        'scripts',
        'tools',
        'migrations',
      ],
    ],
    
    /**
     * Scope must be lowercase
     */
    'scope-case': [2, 'always', 'lower-case'],
    
    /**
     * Scope is optional but recommended
     */
    'scope-empty': [1, 'never'],
    
    // ========================================================================
    // Subject Rules
    // ========================================================================
    
    /**
     * Subject cannot be empty
     */
    'subject-empty': [2, 'never'],
    
    /**
     * Subject must not end with period
     */
    'subject-full-stop': [2, 'never', '.'],
    
    /**
     * Subject must be in imperative mood (lowercase start)
     * 
     * Good: "add feature"
     * Bad: "Added feature", "Adding feature"
     */
    'subject-case': [
      2,
      'never',
      ['sentence-case', 'start-case', 'pascal-case', 'upper-case'],
    ],
    
    /**
     * Subject maximum length
     */
    'subject-max-length': [2, 'always', 72],
    
    /**
     * Subject minimum length
     */
    'subject-min-length': [2, 'always', 10],
    
    /**
     * Subject should not contain issue references
     * Use footer for issue references
     */
    'subject-exclamation-mark': [1, 'never'],
    
    // ========================================================================
    // Body Rules
    // ========================================================================
    
    /**
     * Body must have blank line before it
     */
    'body-leading-blank': [2, 'always'],
    
    /**
     * Body line maximum length
     */
    'body-max-line-length': [2, 'always', 100],
    
    /**
     * Body should be in sentence case
     */
    'body-case': [1, 'always', 'sentence-case'],
    
    /**
     * Body minimum length (if present)
     */
    'body-min-length': [1, 'always', 20],
    
    // ========================================================================
    // Footer Rules
    // ========================================================================
    
    /**
     * Footer must have blank line before it
     */
    'footer-leading-blank': [2, 'always'],
    
    /**
     * Footer line maximum length
     */
    'footer-max-line-length': [2, 'always', 100],
    
    // ========================================================================
    // Header Rules
    // ========================================================================
    
    /**
     * Header (type + scope + subject) maximum length
     */
    'header-max-length': [2, 'always', 100],
    
    /**
     * Header minimum length
     */
    'header-min-length': [2, 'always', 20],
    
    /**
     * Header must be lowercase
     */
    'header-case': [2, 'always', 'lower-case'],
    
    // ========================================================================
    // Reference Rules
    // ========================================================================
    
    /**
     * References must match issue pattern
     * Example: Fixes #123, Closes #456
     */
    'references-empty': [1, 'never'],
    
    // ========================================================================
    // Custom Rules for Personal Project
    // ========================================================================
    
    /**
     * Signed-off-by is not required for personal project
     */
    'signed-off-by': [0, 'always'],
    
    /**
     * Trailer (Co-authored-by, etc.) is optional
     * Not enforced for personal project
     */
    'trailer-exists': [0, 'always'],
  },
  
  // Parser configuration
  parserPreset: {
    parserOpts: {
      // Header pattern: type(scope): subject
      headerPattern: /^(\w+)(?:KATEX_INLINE_OPEN([a-z0-9-]+)KATEX_INLINE_CLOSE)?!?: (.+)$/,
      
      // Breaking change pattern: type(scope)!: subject
      breakingHeaderPattern: /^(\w+)(?:KATEX_INLINE_OPEN([a-z0-9-]+)KATEX_INLINE_CLOSE)?!: (.+)$/,
      
      // Header correspondence
      headerCorrespondence: ['type', 'scope', 'subject'],
      
      // Note keywords for breaking changes
      noteKeywords: ['BREAKING CHANGE', 'BREAKING-CHANGE'],
      
      // Revert pattern
      revertPattern: /^(?:Revert|revert:)\s"?([\s\S]+?)"?\s*This reverts commit (\w{7,40})\./i,
      revertCorrespondence: ['header', 'hash'],
      
      // Issue reference pattern
      issuePrefixes: ['#', 'fixes #', 'closes #', 'refs #'],
      
      // Field pattern
      fieldPattern: /^-(.*?)-$/,
      
      // Merge pattern
      mergePattern: /^Merge pull request #(\d+) from (.*)$/,
      mergeCorrespondence: ['id', 'source'],
    },
  },
  
  // Ignore patterns
  ignores: [
    (commit) => commit.includes('WIP'),
    (commit) => commit.includes('[skip ci]'),
  ],
  
  // Default ignore rules
  defaultIgnores: true,
  
  // Help URL
  helpUrl: 'https://github.com/VoHaiDung/ag-news-text-classification/blob/main/docs/contributing.md#commit-message-format',
  
  // Prompt configuration for interactive commit
  prompt: {
    settings: {},
    messages: {
      skip: ':skip',
      max: 'upper %d chars',
      min: '%d chars at least',
      emptyWarning: 'can not be empty',
      upperLimitWarning: 'over limit',
      lowerLimitWarning: 'below limit',
    },
    questions: {
      type: {
        description: "Select the type of change you are committing",
        enum: {
          feat: {
            description: 'A new feature',
            title: 'Features',
            emoji: '',
          },
          fix: {
            description: 'A bug fix',
            title: 'Bug Fixes',
            emoji: '',
          },
          docs: {
            description: 'Documentation only changes',
            title: 'Documentation',
            emoji: '',
          },
          style: {
            description: 'Changes that do not affect the meaning of the code',
            title: 'Styles',
            emoji: '',
          },
          refactor: {
            description: 'A code change that neither fixes a bug nor adds a feature',
            title: 'Code Refactoring',
            emoji: '',
          },
          perf: {
            description: 'A code change that improves performance',
            title: 'Performance Improvements',
            emoji: '',
          },
          test: {
            description: 'Adding missing tests or correcting existing tests',
            title: 'Tests',
            emoji: '',
          },
          build: {
            description: 'Changes that affect the build system or external dependencies',
            title: 'Builds',
            emoji: '',
          },
          ci: {
            description: 'Changes to our CI configuration files and scripts',
            title: 'Continuous Integrations',
            emoji: '',
          },
          chore: {
            description: "Other changes that don't modify src or test files",
            title: 'Chores',
            emoji: '',
          },
          revert: {
            description: 'Reverts a previous commit',
            title: 'Reverts',
            emoji: '',
          },
          model: {
            description: 'Model architecture or configuration changes',
            title: 'Model Changes',
            emoji: '',
          },
          data: {
            description: 'Dataset or data pipeline changes',
            title: 'Data Changes',
            emoji: '',
          },
          train: {
            description: 'Training configuration or strategy changes',
            title: 'Training Changes',
            emoji: '',
          },
          eval: {
            description: 'Evaluation metrics or methods changes',
            title: 'Evaluation Changes',
            emoji: '',
          },
          exp: {
            description: 'Experiment tracking and results',
            title: 'Experiments',
            emoji: '',
          },
          paper: {
            description: 'Paper implementation or reference',
            title: 'Paper Implementation',
            emoji: '',
          },
        },
      },
      scope: {
        description: 'What is the scope of this change (e.g., models, data, api)',
      },
      subject: {
        description: 'Write a short, imperative tense description of the change',
      },
      body: {
        description: 'Provide a longer description of the change',
      },
      isBreaking: {
        description: 'Are there any breaking changes?',
      },
      breakingBody: {
        description: 'A BREAKING CHANGE commit requires a body. Please enter a longer description of the commit itself',
      },
      breaking: {
        description: 'Describe the breaking changes',
      },
      isIssueAffected: {
        description: 'Does this change affect any open issues?',
      },
      issuesBody: {
        description: 'If issues are closed, the commit requires a body. Please enter a longer description of the commit itself',
      },
      issues: {
        description: 'Add issue references (e.g., "fix #123", "re #123")',
      },
    },
  },
};

/**
 * ============================================================================
 * Example Valid Commit Messages for Personal Project
 * ============================================================================
 * 
 * Feature with scope:
 *   feat(models): add DeBERTa-v3-xlarge with LoRA fine-tuning
 * 
 * Bug fix:
 *   fix(data): correct label encoding for AG News dataset
 * 
 * Documentation:
 *   docs(readme): update installation instructions for CUDA 12.1
 * 
 * Performance improvement:
 *   perf(training): optimize batch processing for 2x speedup
 * 
 * Model change:
 *   model(ensemble): implement weighted voting ensemble
 * 
 * Data change:
 *   data(augmentation): add back-translation pipeline
 * 
 * Training change:
 *   train(lora): optimize rank selection for memory efficiency
 * 
 * Experiment:
 *   exp(sota): achieve 97.8% accuracy with ensemble approach
 * 
 * Breaking change:
 *   feat(api)!: change prediction response format
 *   
 *   BREAKING CHANGE: Response format changed from array to object.
 *   Migration guide available in docs/migrations/v2.0.0.md
 * 
 * With body and footer:
 *   fix(training): resolve OOM error in DeBERTa-xlarge training
 *   
 *   The model was consuming too much GPU memory due to inefficient
 *   gradient accumulation. Implemented gradient checkpointing and
 *   reduced effective batch size.
 *   
 *   Fixes #123
 * 
 * Research experiment:
 *   exp(ablation): compare LoRA ranks 8, 16, 32, 64
 *   
 *   Conducted ablation study on LoRA rank selection. Results show
 *   rank 16 provides best accuracy/efficiency trade-off.
 *   
 *   Refs #456
 * 
 * Multiple scopes:
 *   feat(models,training): add QLoRA 4-bit quantization support
 * 
 * Revert:
 *   revert: feat(models): add experimental attention mechanism
 *   
 *   This reverts commit abc123def456.
 *   The experimental attention caused training instability.
 * 
 * Configuration change:
 *   config(lora): update default rank from 32 to 16
 *   
 *   After benchmarking, rank 16 provides better memory efficiency
 *   with minimal accuracy loss for AG News dataset.
 *   
 *   Fixes #789
 * 
 * Security update:
 *   security(api): add rate limiting to prediction endpoint
 *   
 *   Implemented token bucket algorithm to prevent API abuse.
 *   Default limit: 100 requests per minute per IP.
 * 
 * Deployment:
 *   deploy(docker): optimize image size with multi-stage build
 *   
 *   Reduced Docker image size from 8GB to 3GB using multi-stage
 *   build and selective dependency installation.
 * 
 * UI improvement:
 *   ui(streamlit): add interactive model comparison dashboard
 *   
 *   New dashboard allows side-by-side comparison of model
 *   predictions with attention visualization.
 * 
 * Chore:
 *   chore(deps): update transformers to 4.37.2
 *   
 *   Updated transformers library to latest version for bug fixes
 *   and performance improvements.
 * 
 * ============================================================================
 * Personal Project Best Practices
 * ============================================================================
 * 
 * For a personal project, keep commits:
 * - Clear and descriptive
 * - Focused on single logical change
 * - Following conventional commits format
 * - Including context in body when needed
 * - Referencing issues when applicable
 * 
 * You can be more flexible with:
 * - Not requiring co-author signatures
 * - Less formal tone in body
 * - More detailed experimental notes
 * - Personal reminders in commit messages
 * 
 * But still maintain:
 * - Type and scope conventions
 * - Subject line clarity
 * - Breaking change notifications
 * - Issue references
 * ============================================================================
 */
