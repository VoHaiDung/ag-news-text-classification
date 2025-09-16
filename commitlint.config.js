/**
 * ================================================
 * Commitlint Configuration
 * ================================================
 *
 * Enforces conventional commit messages following:
 * - Conventional Commits specification (https://www.conventionalcommits.org/)
 * - Angular Commit Message Guidelines
 * - Semantic Versioning (https://semver.org/)
 *
 * References:
 * - commitlint documentation (https://commitlint.js.org/)
 * - "A Note About Git Commit Messages" (Tim Pope)
 * - "How to Write a Git Commit Message" (Chris Beams)
 *
 * Author: Võ Hải Dũng
 * Project: AG News Text Classification
 * License: MIT
 */

module.exports = {
  // Extend from conventional commits
  extends: ['@commitlint/config-conventional'],

  // Custom rules for AG News project
  rules: {
    // ===== Type Rules =====
    'type-enum': [
      2,
      'always',
      [
        // -- Core Types --
        'feat',     // New feature (MINOR in semver)
        'fix',      // Bug fix (PATCH in semver)
        'docs',     // Documentation changes
        'style',    // Code style changes (formatting, etc)
        'refactor', // Code refactoring (neither feat nor fix)
        'perf',     // Performance improvements
        'test',     // Testing changes
        'build',    // Build system or dependencies
        'ci',       // CI/CD configuration
        'chore',    // Maintenance tasks
        'revert',   // Revert previous commit
        
        // -- ML/DL Specific Types --
        'model',    // Model architecture changes
        'data',     // Dataset or data pipeline changes
        'train',    // Training configuration or strategy
        'eval',     // Evaluation metrics or methods
        'exp',      // Experiment tracking
        'opt',      // Optimization changes
        
        // -- Research Types --
        'paper',    // Paper implementation or reference
        'bench',    // Benchmark results
        'ablation', // Ablation studies
        
        // -- Project Specific --
        'config',   // Configuration changes
        'deploy',   // Deployment changes
        'api',      // API changes
        'ui',       // UI/Frontend changes
      ],
    ],
    
    // Type must be in lowercase
    'type-case': [2, 'always', 'lower-case'],
    
    // Type cannot be empty
    'type-empty': [2, 'never'],
    
    // ===== Scope Rules =====
    'scope-enum': [
      2,
      'always',
      [
        // -- Core Modules --
        'core',
        'utils',
        'cli',
        'api',
        'app',
        
        // -- Data Modules --
        'data',
        'datasets',
        'preprocessing',
        'augmentation',
        'loaders',
        'sampling',
        'selection',
        
        // -- Model Modules --
        'models',
        'transformers',
        'ensemble',
        'efficient',
        'prompt',
        
        // -- Training Modules --
        'training',
        'objectives',
        'callbacks',
        'strategies',
        'optimization',
        
        // -- Evaluation --
        'evaluation',
        'metrics',
        'analysis',
        'visualization',
        
        // -- Experiments --
        'experiments',
        'baselines',
        'benchmarks',
        'ablation',
        
        // -- Infrastructure --
        'docker',
        'k8s',
        'ci',
        'cd',
        'github',
        
        // -- Configuration --
        'config',
        'env',
        'deps',
        
        // -- Documentation --
        'readme',
        'docs',
        'examples',
        'tutorials',
        
        // -- Testing --
        'unit',
        'integration',
        'e2e',
        'performance',
        
        // -- Release --
        'release',
        'version',
        'changelog',
      ],
    ],
    
    // Scope must be in lowercase
    'scope-case': [2, 'always', 'lower-case'],
    
    // ===== Subject Rules =====
    // Subject cannot be empty
    'subject-empty': [2, 'never'],
    
    // Subject must not end with period
    'subject-full-stop': [2, 'never', '.'],
    
    // Subject must be in sentence case (first letter lowercase)
    'subject-case': [
      2,
      'never',
      ['sentence-case', 'start-case', 'pascal-case', 'upper-case'],
    ],
    
    // Subject max length
    'subject-max-length': [2, 'always', 72],
    
    // Subject min length
    'subject-min-length': [2, 'always', 10],
    
    // ===== Body Rules =====
    // Body must have blank line before
    'body-leading-blank': [2, 'always'],
    
    // Body line max length
    'body-max-line-length': [2, 'always', 100],
    
    // Body must be in sentence case
    'body-case': [1, 'always', 'sentence-case'],
    
    // ===== Footer Rules =====
    // Footer must have blank line before
    'footer-leading-blank': [2, 'always'],
    
    // Footer line max length
    'footer-max-line-length': [2, 'always', 100],
    
    // ===== Header Rules =====
    // Header max length
    'header-max-length': [2, 'always', 100],
    
    // Header min length
    'header-min-length': [2, 'always', 20],
  },

  // Parser presets
  parserPreset: {
    parserOpts: {
      headerPattern: /^(\w+)(?:KATEX_INLINE_OPEN(\w+)KATEX_INLINE_CLOSE)?!?: (.+)$/,
      breakingHeaderPattern: /^(\w+)(?:KATEX_INLINE_OPEN(\w+)KATEX_INLINE_CLOSE)?!: (.+)$/,
      headerCorrespondence: ['type', 'scope', 'subject'],
      noteKeywords: ['BREAKING CHANGE', 'BREAKING-CHANGE'],
      revertPattern: /^(?:Revert|revert:)\s"?([\s\S]+?)"?\s*This reverts commit (\w*)\./i,
      revertCorrespondence: ['header', 'hash'],
    },
  },

  // Help URL for more information
  helpUrl: 'https://github.com/conventional-changelog/commitlint/#what-is-commitlint',

  // Prompt configuration for interactive mode
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
        description: "Select the type of change you're committing",
        enum: {
          feat: {
            description: 'A new feature',
            title: 'Features',
          },
          fix: {
            description: 'A bug fix',
            title: 'Bug Fixes',
          },
          docs: {
            description: 'Documentation only changes',
            title: 'Documentation',
          },
          style: {
            description: 'Markup, formatting, white-space changes',
            title: 'Styles',
          },
          refactor: {
            description: 'Code change that neither fixes a bug nor adds a feature',
            title: 'Code Refactoring',
          },
          perf: {
            description: 'Code change that improves performance',
            title: 'Performance Improvements',
          },
          test: {
            description: 'Adding missing tests or correcting existing tests',
            title: 'Tests',
          },
          build: {
            description: 'Changes that affect the build system or dependencies',
            title: 'Builds',
          },
          ci: {
            description: 'Changes to CI configuration files and scripts',
            title: 'Continuous Integration',
          },
          chore: {
            description: "Other changes that don't modify src or test files",
            title: 'Chores',
          },
          revert: {
            description: 'Reverts a previous commit',
            title: 'Reverts',
          },
          model: {
            description: 'Model architecture or configuration changes',
            title: 'Model Changes',
          },
          data: {
            description: 'Dataset or data pipeline changes',
            title: 'Data Changes',
          },
          train: {
            description: 'Training configuration or strategy changes',
            title: 'Training Changes',
          },
          exp: {
            description: 'Experiment tracking or results',
            title: 'Experiments',
          },
        },
      },
    },
  },
};

/**
 * Example valid commit messages:
 * 
 * feat(models): add DeBERTa-v3 model implementation
 * fix(data): correct label mapping for AG News dataset
 * docs(readme): update installation instructions
 * perf(training): optimize batch processing for 2x speedup
 * model(ensemble): implement weighted voting ensemble
 * data(augmentation): add back-translation pipeline
 * train(strategies): implement curriculum learning
 * exp(baselines): add BERT baseline results
 * 
 * Breaking change:
 * feat(api)!: change response format for predictions
 * 
 * With body:
 * fix(models): resolve OOM error in DeBERTa training
 * 
 * The model was consuming too much memory due to gradient accumulation.
 * Implemented gradient checkpointing and reduced batch size.
 * 
 * Fixes #123
 */
