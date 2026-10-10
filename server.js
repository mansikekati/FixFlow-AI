'use strict';

require('dotenv').config();

const express = require('express');
const cors    = require('cors');
const fetch   = require('node-fetch');
const path    = require('path');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '256kb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ─── IBM watsonx.ai helpers ──────────────────────────────────────────────────

const WATSONX_REGION     = process.env.WATSONX_REGION || 'us-south';
const WATSONX_API_KEY    = process.env.WATSONX_API_KEY || '';
const WATSONX_PROJECT_ID = process.env.WATSONX_PROJECT_ID || '';

const WATSONX_INFERENCE_URL =
  `https://${WATSONX_REGION}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29`;

const IAM_TOKEN_URL = 'https://iam.cloud.ibm.com/identity/token';

let _iamToken = null;
let _iamExpiry = 0;

async function getIAMToken() {
  if (_iamToken && Date.now() < _iamExpiry - 60_000) return _iamToken;
  const res = await fetch(IAM_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=${encodeURIComponent(WATSONX_API_KEY)}`,
  });
  if (!res.ok) throw new Error(`IAM token request failed: ${res.status}`);
  const data = await res.json();
  _iamToken  = data.access_token;
  _iamExpiry = Date.now() + data.expires_in * 1000;
  return _iamToken;
}

async function callWatsonx(prompt) {
  const token = await getIAMToken();
  const body = {
    model_id: 'meta-llama/llama-3-3-70b-instruct',
    project_id: WATSONX_PROJECT_ID,
    input: prompt,
    parameters: {
      decoding_method: 'greedy',
      max_new_tokens: 1200,
      stop_sequences: ['<|end|>'],
      temperature: 0.2,
    },
  };
  const res = await fetch(WATSONX_INFERENCE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`watsonx error ${res.status}: ${text}`);
  }
  const data = await res.json();
  return data?.results?.[0]?.generated_text || '';
}

const watsonxAvailable = () => WATSONX_API_KEY && WATSONX_PROJECT_ID;

// ─── Prompt builders ─────────────────────────────────────────────────────────

function buildDiagnosisPrompt(language, code, errorMsg) {
  return `You are FixFlow AI, an expert coding tutor for beginner developers.
A ${language} developer has the following error.

ERROR MESSAGE:
${errorMsg}

CODE:
\`\`\`${language}
${code}
\`\`\`

Respond ONLY with a valid JSON object (no markdown fences) with these exact keys:
{
  "errorType": "short name of the error type",
  "plainExplanation": "explain the error in 2-3 simple sentences a beginner can understand",
  "faultyLine": "the exact line(s) of code causing the problem, or empty string if unknown",
  "fixedCode": "the complete corrected code",
  "conceptTitle": "name of the underlying programming concept",
  "conceptExplanation": "explain the concept in 3-4 sentences: what it is, why the fix works, how to avoid this mistake",
  "quizQuestion": "a short multiple-choice question to test understanding",
  "quizOptions": ["option A", "option B", "option C", "option D"],
  "quizAnswer": 0
}
quizAnswer is the 0-based index of the correct option. Do not include any text outside the JSON.`;
}

function buildQuizFeedbackPrompt(language, concept, question, options, correctIdx, userIdx) {
  const correct  = options[correctIdx];
  const selected = options[userIdx];
  const isRight  = correctIdx === userIdx;
  return `You are FixFlow AI, a coding tutor.
Concept: ${concept} (${language})
Quiz question: "${question}"
Correct answer: "${correct}"
Student chose: "${selected}" which is ${isRight ? 'CORRECT' : 'WRONG'}.

Write a 2-3 sentence encouraging feedback message. If wrong, explain why "${correct}" is correct in simple terms.
Respond with plain text only – no JSON, no markdown.`;
}

// ─── Demo-mode responses ─────────────────────────────────────────────────────

const DEMO_RESPONSES = {
  python_nameerror: {
    errorType: 'NameError',
    plainExplanation:
      "Python can't find a variable named 'greet' because it was never defined before you used it. Variables must be created (assigned a value) before you can call or use them.",
    faultyLine: 'greet()',
    fixedCode: `def greet():\n    print("Hello, World!")\n\ngreet()`,
    conceptTitle: 'Function Definition vs. Function Call',
    conceptExplanation:
      "In Python, a function must be *defined* using the `def` keyword before it can be *called*. A definition creates the function; a call executes it. If you call a function before defining it, Python raises a NameError because the name doesn't exist yet. Always define your functions above the lines that call them.",
    quizQuestion: 'What keyword is used to define a function in Python?',
    quizOptions: ['function', 'def', 'define', 'fn'],
    quizAnswer: 1,
  },
  js_typeerror: {
    errorType: 'TypeError',
    plainExplanation:
      "JavaScript is trying to read the property 'length' from `undefined`. This happens when a variable you expect to hold an array or string is actually `undefined`—often because a function returned nothing or a fetch didn't complete.",
    faultyLine: 'console.log(items.length)',
    fixedCode: `const items = [];\nconsole.log(items.length); // 0`,
    conceptTitle: 'Undefined vs. Null and Defensive Programming',
    conceptExplanation:
      '`undefined` means a variable was declared but never assigned a value. Accessing properties on `undefined` throws a TypeError. You can guard against this with a nullish check (`if (items)`) or use optional chaining (`items?.length`). Always initialise variables to a sensible default.',
    quizQuestion: "What does `undefined?.length` return in JavaScript?",
    quizOptions: ['Throws a TypeError', '0', 'undefined', 'null'],
    quizAnswer: 2,
  },
  python_indentation: {
    errorType: 'IndentationError',
    plainExplanation:
      'Python uses indentation (spaces or tabs at the start of a line) to define code blocks. One of your lines is not indented correctly—either it is missing indentation inside a function/if block, or it has unexpected extra spaces.',
    faultyLine: 'return result',
    fixedCode: `def add(a, b):\n    result = a + b\n    return result\n\nprint(add(3, 4))`,
    conceptTitle: 'Python Indentation and Code Blocks',
    conceptExplanation:
      'Unlike languages that use curly braces `{}`, Python uses consistent indentation to group statements into blocks. Every statement inside a function, loop, or conditional must be indented by the same amount. The standard is 4 spaces per level. Mixing tabs and spaces also causes IndentationError.',
    quizQuestion: 'What is the recommended indentation size in Python (PEP 8)?',
    quizOptions: ['2 spaces', '4 spaces', '8 spaces', '1 tab'],
    quizAnswer: 1,
  },
  js_syntaxerror: {
    errorType: 'SyntaxError',
    plainExplanation:
      "JavaScript found a character or token it didn't expect. This is usually a missing bracket, parenthesis, or quotation mark. The parser stopped understanding your code at that point.",
    faultyLine: 'console.log("Hello World"',
    fixedCode: `console.log("Hello World");`,
    conceptTitle: 'JavaScript Syntax and Delimiters',
    conceptExplanation:
      'Every opening symbol like `(`, `[`, or `{` must have a matching closing symbol `)`, `]`, or `}`. JavaScript reads your code character by character; when delimiters are unbalanced, it raises a SyntaxError. Always close every bracket you open, and use a code editor with bracket-matching highlighting.',
    quizQuestion: 'Which of the following will cause a SyntaxError?',
    quizOptions: [
      'console.log("hi");',
      'let x = 5;',
      'if (true { console.log("yes"); }',
      'const arr = [1, 2, 3];',
    ],
    quizAnswer: 2,
  },
};

function pickDemoResponse(language, errorMsg, code) {
  const lang = language.toLowerCase();
  const err  = (errorMsg + code).toLowerCase();
  if (lang === 'python' && err.includes('indent')) return DEMO_RESPONSES.python_indentation;
  if (lang === 'python') return DEMO_RESPONSES.python_nameerror;
  if (lang === 'javascript' && err.includes('syntax')) return DEMO_RESPONSES.js_syntaxerror;
  return DEMO_RESPONSES.js_typeerror;
}

// ─── Routes ──────────────────────────────────────────────────────────────────

// Health / mode check
app.get('/api/status', (_req, res) => {
  res.json({
    mode: watsonxAvailable() ? 'live' : 'demo',
    watsonxConfigured: !!watsonxAvailable(),
  });
});

// Main analysis endpoint
app.post('/api/analyze', async (req, res) => {
  const { language, code, errorMsg } = req.body;

  if (!language || typeof language !== 'string') {
    return res.status(400).json({ error: 'language is required.' });
  }
  if (!code || typeof code !== 'string' || code.trim().length < 3) {
    return res.status(400).json({ error: 'code must be at least 3 characters.' });
  }
  if (!errorMsg || typeof errorMsg !== 'string' || errorMsg.trim().length < 2) {
    return res.status(400).json({ error: 'errorMsg is required.' });
  }

  // Demo mode
  if (!watsonxAvailable()) {
    const demo = pickDemoResponse(language, errorMsg, code);
    return res.json({ ...demo, demo: true });
  }

  // Live watsonx
  try {
    const prompt = buildDiagnosisPrompt(language, code.trim(), errorMsg.trim());
    const raw    = await callWatsonx(prompt);

    // Extract JSON from response (model may wrap with text)
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) throw new Error('Model did not return valid JSON');
    const parsed = JSON.parse(match[0]);

    // Sanity-check required keys
    const required = [
      'errorType','plainExplanation','faultyLine','fixedCode',
      'conceptTitle','conceptExplanation','quizQuestion','quizOptions','quizAnswer',
    ];
    for (const k of required) {
      if (!(k in parsed)) throw new Error(`Missing key: ${k}`);
    }
    return res.json({ ...parsed, demo: false });
  } catch (err) {
    console.error('[analyze]', err.message);
    return res.status(500).json({
      error: 'AI analysis failed. Please try again or use a demo example.',
      detail: err.message,
    });
  }
});

// Quiz feedback endpoint
app.post('/api/quiz-feedback', async (req, res) => {
  const { language, conceptTitle, quizQuestion, quizOptions, quizAnswer, userAnswer } = req.body;

  if (typeof quizAnswer !== 'number' || typeof userAnswer !== 'number') {
    return res.status(400).json({ error: 'quizAnswer and userAnswer must be numbers.' });
  }

  const isCorrect = quizAnswer === userAnswer;

  if (!watsonxAvailable()) {
    const fb = isCorrect
      ? `Excellent work! You correctly identified "${quizOptions?.[quizAnswer]}". That understanding will help you avoid this class of bug in the future. Keep it up!`
      : `Not quite — the correct answer is "${quizOptions?.[quizAnswer]}". Review the concept explanation above and try to connect it to your code. You'll get it next time!`;
    return res.json({ feedback: fb, correct: isCorrect, demo: true });
  }

  try {
    const prompt = buildQuizFeedbackPrompt(
      language, conceptTitle, quizQuestion, quizOptions, quizAnswer, userAnswer
    );
    const feedback = await callWatsonx(prompt);
    return res.json({ feedback: feedback.trim(), correct: isCorrect, demo: false });
  } catch (err) {
    console.error('[quiz-feedback]', err.message);
    const fb = isCorrect
      ? `Correct! "${quizOptions?.[quizAnswer]}" is the right answer.`
      : `The correct answer is "${quizOptions?.[quizAnswer]}". Review the explanation to understand why.`;
    return res.json({ feedback: fb, correct: isCorrect, demo: true });
  }
});

// Fallback: serve index.html for any unknown route
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  const mode = watsonxAvailable() ? '🤖 Live watsonx.ai mode' : '🎭 Demo mode (no API key configured)';
  console.log(`\n  FixFlow AI server running on http://localhost:${PORT}`);
  console.log(`  ${mode}\n`);
});
