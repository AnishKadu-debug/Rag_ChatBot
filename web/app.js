/**
 * Maharashtra Land Records RAG Chatbot — Frontend Application Logic
 * Vanilla JavaScript (ES6+), mobile-responsive, no external framework dependencies.
 *
 * Highlights:
 * 1. English default for first-time visitors with persistent Marathi selection in localStorage.
 * 2. Mobile-first design calibrated for Maharashtra Land Records (7/12, 8A, Ferfar mutation).
 * 3. Server-bound official document retrieval: no client-side hardcoded document IDs.
 * 4. Grounded document mode with page citations, distinct from general assistant fallback.
 * 5. Full bilingual dictionary for all UI text, placeholders, errors, and suggested questions.
 */

'use strict';

(function () {
  // ───────────────────────────────────────────────────────────────────────────
  // 1. Localization Dictionary (English & Marathi)
  // ───────────────────────────────────────────────────────────────────────────

  const I18N = {
    en: {
      brandTitle: 'Land Records AI',
      brandSubtitle: 'Maharashtra Land Records Bot',
      activeDocName: 'Official Land Records',
      activeDocTitle: 'Official Maharashtra Land Records (500+ Pages)',
      newChat: 'New Chat',
      welcomeBadge: '🌱 Official Maharashtra Land Records AI',
      welcomeHeading: 'Hello! Ask questions about Maharashtra Land Records.',
      welcomeDesc:
        'This chatbot is grounded strictly in the official Maharashtra Land Records document (approx. 500 pages). You can ask about 7/12 extracts, 8A holdings, Ferfar mutation entries, potkharaba, or revenue laws in English or Marathi.',
      suggestedHeading: 'Suggested Questions:',
      suggestedQuestions: [
        {
          label: '👤 Who is recorded as the owner in the land records?',
          query: 'Who is recorded as the owner in the land records?',
        },
        {
          label: '📐 What is potkharaba and how is it calculated?',
          query: 'What is potkharaba and how is it calculated in land records?',
        },
        {
          label: '📜 How is heirship and mutation (Ferfar) recorded?',
          query: 'How is heirship and mutation (Ferfar) recorded in Maharashtra land records?',
        },
        {
          label: '💡 How do I read a digital 7/12 extract?',
          query: 'How do I read a digital 7/12 extract and what do its columns represent?',
        },
      ],
      placeholder: 'Ask a question about Maharashtra land records... (Enter to send)',
      sendTitle: 'Send (Enter)',
      disclaimer:
        'Answers are grounded in official Maharashtra Land Records. For legal certification, consult the official revenue portal.',
      thinking: 'Analyzing official land records...',
      docAnswer: '📄 Official Document Answer',
      generalAnswer: '🌐 General Assistant Answer',
      citedSourcesHeader: '📚 Cited Sources from Official Document:',
      pageLabel: 'Page',
      pageLabelPlural: 'Pages',
      errorHeading: 'Error:',
      retryBtn: '🔄 Retry',
      serviceUnavailable:
        'Official land records document is temporarily unavailable. Please contact the administrator.',
      apiKeyBannerText:
        'Developer API Key Required: Enter your dev API key in Settings to connect.',
      openSettings: 'Open Settings',
      settingsModalTitle: 'Developer Settings',
      saveSettings: 'Save Settings',
      settingsSaved: 'Settings saved successfully.',
      newChatToast: 'New conversation started.',
      invalidApiKeyToast: 'Invalid or missing API key. Please check Settings.',
      serverErrorToast: 'Could not connect to backend server. Ensure FastAPI is running.',
    },
    mr: {
      brandTitle: 'भूमी अभिलेख AI',
      brandSubtitle: 'महाराष्ट्र भूमी अभिलेख बॉट',
      activeDocName: 'अधिकृत भूमी अभिलेख',
      activeDocTitle: 'अधिकृत महाराष्ट्र भूमी अभिलेख (५००+ पाने)',
      newChat: 'नवीन चॅट',
      welcomeBadge: '🌱 अधिकृत महाराष्ट्र भूमी अभिलेख AI',
      welcomeHeading: 'नमस्कार! महाराष्ट्र भूमी अभिलेखांविषयी प्रश्न विचारा.',
      welcomeDesc:
        'हा बॉट सुमारे ५०० पानांच्या अधिकृत महाराष्ट्र भूमी अभिलेख दस्तऐवजावर आधारित आहे. आपण ७/१२ उतारा, ८अ खाते, फेरफार नोंद, पोटखराबा किंवा महसूल नियमांविषयी मराठी किंवा इंग्रजीत विचारू शकता.',
      suggestedHeading: 'उदाहरण प्रश्न (Suggested Questions):',
      suggestedQuestions: [
        {
          label: '👤 जमिनीचे खातेदार / मालक कोण आहेत?',
          query: 'या जमिनीचे खातेदार / मालक कोण आहेत?',
        },
        {
          label: '📐 पोटखराबा म्हणजे काय आणि त्याचे प्रकार कोणते?',
          query: 'पोटखराबा म्हणजे काय आणि त्याचे प्रकार व गणना कशी केली जाते?',
        },
        {
          label: '📜 वारस हक्क व फेरफार नोंदीचे नियम काय आहेत?',
          query: 'वारस हक्क व फेरफार नोंदीचे नियम काय आहेत?',
        },
        {
          label: '💡 ७/१२ उताऱ्यातील इतर हक्क रकाना काय दर्शवतो?',
          query: '७/१२ उताऱ्यातील इतर हक्क रकाना काय दर्शवतो आणि त्याचा अर्थ काय?',
        },
      ],
      placeholder: 'भूमी अभिलेखांविषयी प्रश्न विचारा... (पाठवण्यासाठी Enter दाबा)',
      sendTitle: 'पाठवा (Enter)',
      disclaimer:
        'अधिकृत भूमी अभिलेख पुराव्यानुसार उत्तरे दिली जातात. अंतिम पडताळणीसाठी महाभूमी संकेतस्थळाचा वापर करा.',
      thinking: 'अधिकृत भूमी अभिलेख तपासत आहे...',
      docAnswer: '📄 अधिकृत दस्तऐवज उत्तर (Official Document)',
      generalAnswer: '🌐 सामान्य सहाय्यक उत्तर (General Assistant)',
      citedSourcesHeader: '📚 अधिकृत दस्तऐवजातील संदर्भ स्रोत (Cited Sources):',
      pageLabel: 'पान',
      pageLabelPlural: 'पाने',
      errorHeading: 'त्रुटी:',
      retryBtn: '🔄 पुन्हा प्रयत्न करा',
      serviceUnavailable:
        'अधिकृत भूमी अभिलेख दस्तऐवज तात्पुरता अनुपलब्ध आहे. कृपया नंतर पुन्हा प्रयत्न करा.',
      apiKeyBannerText:
        'स्थानिक विकास API Key आवश्यक: विनंत्या पाठवण्यासाठी सेटिंग्जमध्ये आपली डेव्हलपमेंट API Key टाका.',
      openSettings: 'सेटिंग्ज उघडा',
      settingsModalTitle: 'डेव्हलपर सेटिंग्ज',
      saveSettings: 'जतन करा',
      settingsSaved: 'सेटिंग्ज यशस्वीरित्या जतन केल्या.',
      newChatToast: 'नवीन संभाषण सुरू केले.',
      invalidApiKeyToast: 'API Key अवैध किंवा गहाळ आहे. कृपया सेटिंग्ज तपासा.',
      serverErrorToast: 'सर्व्हरशी संपर्क होऊ शकला नाही. कृपया FastAPI सर्व्हर सुरू असल्याची खात्री करा.',
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // 2. Application State & Storage
  // ───────────────────────────────────────────────────────────────────────────

  const urlParams = new URLSearchParams(window.location.search);

  // Auto-detect default backend base URL
  const defaultApiUrl =
    window.location.protocol.startsWith('http') &&
    window.location.port !== '3000' &&
    window.location.port !== '5500'
      ? window.location.origin
      : 'http://127.0.0.1:8000';

  // Language: English by default for first-time visitors; persisted in localStorage
  const savedLang = localStorage.getItem('rag_language');
  const initialLang = savedLang === 'mr' || savedLang === 'en' ? savedLang : 'en';

  const state = {
    // Current interface language: 'en' or 'mr'
    language: initialLang,

    // Tenant user identifier (read from URL ?user_id= or sessionStorage or generated)
    userId:
      urlParams.get('user_id') ||
      sessionStorage.getItem('rag_user_id') ||
      'u_' + Math.random().toString(36).substring(2, 10),

    // Backend API key (read from URL ?api_key= or sessionStorage, never committed)
    apiKey:
      urlParams.get('api_key') ||
      urlParams.get('key') ||
      sessionStorage.getItem('rag_api_key') ||
      '',

    // Backend server URL
    apiBaseUrl:
      urlParams.get('api_url') ||
      sessionStorage.getItem('rag_api_url') ||
      defaultApiUrl,

    // Multi-turn conversation history
    history: [],

    // Request in progress flag
    isProcessing: false,

    // Backward-compatibility references for test fixtures
    activeDoc: { doc_id: 'official', filename: 'official_land_records.pdf', pages: 500 },
    documents: [],
  };

  // Persist session identifiers
  sessionStorage.setItem('rag_user_id', state.userId);
  if (state.apiKey) {
    sessionStorage.setItem('rag_api_key', state.apiKey);
  }
  sessionStorage.setItem('rag_api_url', state.apiBaseUrl);

  // ───────────────────────────────────────────────────────────────────────────
  // 3. DOM Elements
  // ───────────────────────────────────────────────────────────────────────────

  const dom = {
    // Header & Branding
    brandTitle: document.getElementById('brand-title'),
    brandSubtitle: document.getElementById('brand-subtitle'),
    activeDocPill: document.getElementById('active-doc-pill'),
    activeDocName: document.getElementById('active-doc-name'),
    langEnBtn: document.getElementById('lang-en-btn'),
    langMrBtn: document.getElementById('lang-mr-btn'),
    newChatBtn: document.getElementById('new-chat-btn'),
    newChatLabel: document.getElementById('new-chat-label'),
    settingsBtn: document.getElementById('settings-btn'),

    // Chat Area & Welcome Card
    chatMessages: document.getElementById('chat-messages'),
    welcomeScreen: document.getElementById('welcome-screen'),
    welcomeBadge: document.getElementById('welcome-badge'),
    welcomeHeading: document.getElementById('welcome-heading'),
    welcomeDesc: document.getElementById('welcome-desc'),
    suggestedHeading: document.getElementById('suggested-heading'),
    queryChips: document.getElementById('query-chips'),
    apiKeyBanner: document.getElementById('api-key-banner'),
    bannerMsg: document.getElementById('banner-msg'),
    bannerSettingsBtn: document.getElementById('banner-settings-btn'),

    // Input Bar
    chatForm: document.getElementById('chat-form'),
    chatTextarea: document.getElementById('chat-textarea'),
    sendBtn: document.getElementById('send-btn'),
    inputDisclaimerText: document.getElementById('input-disclaimer-text'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    settingsModalTitle: document.getElementById('settings-modal-title'),
    closeSettingsModalBtn: document.getElementById('close-settings-modal-btn'),
    settingApiKeyInput: document.getElementById('setting-api-key'),
    settingUserIdInput: document.getElementById('setting-user-id'),
    settingApiUrlInput: document.getElementById('setting-api-url'),
    toggleKeyVisibilityBtn: document.getElementById('toggle-key-visibility-btn'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),

    // Toast Container
    toastContainer: document.getElementById('toast-container'),

    // Hidden test hooks for backward-compatibility
    pdfFileInput: document.getElementById('pdf-file-input'),
    sidebar: document.getElementById('sidebar'),
  };

  // ───────────────────────────────────────────────────────────────────────────
  // 4. API Communication Helper
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Unified API fetch wrapper with authentication header and status handling.
   *
   * @param {string} endpoint - API path (e.g. '/api/v1/chat')
   * @param {RequestInit} [options={}] - Standard fetch options
   * @returns {Promise<any>} Parsed JSON response
   */
  async function apiRequest(endpoint, options = {}) {
    const url = state.apiBaseUrl.replace(/\/+$/, '') + endpoint;
    const headers = new Headers(options.headers || {});

    // Attach X-API-Key header if configured
    if (state.apiKey) {
      headers.set('X-API-Key', state.apiKey);
    }

    const config = {
      ...options,
      headers: headers,
    };

    let response;
    try {
      response = await fetch(url, config);
    } catch (networkErr) {
      console.error('Network error contacting backend:', networkErr);
      const dict = I18N[state.language] || I18N.en;
      throw new Error(dict.serverErrorToast);
    }

    // Handle authentication error
    if (response.status === 401) {
      const dict = I18N[state.language] || I18N.en;
      showToast(dict.invalidApiKeyToast, 'error');
      if (dom.apiKeyBanner) dom.apiKeyBanner.style.display = 'flex';
      throw new Error('401 Unauthorized: Invalid or missing API key.');
    }

    // Try parsing JSON error body
    let data;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = { detail: text };
    }

    if (!response.ok) {
      const detail = data && data.detail ? data.detail : `HTTP Error ${response.status}`;
      const err = new Error(detail);
      err.status = response.status;
      throw err;
    }

    return data;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 5. UI Helper Functions & Sanitization
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Escape raw untrusted string to prevent HTML/XSS injection.
   */
  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Format basic markdown features safely on escaped text.
   */
  function formatSafeMarkdown(rawText) {
    const escaped = escapeHtml(rawText);

    return escaped
      // Headers
      .replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; font-weight:700;">$1</h4>')
      .replace(/^## (.*$)/gim, '<h3 style="margin: 10px 0 4px; font-weight:700;">$1</h3>')
      .replace(/^# (.*$)/gim, '<h2 style="margin: 12px 0 6px; font-weight:700;">$1</h2>')
      // Bold & Italic
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Page citations e.g. (Page 1) or (पान १)
      .replace(/\((Page|पान)\s*(\d+)\)/gi, '<span class="source-page-badge">📄 $1 $2</span>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Bullet points
      .replace(/^\s*[\-\*]\s+(.*)$/gim, '<li>$1</li>')
      // Group consecutive list items
      .replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>')
      .replace(/<\/ul>\s*<ul>/gim, '')
      // Line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');
  }

  /**
   * Display a temporary floating toast message.
   */
  function showToast(message, type = 'info') {
    if (!dom.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  /**
   * Auto-scroll chat container to bottom when appropriate.
   */
  function scrollToBottom() {
    if (dom.chatMessages) {
      dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 6. Language Switching & Localization Engine
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Set and apply interface language ('en' or 'mr').
   */
  function setLanguage(lang) {
    if (lang !== 'en' && lang !== 'mr') return;
    state.language = lang;
    localStorage.setItem('rag_language', lang);

    // Update document HTML lang attribute
    document.documentElement.lang = lang === 'mr' ? 'mr-IN' : 'en';

    // Update switcher toggle buttons
    if (dom.langEnBtn && dom.langMrBtn) {
      const isEn = lang === 'en';
      dom.langEnBtn.classList.toggle('active', isEn);
      dom.langEnBtn.setAttribute('aria-pressed', isEn ? 'true' : 'false');
      dom.langMrBtn.classList.toggle('active', !isEn);
      dom.langMrBtn.setAttribute('aria-pressed', !isEn ? 'true' : 'false');
    }

    applyLanguageStrings(lang);
  }

  /**
   * Re-render all static and dynamic UI labels to the target language.
   */
  function applyLanguageStrings(lang) {
    const dict = I18N[lang] || I18N.en;

    // Header strings
    if (dom.brandTitle) dom.brandTitle.textContent = dict.brandTitle;
    if (dom.brandSubtitle) dom.brandSubtitle.textContent = dict.brandSubtitle;
    if (dom.activeDocName) dom.activeDocName.textContent = dict.activeDocName;
    if (dom.activeDocPill) dom.activeDocPill.setAttribute('title', dict.activeDocTitle);
    if (dom.newChatLabel) dom.newChatLabel.textContent = dict.newChat;

    // Welcome Card strings
    if (dom.welcomeBadge) dom.welcomeBadge.textContent = dict.welcomeBadge;
    if (dom.welcomeHeading) dom.welcomeHeading.textContent = dict.welcomeHeading;
    if (dom.welcomeDesc) dom.welcomeDesc.textContent = dict.welcomeDesc;
    if (dom.suggestedHeading) dom.suggestedHeading.textContent = dict.suggestedHeading;

    // Render suggested question chips
    renderSuggestedQuestions(dict.suggestedQuestions);

    // Composer strings
    if (dom.chatTextarea) {
      dom.chatTextarea.setAttribute('placeholder', dict.placeholder);
    }
    if (dom.sendBtn) {
      dom.sendBtn.setAttribute('title', dict.sendTitle);
    }
    if (dom.inputDisclaimerText) {
      dom.inputDisclaimerText.textContent = dict.disclaimer;
    }

    // Modal & Banner strings
    if (dom.settingsModalTitle) dom.settingsModalTitle.textContent = dict.settingsModalTitle;
    if (dom.saveSettingsBtn) dom.saveSettingsBtn.textContent = dict.saveSettings;
    if (dom.bannerMsg) dom.bannerMsg.innerHTML = `<strong>${dict.apiKeyBannerText}</strong>`;
    if (dom.bannerSettingsBtn) dom.bannerSettingsBtn.textContent = dict.openSettings;
  }

  /**
   * Render suggested questions chips in the welcome card.
   */
  function renderSuggestedQuestions(questions) {
    if (!dom.queryChips) return;
    dom.queryChips.innerHTML = '';

    questions.forEach((q) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'query-chip';
      chip.textContent = q.label;
      chip.setAttribute('data-query', q.query);
      chip.addEventListener('click', () => {
        submitChatQuestion(q.query);
      });
      dom.queryChips.appendChild(chip);
    });
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 7. Chat Message Handling & RAG Pipeline Interaction
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Send a chat question to POST /api/v1/chat.
   */
  async function submitChatQuestion(questionText) {
    const question = (questionText || dom.chatTextarea.value).trim();
    if (!question || state.isProcessing) return;

    const dict = I18N[state.language] || I18N.en;

    if (!state.apiKey) {
      dom.settingsModal.style.display = 'flex';
      showToast(dict.apiKeyBannerText, 'error');
      return;
    }

    // Clear input & reset height
    dom.chatTextarea.value = '';
    dom.chatTextarea.style.height = 'auto';
    dom.sendBtn.disabled = true;

    // Hide welcome card on first message
    if (dom.welcomeScreen) {
      dom.welcomeScreen.style.display = 'none';
    }

    // Append user message bubble to UI
    appendUserMessage(question);

    // Show typing loading indicator
    const typingIndicatorEl = appendTypingIndicator(dict.thinking);
    state.isProcessing = true;

    // Build payload according to ChatRequest schema
    // In fixed-document mode, the backend automatically binds to the active official PDF
    const payload = {
      user_id: state.userId,
      question: question,
      language: state.language,
      history: state.history.slice(-6),
    };

    try {
      const response = await apiRequest('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      // Remove typing indicator
      typingIndicatorEl.remove();

      // Append assistant answer
      appendAssistantMessage(response);

      // Record in conversation history
      state.history.push({ role: 'user', content: question });
      state.history.push({ role: 'assistant', content: response.answer });
    } catch (err) {
      typingIndicatorEl.remove();

      // Check for 503 Service Unavailable (official document missing or not indexed)
      let displayError = err.message;
      if (err.status === 503 || (err.message && err.message.includes('503'))) {
        displayError = dict.serviceUnavailable;
      }

      appendErrorMessage(displayError, question);
    } finally {
      state.isProcessing = false;
      dom.sendBtn.disabled = dom.chatTextarea.value.trim().length === 0;
      scrollToBottom();
    }
  }

  /**
   * Append a user bubble to the chat container.
   */
  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';

    const locale = state.language === 'mr' ? 'mr-IN' : 'en-US';
    const now = new Date().toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });

    row.innerHTML = `
      <div class="message-bubble">
        <div class="message-content">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
        <div class="message-time">${now}</div>
      </div>
      <div class="avatar" title="User">👤</div>
    `;

    dom.chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Append an assistant answer bubble to the chat container.
   */
  function appendAssistantMessage(res) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const dict = I18N[state.language] || I18N.en;
    const locale = state.language === 'mr' ? 'mr-IN' : 'en-US';
    const now = new Date().toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });
    const isDocMode = res.mode === 'document';

    // Mode badge
    const modeBadgeHtml = isDocMode
      ? `<div class="mode-badge document">${dict.docAnswer}</div>`
      : `<div class="mode-badge general">${dict.generalAnswer}</div>`;

    // Render concise page-only citations in document mode: "Page 17" / "Pages 17, 18, 19"
    let sourcesHtml = '';
    if (isDocMode && Array.isArray(res.sources) && res.sources.length > 0) {
      const pages = [...new Set(res.sources.map((s) => s.page))].sort((a, b) => a - b);
      const pagesLabel =
        pages.length === 1
          ? `${dict.pageLabel} ${pages[0]}`
          : `${dict.pageLabelPlural} ${pages.join(', ')}`;

      sourcesHtml = `
        <div class="sources-section">
          <div class="sources-header">
            <span>${dict.citedSourcesHeader}</span>
          </div>
          <div class="sources-chips">
            <span class="source-page-badge">📄 ${pagesLabel}</span>
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="avatar" title="${dict.brandTitle}">🌱</div>
      <div class="message-bubble">
        ${modeBadgeHtml}
        <div class="message-content"><p>${formatSafeMarkdown(res.answer)}</p></div>
        ${sourcesHtml}
        <div class="message-time">${now}</div>
      </div>
    `;

    dom.chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Append an error notification bubble with a retry button.
   */
  function appendErrorMessage(errorMessage, retryQuestion) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    const dict = I18N[state.language] || I18N.en;

    row.innerHTML = `
      <div class="avatar" style="background: var(--danger-light); color: var(--danger);">⚠️</div>
      <div class="message-bubble" style="border-color: var(--danger-border); background: var(--danger-light);">
        <div class="message-content" style="color: #991b1b;">
          <strong>${dict.errorHeading}</strong> ${escapeHtml(errorMessage)}
        </div>
        <button type="button" class="secondary-btn retry-btn" style="margin-top: 8px; font-size: 12px; padding: 4px 10px;">
          ${dict.retryBtn}
        </button>
      </div>
    `;

    const retryBtn = row.querySelector('.retry-btn');
    retryBtn.addEventListener('click', () => {
      row.remove();
      submitChatQuestion(retryQuestion);
    });

    dom.chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Append animated typing indicator while waiting for LLM response.
   */
  function appendTypingIndicator(label) {
    const row = document.createElement('div');
    row.className = 'message-row assistant typing-row';
    row.innerHTML = `
      <div class="avatar">🌱</div>
      <div class="message-bubble">
        <div class="typing-indicator" aria-label="${escapeHtml(label || 'Thinking...')}">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    dom.chatMessages.appendChild(row);
    scrollToBottom();
    return row;
  }

  /**
   * Reset conversation (New Chat).
   */
  function resetConversation() {
    state.history = [];
    dom.chatMessages.innerHTML = '';
    if (dom.welcomeScreen) {
      dom.chatMessages.appendChild(dom.welcomeScreen);
      dom.welcomeScreen.style.display = 'block';
    }
    const dict = I18N[state.language] || I18N.en;
    showToast(dict.newChatToast, 'info');
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 8. Administrative / Test Backward-Compatibility Stubs
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Stub maintained for automated test suites that check handleFileUpload.
   */
  async function handleFileUpload(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', state.userId);
    return await apiRequest('/api/v1/upload', {
      method: 'POST',
      body: formData,
    });
  }

  /**
   * Stub maintained for automated test suites that check selectDocument.
   */
  function selectDocument(doc) {
    state.activeDoc = doc;
    sessionStorage.setItem('rag_active_doc', JSON.stringify(doc));
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 9. Settings Modal & Event Listeners
  // ───────────────────────────────────────────────────────────────────────────

  function openSettingsModal() {
    dom.settingApiKeyInput.value = state.apiKey;
    dom.settingUserIdInput.value = state.userId;
    dom.settingApiUrlInput.value = state.apiBaseUrl;
    dom.settingsModal.style.display = 'flex';
  }

  function saveSettings() {
    const newKey = dom.settingApiKeyInput.value.trim();
    const newUserId = dom.settingUserIdInput.value.trim();
    const newUrl = dom.settingApiUrlInput.value.trim();

    if (!newUserId) {
      showToast('User ID cannot be empty.', 'error');
      return;
    }

    state.apiKey = newKey;
    state.userId = newUserId;
    if (newUrl) {
      state.apiBaseUrl = newUrl;
    }

    sessionStorage.setItem('rag_api_key', state.apiKey);
    sessionStorage.setItem('rag_user_id', state.userId);
    sessionStorage.setItem('rag_api_url', state.apiBaseUrl);

    dom.settingsModal.style.display = 'none';

    if (state.apiKey && dom.apiKeyBanner) {
      dom.apiKeyBanner.style.display = 'none';
    }

    const dict = I18N[state.language] || I18N.en;
    showToast(dict.settingsSaved, 'success');
  }

  function setupEventListeners() {
    // Language Switcher buttons
    if (dom.langEnBtn) {
      dom.langEnBtn.addEventListener('click', () => setLanguage('en'));
    }
    if (dom.langMrBtn) {
      dom.langMrBtn.addEventListener('click', () => setLanguage('mr'));
    }

    // Chat form submit
    dom.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      submitChatQuestion();
    });

    // Textarea Enter key & auto-resize
    dom.chatTextarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitChatQuestion();
      }
    });

    dom.chatTextarea.addEventListener('input', () => {
      dom.chatTextarea.style.height = 'auto';
      dom.chatTextarea.style.height = `${Math.min(dom.chatTextarea.scrollHeight, 140)}px`;
      dom.sendBtn.disabled = dom.chatTextarea.value.trim().length === 0 || state.isProcessing;
    });

    // New Chat
    dom.newChatBtn.addEventListener('click', resetConversation);

    // Settings Modal
    dom.settingsBtn.addEventListener('click', openSettingsModal);
    if (dom.bannerSettingsBtn) {
      dom.bannerSettingsBtn.addEventListener('click', openSettingsModal);
    }
    dom.closeSettingsModalBtn.addEventListener('click', () => {
      dom.settingsModal.style.display = 'none';
    });

    dom.toggleKeyVisibilityBtn.addEventListener('click', () => {
      const isPassword = dom.settingApiKeyInput.type === 'password';
      dom.settingApiKeyInput.type = isPassword ? 'text' : 'password';
      dom.toggleKeyVisibilityBtn.textContent = isPassword ? '🔒' : '👁️';
    });

    dom.saveSettingsBtn.addEventListener('click', saveSettings);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 10. Application Initialization
  // ───────────────────────────────────────────────────────────────────────────

  function init() {
    // Apply initial language (English default or saved localStorage)
    setLanguage(state.language);

    setupEventListeners();

    // Check API Key status for local developer banner
    if (!state.apiKey && dom.apiKeyBanner) {
      dom.apiKeyBanner.style.display = 'flex';
    } else if (dom.apiKeyBanner) {
      dom.apiKeyBanner.style.display = 'none';
    }
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose helpers for testing if running in test environment
  window.__ragApp = {
    state,
    setLanguage,
    submitChatQuestion,
    resetConversation,
    handleFileUpload,
    selectDocument,
    apiRequest,
  };
})();
