"""Tests for Phase 5 web chat frontend static delivery and structure."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_frontend_index_accessible():
    """Verify that GET /chat/ serves the web interface HTML successfully."""
    response = client.get("/chat/")
    assert response.status_code == 200
    html = response.text
    # Verify key structure and elements exist
    assert "Land Records AI" in html
    assert 'id="chat-messages"' in html
    assert 'id="chat-textarea"' in html
    assert 'id="send-btn"' in html
    assert 'id="pdf-file-input"' in html
    assert 'id="sidebar"' in html
    assert 'id="settings-modal"' in html
    assert 'id="active-doc-pill"' in html


def test_frontend_css_accessible():
    """Verify that GET /chat/style.css serves the stylesheet."""
    response = client.get("/chat/style.css")
    assert response.status_code == 200
    css = response.text
    assert "--primary" in css
    assert ".message-bubble" in css


def test_frontend_js_accessible():
    """Verify that GET /chat/app.js serves the frontend application script."""
    response = client.get("/chat/app.js")
    assert response.status_code == 200
    js = response.text
    assert "apiRequest" in js
    assert "handleFileUpload" in js
    assert "submitChatQuestion" in js
    assert "selectDocument" in js
    # Verify bilingual engine and localStorage key
    assert "rag_language" in js
    assert "setLanguage" in js
    assert "I18N" in js


def test_frontend_english_default_and_language_switcher():
    """Verify that the frontend HTML defaults to English and includes the language switcher."""
    response = client.get("/chat/")
    assert response.status_code == 200
    html = response.text

    # Default html lang is English
    assert '<html lang="en">' in html

    # Language switcher elements
    assert 'id="lang-en-btn"' in html
    assert 'id="lang-mr-btn"' in html
    assert 'English' in html
    assert 'मराठी' in html

    # Official document pill
    assert 'id="active-doc-pill"' in html
    assert 'Official Land Records' in html

    # Responsive viewport meta tag
    assert 'viewport-fit=cover' in html
    assert 'width=device-width' in html

    # Public chat input bar has no attachment/upload button
    assert 'id="attachment-btn"' not in html


def test_frontend_citations_are_page_labels_only():
    """Citation chips must show concise page labels (Page 17 / Pages 17, 18) and
    must never render long snippets, even though the API still returns them."""
    response = client.get("/chat/app.js")
    assert response.status_code == 200
    js = response.text

    # Page-only grouped citation rendering
    assert "source-page-badge" in js
    assert "pageLabelPlural" in js
    assert "pages.join(', ')" in js
    assert "new Set(res.sources.map((s) => s.page))" in js

    # No snippet rendering in citation labels
    assert "source-snippet" not in js
    assert "source-item" not in js
