/**
 * miniKB Frontend — Vue 3 Single Page Application
 *
 * Features:
 * - Knowledge base management (CRUD)
 * - Document upload (drag & drop + click)
 * - RAG chat with SSE streaming
 * - Source citations display
 * - Model config viewer
 * - Dashboard stats
 */

const { createApp, ref, reactive, nextTick, onMounted } = Vue;

createApp({
  setup() {
    // ── State ──
    const sidebarCollapsed = ref(false);
    const knowledgeBases = ref([]);
    const currentKB = ref(null);
    const documents = ref([]);
    const chatSessions = ref([]);
    const currentSession = ref(null);
    const messages = ref([]);
    const inputText = ref('');
    const streaming = ref(false);
    const streamBuffer = ref('');
    const streamSources = ref([]);
    const stats = ref({});
    const modelConfig = ref({});
    const dragActive = ref(false);

    // Modals
    const showNewKB = ref(false);
    const showSettings = ref(false);
    const sourcePreview = ref(null);
    const newKBName = ref('');
    const newKBDesc = ref('');

    const messagesContainer = ref(null);

    // ── API helper ──
    async function api(url, options = {}) {
      const res = await fetch(url, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...options.headers },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json();
    }

    // ── Load functions ──
    async function loadKBs() {
      knowledgeBases.value = await api('/api/kb');
    }

    async function loadStats() {
      stats.value = await api('/api/system/stats');
    }

    async function loadConfig() {
      modelConfig.value = await api('/api/system/config');
    }

    async function loadDocuments() {
      if (!currentKB.value) return;
      documents.value = await api(`/api/kb/${currentKB.value.id}/docs`);
    }

    async function loadSessions() {
      if (!currentKB.value) return;
      chatSessions.value = await api(`/api/chat?kb_id=${currentKB.value.id}`);
    }

    async function loadMessages() {
      if (!currentSession.value) return;
      const data = await api(`/api/chat/${currentSession.value.id}`);
      messages.value = data.messages || [];
      await scrollToBottom();
    }

    // ── Actions ──
    async function selectKB(kb) {
      currentKB.value = kb;
      currentSession.value = null;
      messages.value = [];
      await Promise.all([loadDocuments(), loadSessions()]);
    }

    async function selectSession(s) {
      currentSession.value = s;
      await loadMessages();
    }

    async function createKB() {
      const kb = await api('/api/kb', {
        method: 'POST',
        body: JSON.stringify({ name: newKBName.value, description: newKBDesc.value }),
      });
      showNewKB.value = false;
      newKBName.value = '';
      newKBDesc.value = '';
      await loadKBs();
      await selectKB(kb);
      await loadStats();
    }

    async function newChat() {
      const s = await api('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ kb_id: currentKB.value.id }),
      });
      chatSessions.value.unshift(s);
      await selectSession(s);
    }

    async function handleFileSelect(e) {
      const files = Array.from(e.target.files);
      for (const f of files) await uploadFile(f);
      e.target.value = '';
    }

    async function handleDrop(e) {
      dragActive.value = false;
      const files = Array.from(e.dataTransfer.files);
      for (const f of files) await uploadFile(f);
    }

    async function uploadFile(file) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch(`/api/kb/${currentKB.value.id}/upload`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          alert(`上传失败: ${err.detail || res.statusText}`);
          return;
        }
        await loadDocuments();
        await loadStats();
      } catch (e) {
        alert(`上传失败: ${e.message}`);
      }
    }

    async function deleteDoc(doc) {
      if (!confirm(`确认删除文档「${doc.name}」？`)) return;
      await api(`/api/kb/${currentKB.value.id}/docs/${doc.id}`, { method: 'DELETE' });
      await loadDocuments();
      await loadStats();
    }

    async function sendMessage() {
      const text = inputText.value.trim();
      if (!text || streaming.value) return;

      // Optimistic UI: show user message immediately
      messages.value.push({
        id: Date.now(),
        role: 'user',
        content: text,
        sources: {},
      });
      inputText.value = '';
      streaming.value = true;
      streamBuffer.value = '';
      streamSources.value = [];
      await scrollToBottom();

      try {
        const res = await fetch(`/api/chat/${currentSession.value.id}/send`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.type === 'sources') {
              streamSources.value = data.content;
            } else if (data.type === 'token') {
              streamBuffer.value += data.content;
              await scrollToBottom();
            } else if (data.type === 'done') {
              messages.value.push({
                id: Date.now(),
                role: 'assistant',
                content: data.content,
                sources: { chunks: streamSources.value },
              });
              streamBuffer.value = '';
              streamSources.value = [];
              await loadSessions();
            } else if (data.type === 'error') {
              alert(`生成失败: ${data.content}`);
            }
          }
        }
      } catch (e) {
        alert(`请求失败: ${e.message}`);
      } finally {
        streaming.value = false;
      }
    }

    // ── Helpers ──
    async function scrollToBottom() {
      await nextTick();
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
      }
    }

    function docIcon(type) {
      const icons = { pdf: '📕', docx: '📘', txt: '📄', md: '📝', html: '🌐' };
      return icons[type] || '📄';
    }

    function formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1024 / 1024).toFixed(1) + ' MB';
    }

    function statusText(status) {
      return { ready: '已就绪', processing: '处理中', error: '失败' }[status] || status;
    }

    function renderMarkdown(text) {
      if (!text) return '';
      // Simple markdown rendering
      let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Code blocks
        .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Headings
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Line breaks
        .replace(/\n/g, '<br>');
      return html;
    }

    function showSource(src) {
      sourcePreview.value = src;
    }

    // ── Init ──
    onMounted(async () => {
      await Promise.all([loadKBs(), loadStats(), loadConfig()]);
    });

    return {
      sidebarCollapsed, knowledgeBases, currentKB, documents,
      chatSessions, currentSession, messages, inputText,
      streaming, streamBuffer, streamSources, stats, modelConfig,
      dragActive, showNewKB, showSettings, sourcePreview,
      newKBName, newKBDesc, messagesContainer,
      selectKB, selectSession, createKB, newChat,
      handleFileSelect, handleDrop, deleteDoc, sendMessage,
      docIcon, formatSize, statusText, renderMarkdown, showSource,
    };
  },
}).mount('#app');
