const form = document.getElementById('ask-form');
const input = document.getElementById('question');
const submitBtn = document.getElementById('submit-btn');
const thread = document.getElementById('thread');
const turnTemplate = document.getElementById('tpl-turn');

const STAGE_SEQUENCE = ['retrieve', 'rerank', 'generate', 'verify'];
const STAGE_DELAYS_MS = [300, 700, 1400, 900]; // cosmetic pacing while the single request is in flight

document.getElementById('examples').addEventListener('click', (e) => {
  const btn = e.target.closest('.pill');
  if (!btn) return;
  input.value = btn.dataset.q;
  form.requestSubmit();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  submitBtn.disabled = true;
  const turn = renderTurn(question);
  thread.append(turn.el);
  turn.el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  input.value = '';

  const stageTimer = animateStages(turn);

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    const data = await res.json();
    stageTimer.finish();
    showAnswer(turn, data);
  } catch (err) {
    stageTimer.finish();
    showError(turn, err.message);
  } finally {
    submitBtn.disabled = false;
  }
});

function renderTurn(question) {
  const node = turnTemplate.content.cloneNode(true);
  const el = node.querySelector('.turn');
  el.querySelector('.turn-question').textContent = question;
  return {
    el,
    stages: [...el.querySelectorAll('.stage')],
    answerCard: el.querySelector('.answer-card'),
    answerText: el.querySelector('.answer-text'),
    citationsEl: el.querySelector('.citations'),
    badgeGrounded: el.querySelector('.badge-grounded'),
    badgeRefused: el.querySelector('.badge-refused'),
    retrievedBlock: el.querySelector('.retrieved-block'),
    chunkList: el.querySelector('.chunk-list'),
    chunkCount: el.querySelector('.chunk-count'),
    statusTrack: el.querySelector('.status-track'),
  };
}

function animateStages(turn) {
  let cancelled = false;
  let i = 0;
  const mark = () => {
    if (cancelled || i >= STAGE_SEQUENCE.length) return;
    turn.stages[i].classList.add('active');
    const delay = STAGE_DELAYS_MS[i];
    i += 1;
    setTimeout(() => {
      if (cancelled) return;
      if (i - 1 < turn.stages.length) turn.stages[i - 1].classList.remove('active');
      if (i - 1 < turn.stages.length) turn.stages[i - 1].classList.add('done');
      mark();
    }, delay);
  };
  mark();
  return {
    finish() {
      cancelled = true;
      turn.stages.forEach((s) => { s.classList.remove('active'); s.classList.add('done'); });
    },
  };
}

function showAnswer(turn, data) {
  turn.statusTrack.style.display = 'none';
  turn.answerCard.hidden = false;

  if (data.refused) {
    turn.badgeRefused.hidden = false;
  } else {
    turn.badgeGrounded.hidden = false;
  }

  typeReveal(turn.answerText, data.answer);

  data.citations.forEach((tag) => {
    const chip = document.createElement('button');
    chip.className = 'citation-chip';
    chip.type = 'button';
    chip.textContent = tag;
    chip.addEventListener('click', () => highlightChunk(turn, tag));
    turn.citationsEl.append(chip);
  });

  if (data.retrieved.length) {
    turn.retrievedBlock.hidden = false;
    turn.chunkCount.textContent = `(${data.retrieved.length})`;
    data.retrieved.forEach((c) => {
      const div = document.createElement('div');
      div.className = 'chunk';
      div.dataset.tag = `${c.source}#${c.section}`;
      div.innerHTML = `<div class="chunk-source"></div><div class="chunk-text"></div>`;
      div.querySelector('.chunk-source').textContent = `${c.source}#${c.section}`;
      div.querySelector('.chunk-text').textContent = c.text;
      turn.chunkList.append(div);
    });
  }
}

function highlightChunk(turn, tag) {
  const chunks = [...turn.chunkList.querySelectorAll('.chunk')];
  chunks.forEach((c) => c.classList.toggle('highlight', c.dataset.tag === tag));
  const target = chunks.find((c) => c.dataset.tag === tag);
  if (target) {
    turn.retrievedBlock.open = true;
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function showError(turn, message) {
  turn.statusTrack.style.display = 'none';
  turn.answerCard.hidden = false;
  turn.badgeRefused.hidden = false;
  turn.badgeRefused.textContent = 'Error';
  turn.answerText.textContent = `Request failed: ${message}. Is the server (and Ollama) running?`;
}

function typeReveal(el, text) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    el.textContent = text;
    return;
  }
  let i = 0;
  const step = Math.max(1, Math.round(text.length / 120));
  const id = setInterval(() => {
    i += step;
    el.textContent = text.slice(0, i);
    if (i >= text.length) clearInterval(id);
  }, 12);
}
