const guidePage = document.querySelector('#guide-page');
const analysisPage = document.querySelector('#analysis-page');
const isGuide = window.location.pathname.replace(/\/$/, '') === '/model-guide';
if (isGuide) {
  guidePage.hidden = false;
  analysisPage.hidden = true;
  document.querySelectorAll('.nav-link').forEach((link) => link.classList.toggle('active', link.textContent.includes('Model guide')));
}

const labels = { COVID: 'COVID', Lung_Opacity: 'Lung Opacity', Normal: 'Normal', 'Viral Pneumonia': 'Viral Pneumonia' };
const fileInput = document.querySelector('#file-input');
let selectedFile = null;
let sampleSelected = false;
let analyzeAfterChoose = false;
const preview = document.querySelector('#preview');
const dropzone = document.querySelector('#dropzone');

function displayFile(file) {
  if (!file) return false;
  const extension = file.name.split('.').pop().toLowerCase();
  const extensionType = ({ jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png' })[extension];
  const mimeType = ['image/jpeg', 'image/png'].includes(file.type) ? file.type : extensionType;
  if (!['image/jpeg', 'image/png'].includes(mimeType)) {
    setStatus('Please choose a JPG or PNG image.', true);
    return false;
  }
  if (file.size > 10 * 1024 * 1024) {
    setStatus('The image must be 10 MB or smaller.', true);
    return false;
  }
  selectedFile = file.type === mimeType ? file : new File([file], file.name, { type: mimeType, lastModified: file.lastModified });
  sampleSelected = false;
  preview.src = URL.createObjectURL(selectedFile);
  preview.onload = () => setStatus('Ready');
  preview.onerror = () => setStatus('This image could not be previewed. Try another JPG or PNG.', true);
  document.querySelector('.image-view').classList.remove('empty-preview');
  preview.alt = `Preview of ${file.name}`;
  document.querySelector('#file-name').textContent = file.name;
  document.querySelector('#file-meta').textContent = `${(file.size / 1024).toFixed(0)} KB · Ready to analyze`;
  document.querySelector('#image-status').textContent = 'Image selected';
  document.querySelector('#clear-image').hidden = false;
  document.querySelector('#analyze-button').disabled = false;
  setStatus('Ready');
  clearResult();
  return true;
}

function setStatus(message, error = false) {
  const status = document.querySelector('#result-status');
  status.textContent = message;
  status.style.color = error ? '#ad4b40' : '';
}

function clearResult() {
  document.querySelector('#predicted-class').textContent = 'Awaiting analysis';
  document.querySelector('#confidence').textContent = '—';
  document.querySelector('#total-probability').textContent = 'Total —';
  document.querySelector('#probabilities').innerHTML = '<p class="empty-state">Choose an image and select Analyze image to see the model output.</p>';
  document.querySelector('#prediction-card').classList.remove('has-result');
}

function renderProbabilities(data) {
  const probabilityList = Object.entries(data.probabilities);
  document.querySelector('#predicted-class').textContent = labels[data.predicted_class] || data.predicted_class.replaceAll('_', ' ');
  document.querySelector('#confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
  const total = probabilityList.reduce((sum, [, value]) => sum + value, 0) * 100;
  document.querySelector('#total-probability').textContent = `Total ${total.toFixed(0)}%`;
  document.querySelector('#probabilities').innerHTML = probabilityList.map(([name, value]) => {
    const percent = value * 100;
    const predicted = name === data.predicted_class;
    return `<div class="probability-row${predicted ? ' is-predicted' : ''}"><div class="probability-label"><span>${labels[name] || name.replaceAll('_', ' ')}${predicted ? '　←' : ''}</span><span>${percent.toFixed(1)}%</span></div><div class="probability-track"><div class="probability-fill" style="width:${Math.min(100, percent)}%"></div></div></div>`;
  }).join('');
  document.querySelector('#result-status').textContent = sampleSelected ? 'Sample result' : 'Analysis complete';
}

if (fileInput) {
  document.querySelector('#browse-button').addEventListener('click', (event) => { event.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
      const valid = displayFile(file);
      if (analyzeAfterChoose && valid) {
        analyzeAfterChoose = false;
        document.querySelector('#analyze-button').click();
      } else if (analyzeAfterChoose) {
        analyzeAfterChoose = false;
      }
    }
    fileInput.value = '';
  });
  for (const eventName of ['dragenter', 'dragover']) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
  for (const eventName of ['dragleave', 'drop']) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); });
  dropzone.addEventListener('drop', (event) => displayFile(event.dataTransfer.files[0]));
  document.querySelector('#clear-image').addEventListener('click', () => {
    selectedFile = null; sampleSelected = false; fileInput.value = ''; preview.removeAttribute('src');
    document.querySelector('.image-view').classList.add('empty-preview');
    preview.alt = 'No image selected'; document.querySelector('#file-name').textContent = 'No image selected';
    document.querySelector('#file-meta').textContent = 'JPG or PNG · Up to 10 MB';
    document.querySelector('#image-status').textContent = 'No image';
    document.querySelector('#clear-image').hidden = true;
    clearResult();
  });
  fileInput.addEventListener('cancel', () => { analyzeAfterChoose = false; });
  document.querySelector('#analyze-button').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    if (!selectedFile) {
      setStatus('Choose a JPG or PNG image to analyze.');
      analyzeAfterChoose = true;
      fileInput.click();
      return;
    }
    button.disabled = true;
    button.textContent = 'Analyzing…';
    setStatus('Analyzing');
    try {
      const body = new FormData(); body.append('file', selectedFile);
      const response = await fetch('/predict', { method: 'POST', body });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || 'The image could not be analyzed.');
      if (!result.probabilities || !result.predicted_class) throw new Error('The model returned an incomplete response. Please try again.');
      renderProbabilities(result);
    } catch (error) {
      setStatus(error instanceof TypeError ? 'Could not reach the analysis API. Confirm the app is running, then retry.' : (error.message || 'Analysis failed.'), true);
    } finally {
      button.disabled = false;
      button.innerHTML = '<span>⛶</span> Analyze image';
    }
  });
}

const exampleValues = { COVID: 1.2, 'Lung Opacity': 4.8, Normal: 92.4, 'Viral Pneumonia': 1.6 };
const exampleBars = document.querySelector('.example-bars');
if (exampleBars) exampleBars.innerHTML = Object.entries(exampleValues).map(([name, value]) => `<div class="probability-row${name === 'Normal' ? ' is-predicted' : ''}"><div class="probability-label"><span>${name}${name === 'Normal' ? '　←' : ''}</span><span>${value.toFixed(1)}%</span></div><div class="probability-track"><div class="probability-fill" style="width:${value}%"></div></div></div>`).join('');
