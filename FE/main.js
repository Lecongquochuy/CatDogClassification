const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
const modelSelect = document.getElementById('modelSelect');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  resultDiv.innerHTML = '';
});

uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  const model = modelSelect.value;
  if (!file) { alert('Vui lòng chọn ảnh trước!'); return; }

  const form = new FormData();
  form.append('image', file);
  form.append('model', model);

  resultDiv.innerHTML = '⏳ Đang dự đoán...';

  try {
    const res = await fetch('http://localhost:8001/predict', {
      method: 'POST',
      body: form
    });
    const data = await res.json();

    if (!res.ok) {
      resultDiv.innerHTML = 'Lỗi: ' + (data.error || res.statusText);
      return;
    }

    const probs = data.probabilities;
    let tableHTML = `
        <table class="prob-table">
          <thead>
            <tr><th>Class</th><th>Probability</th></tr>
          </thead>
          <tbody>
            ${Object.entries(probs)
              .map(([label, value]) => `<tr><td>${label}</td><td>${value.toFixed(4)}</td></tr>`)
              .join('')}
          </tbody>
        </table>
      `;
    resultDiv.innerHTML = `
      <div class="label">🏷️ <b>Label:</b> ${data.label}</div>
      ${tableHTML}
    `;

  } catch (e) {
    resultDiv.innerHTML = 'Lỗi mạng hoặc backend chưa chạy: ' + e.message;
  }
});
