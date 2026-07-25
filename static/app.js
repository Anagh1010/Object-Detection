const input = document.querySelector('#image');
const zone = document.querySelector('.drop-zone');
const preview = document.querySelector('#preview');

function showPreview(file) {
  if (!file || !file.type.startsWith('image/')) return;
  preview.src = URL.createObjectURL(file);
  preview.classList.add('visible');
}

input.addEventListener('change', () => showPreview(input.files[0]));
['dragenter', 'dragover'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', (event) => { input.files = event.dataTransfer.files; showPreview(input.files[0]); });
