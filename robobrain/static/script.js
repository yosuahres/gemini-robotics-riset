document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const thumbnailGallery = document.querySelector('.thumbnail-gallery');
    const canvas = document.getElementById('annotation-canvas');
    const ctx = canvas.getContext('2d');
    const undoBtn = document.getElementById('undo-btn');
    const clearBtn = document.getElementById('clear-btn');
    const saveBtn = document.getElementById('save-btn');
    const instructionInput = document.getElementById('instruction-input');

    let images = [];
    let activeImageIndex = -1;

    async function fetchImages() {
        const response = await fetch('/images');
        const imageNames = await response.json();
        images = await Promise.all(imageNames.map(async (name) => {
            const img = new Image();
            img.src = `/uploads/${name}`;
            await img.decode();
            const annotation = await fetch(`/get_annotation/${name}`).then(res => res.json());
            return { 
                element: img, 
                name, 
                trajectory: annotation.points || [], 
                instruction: annotation.instruction || '' 
            };
        }));
        renderThumbnails();
        if (images.length > 0) {
            setActiveImage(0);
        }
    }

    imageUpload.addEventListener('change', async (event) => {
        const files = event.target.files;
        const formData = new FormData();
        for (const file of files) {
            formData.append('files[]', file);
        }

        await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        fetchImages();
    });

    function renderThumbnails() {
        thumbnailGallery.innerHTML = '';
        images.forEach((image, index) => {
            const thumb = image.element.cloneNode();
            thumb.addEventListener('click', () => setActiveImage(index));
            if (index === activeImageIndex) {
                thumb.classList.add('active');
            }
            thumbnailGallery.appendChild(thumb);
        });
    }

    async function setActiveImage(index) {
        if (index >= 0 && index < images.length) {
            activeImageIndex = index;
            const img = images[activeImageIndex].element;
            canvas.width = img.width;
            canvas.height = img.height;
            const annotation = await fetch(`/get_annotation/${images[activeImageIndex].name}`).then(res => res.json());
            images[activeImageIndex].trajectory = annotation.points || [];
            images[activeImageIndex].instruction = annotation.instruction || '';
            instructionInput.value = images[activeImageIndex].instruction;
            redrawCanvas();
            renderThumbnails();
        }
    }

    function redrawCanvas() {
        if (activeImageIndex === -1) return;

        const img = images[activeImageIndex].element;
        ctx.drawImage(img, 0, 0);

        const trajectory = images[activeImageIndex].trajectory;
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';

        for (let i = 0; i < trajectory.length; i++) {
            const point = trajectory[i];
            ctx.beginPath();
            ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();

            if (i > 0) {
                const prevPoint = trajectory[i - 1];
                ctx.beginPath();
                ctx.moveTo(prevPoint[0], prevPoint[1]);
                ctx.lineTo(point[0], point[1]);
                ctx.stroke();
            }
        }
    }

    canvas.addEventListener('mousedown', (event) => {
        if (activeImageIndex === -1) return;

        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        images[activeImageIndex].trajectory.push([x, y]);
        redrawCanvas();
    });

    canvas.addEventListener('dblclick', () => {
        setActiveImage((activeImageIndex + 1) % images.length);
    });

    undoBtn.addEventListener('click', () => {
        if (activeImageIndex !== -1 && images[activeImageIndex].trajectory.length > 0) {
            images[activeImageIndex].trajectory.pop();
            redrawCanvas();
        }
    });

    clearBtn.addEventListener('click', () => {
        if (activeImageIndex !== -1) {
            images[activeImageIndex].trajectory = [];
            redrawCanvas();
        }
    });

    saveBtn.addEventListener('click', async () => {
        if (activeImageIndex !== -1) {
            const data = {
                filename: images[activeImageIndex].name,
                trajectory: images[activeImageIndex].trajectory,
                instruction: instructionInput.value
            };
            await fetch('/save_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            alert('Annotation saved!');
        }
    });

    fetchImages();
});
