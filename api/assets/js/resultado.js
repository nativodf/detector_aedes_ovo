async function buscarResultados() {
    const res = await fetch("http:/api");
    const data = await res.json();

    const div = document.getElementById("resultado");
    div.innerHTML = `<p>Total de ovos detectados: <strong>${data.total}</strong></p>`;

    data.results.forEach((item, index) => {
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + item.image;
        img.style.width = "600px";
        img.style.margin = "20px";

        const caption = document.createElement("p");
        caption.textContent = `Parte ${index + 1}: ${item.count} ovos`;

        div.appendChild(img);
        div.appendChild(caption);
    });
}

function showImg() {
    document.getElementById('inputImg').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview');
                img.src = e.target.result;
                img.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
}

async function uploadImg() {
    const fileInput = document.getElementById("inputImg");
    const file = fileInput.files[0];

    if (!file) {
        alert("Selecione uma imagem primeiro");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (res.ok) {
            alert("Imagem enviada com sucesso!");
        } else {
            alert("Erro ao enviar imagem");
        }
    } catch (err) {
        console.error(err);
        alert("Erro de conexão com o servidor");
    }
}

document.addEventListener('DOMContentLoaded', showImg);
