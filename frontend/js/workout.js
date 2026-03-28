const API_URL = "http://127.0.0.1:8000";

async function loadWorkout() {
    try {
        // ── Get risk and age from localStorage ──
        const risk     = localStorage.getItem("risk") || 50;
        const userData = JSON.parse(localStorage.getItem("userData") || "{}");
        const age      = userData.age || 35;

        const res  = await fetch(
            `${API_URL}/workout?risk=${risk}&age=${age}`
        );
        const data = await res.json();

        // ── Show fitness level banner ──
        if (data.fitness_level) {
            const banner = document.createElement("p");
            banner.style.textAlign  = "center";
            banner.style.fontWeight = "bold";
            banner.style.fontSize   = "1.1rem";
            banner.innerText =
                `🏋️ Your Fitness Level: ${data.fitness_level}`;
            document.querySelector(".container")
                    ?.prepend(banner);
        }

        render("cardio",      data.cardio);
        render("strength",    data.strength);
        render("flexibility", data.flexibility);

    } catch (err) {
        console.error(err);
        document.body.innerHTML += "<p>Error loading workout</p>";
    }
}

function render(id, items) {
    const container = document.getElementById(id);
    container.innerHTML = "";

    items.forEach(item => {
        container.innerHTML += `
            <div class="card">
                <h3>${item.name}</h3>
                ${item.level    ? `<p>Level: ${item.level}</p>`       : ""}
                ${item.sets     ? `<p>Sets: ${item.sets}</p>`         : ""}
                ${item.reps     ? `<p>Reps: ${item.reps}</p>`         : ""}
                ${item.duration ? `<p>Duration: ${item.duration}</p>` : ""}
            </div>
        `;
    });
}

window.onload = loadWorkout;