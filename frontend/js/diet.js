// ✅ FIXED diet.js
async function loadDiet() {
    try {
        // ── Get risk and preference from localStorage ──
        const risk     = localStorage.getItem("risk") || 50;
        const userData = JSON.parse(localStorage.getItem("userData") || "{}");
        const dietPref = userData.diet || "veg"; // ← reads from your form input

        const res  = await fetch(
            `http://127.0.0.1:8000/diet?risk=${risk}&diet_preference=${dietPref}`
        );
        const data = await res.json();

        function render(meal) {
            return meal.map(item => `
                <div class="card">
                    <h4>${item.food}</h4>
                    <p>Calories: ${item.calories}</p>
                    <p>Protein: ${item.protein}</p>
                    <p>Carbs: ${item.carbs}</p>
                    <p>Fat: ${item.fat}</p>
                </div>
            `).join("");
        }

        document.getElementById("breakfast").innerHTML = render(data.breakfast);
        document.getElementById("lunch").innerHTML     = render(data.lunch);
        document.getElementById("dinner").innerHTML    = render(data.dinner);

    } catch (err) {
        console.error(err);
        document.body.innerHTML += "<p>Error loading diet</p>";
    }
}

loadDiet();