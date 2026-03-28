const API_URL = "http://127.0.0.1:8000";

let latestInput = null;
let latestRisk  = null;

function getValue(id) {
    const el = document.getElementById(id);
    return el ? el.value : null;
}

function getInputData() {
    return {
        age      : Number(getValue("age")),
        sex      : Number(getValue("sex")),
        cp       : Number(getValue("cp")),
        trestbps : Number(getValue("trestbps")),
        chol     : Number(getValue("chol")),
        fbs      : Number(getValue("fbs")),
        restecg  : Number(getValue("restecg")),
        thalach  : Number(getValue("thalach")),
        exang    : Number(getValue("exang")),
        oldpeak  : Number(getValue("oldpeak")),
        slope    : Number(getValue("slope")),
        ca       : Number(getValue("ca")),
        thal     : Number(getValue("thal")),
        weight   : Number(getValue("weight")),
        height   : Number(getValue("height")),
        diet     : getValue("diet")
    };
}

async function predictRisk() {
    const data  = getInputData();
    latestInput = data;

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify(data)
        });

        const result = await res.json();
        console.log("API RESPONSE:", result);

        let risk =
            result.risk ??
            result.prediction ??
            result.risk_percentage ??
            0;

        risk       = Number(risk).toFixed(2);
        latestRisk = risk;

        document.getElementById("result").innerText = `Your Risk: ${risk}%`;

    } catch (err) {
        console.error(err);
        alert("Error predicting risk");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("predictBtn")
        ?.addEventListener("click", predictRisk);

    document.getElementById("dietBtn")
        ?.addEventListener("click", () => {
            localStorage.setItem("userData", JSON.stringify(latestInput));
            localStorage.setItem("risk",     latestRisk);              // ✅ ADD
            window.location.href = "diet.html";
        });

    document.getElementById("workoutBtn")
        ?.addEventListener("click", () => {
            localStorage.setItem("userData", JSON.stringify(latestInput));
            localStorage.setItem("risk",     latestRisk);              // ✅ ADD
            window.location.href = "workout.html";
        });

    document.getElementById("reportBtn")
        ?.addEventListener("click", () => {
            localStorage.setItem("userData", JSON.stringify(latestInput));
            localStorage.setItem("result",   JSON.stringify(latestInput));
            localStorage.setItem("risk",     latestRisk);
            window.location.href = "report.html";
        });
});