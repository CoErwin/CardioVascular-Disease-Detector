window.onload = async () => {

    // ── Get stored data ──
    const risk   = parseFloat(localStorage.getItem("risk") || 0);
    const result = JSON.parse(localStorage.getItem("result") || "{}");

    // ── Merge risk into result for API ──
    const payload = { ...result, risk: risk };

    try {
        // ── Call ML report API ──
        const response = await fetch("http://localhost:8000/report", {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify(payload)
        });

        const data = await response.json();
        renderReport(data);

    } catch (err) {
        console.error("Report API error:", err);

        // ── Fallback ──
        renderReport({
            risk        : risk,
            risk_level  : risk > 70 ? "High" : risk > 40 ? "Moderate" : "Low",
            advice      : "Could not load ML report. Please try again.",
            doctor      : "Consult a cardiologist if unsure.",
            precautions : ["Exercise daily", "Eat healthy", "Monitor BP"],
            top_factors : [],
            warnings    : [],
            insight     : "ML insight unavailable."
        });
    }
};

function renderReport(data) {

    // ── Top factors HTML ──
    const factorsHTML = data.top_factors.length > 0
        ? data.top_factors.map(f => `
            <li>
                <b>${f.feature}</b> — value: ${f.value}
                (${f.importance}% importance)
            </li>`).join("")
        : "<li>Feature data unavailable</li>";

    // ── Warnings HTML ──
    const warningsHTML = data.warnings.length > 0
        ? data.warnings.map(w => `<li>⚠️ ${w}</li>`).join("")
        : "<li>✅ No critical warnings detected</li>";

    // ── Precautions HTML ──
    const precautionsHTML = data.precautions
        .map(p => `<li>${p}</li>`).join("");

    document.getElementById("report").innerHTML = `
        <h2>${data.emoji || ""} Risk: ${data.risk}% — ${data.risk_level}</h2>

        <h3>📊 Summary</h3>
        <p>Your cardiovascular risk is <b>${data.risk}%</b>
           based on clinical parameters.</p>

        <h3>🧠 AI Insight</h3>
        <p>${data.insight}</p>

        <h3>📌 Your Top Risk Factors</h3>
        <ul>${factorsHTML}</ul>

        <h3>⚠️ Personal Warnings</h3>
        <ul>${warningsHTML}</ul>

        <h3>🛡️ Precautions</h3>
        <ul>${precautionsHTML}</ul>

        <h3>💊 Recommendation</h3>
        <p>${data.advice}</p>

        <h3>👨‍⚕️ Doctor Advice</h3>
        <p>${data.doctor}</p>
    `;
}