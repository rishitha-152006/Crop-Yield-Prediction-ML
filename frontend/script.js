async function predict() {
    const rainfall = document.getElementById("rainfall").value;
    const temperature = document.getElementById("temperature").value;

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            rainfall: rainfall,
            temperature: temperature
        })
    });

    const data = await response.json();

    document.getElementById("result").innerText =
        "Predicted Yield: " + data.predicted_yield;
}