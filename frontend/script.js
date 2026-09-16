// ==========================================
// MACHINE FAILURE PREDICTION FRONTEND
// ==========================================

const API_URL = "http://127.0.0.1:8001";


// ==========================================
// GET HTML ELEMENTS
// ==========================================

const predictButton = document.getElementById("predictButton");
const resultDiv = document.getElementById("result");
const apiStatus = document.getElementById("apiStatus");


// ==========================================
// CHECK BACKEND STATUS
// ==========================================

async function checkAPI() {

    try {

        const response = await fetch(
            `${API_URL}/health`
        );

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        const data = await response.json();

        if (data.status === "healthy") {

            apiStatus.textContent = "Online";

        } else {

            apiStatus.textContent = "Unavailable";
        }

    } catch (error) {

        apiStatus.textContent = "Offline";

    }
}


// ==========================================
// GET INPUT VALUE
// ==========================================

function getInputValue(id) {

    const value = document.getElementById(id).value.trim();

    if (value === "") {
        return NaN;
    }

    return Number(value);
}


// ==========================================
// VALIDATE INPUTS
// ==========================================

function validateInputs(values) {

    for (const value of Object.values(values)) {

        if (!Number.isFinite(value)) {

            return false;
        }
    }

    return true;
}


// ==========================================
// PREDICT MACHINE FAILURE
// ==========================================

async function predictFailure() {

    const values = {

        temperature:
            getInputValue("temperature"),

        pressure:
            getInputValue("pressure"),

        vibration:
            getInputValue("vibration"),

        rotational_speed:
            getInputValue("rotational_speed"),

        operating_hours:
            getInputValue("operating_hours"),

        ambient_temperature:
            getInputValue("ambient_temperature"),

        load_percent:
            getInputValue("load_percent")
    };


    // --------------------------------------
    // INPUT VALIDATION
    // --------------------------------------

    if (!validateInputs(values)) {

        resultDiv.innerHTML = `
            <div class="result-box failure">
                <div class="result-status">
                    Invalid Input
                </div>

                <p>
                    Please enter values for all
                    machine sensor fields.
                </p>
            </div>
        `;

        return;
    }


    // --------------------------------------
    // DISABLE BUTTON
    // --------------------------------------

    predictButton.disabled = true;

    predictButton.textContent =
        "Predicting...";


    resultDiv.innerHTML = `
        <div class="result-placeholder">
            Sending sensor data to the ML model...
        </div>
    `;


    try {

        // ----------------------------------
        // SEND DATA TO FASTAPI
        // ----------------------------------

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(values)
            }
        );


        // ----------------------------------
        // HANDLE API ERROR
        // ----------------------------------

        if (!response.ok) {

            const errorData =
                await response.json();

            throw new Error(
                errorData.detail ||
                "Prediction failed."
            );
        }


        // ----------------------------------
        // READ RESPONSE
        // ----------------------------------

        const data =
            await response.json();


        // ----------------------------------
        // DISPLAY RESULT
        // ----------------------------------

        const isFailure =
            data.prediction === 1;

        const probability =
            (data.failure_probability * 100)
            .toFixed(1);


        if (isFailure) {

            resultDiv.innerHTML = `

                <div class="result-box failure">

                    <div class="result-status">
                        🔴 FAILURE RISK
                    </div>

                    <div class="probability">
                        Failure Probability:
                        ${probability}%
                    </div>

                    <p>
                        The machine shows a
                        significant risk of failure.
                    </p>

                </div>
            `;

        } else {

            resultDiv.innerHTML = `

                <div class="result-box normal">

                    <div class="result-status">
                        🟢 NORMAL
                    </div>

                    <div class="probability">
                        Failure Probability:
                        ${probability}%
                    </div>

                    <p>
                        The machine is currently
                        predicted to be operating normally.
                    </p>

                </div>
            `;
        }


    } catch (error) {

        // ----------------------------------
        // DISPLAY ERROR
        // ----------------------------------

        resultDiv.innerHTML = `

            <div class="result-box failure">

                <div class="result-status">
                    ⚠️ Connection Error
                </div>

                <p>
                    ${error.message}
                </p>

                <p>
                    Make sure the FastAPI backend
                    is running.
                </p>

            </div>
        `;
    }


    // --------------------------------------
    // ENABLE BUTTON
    // --------------------------------------

    predictButton.disabled = false;

    predictButton.textContent =
        "Predict Machine Failure";
}


// ==========================================
// BUTTON EVENT
// ==========================================

predictButton.addEventListener(
    "click",
    predictFailure
);


// ==========================================
// CHECK API WHEN PAGE LOADS
// ==========================================

checkAPI();
