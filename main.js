// Poll the system endpoint every 1.5 seconds to pull the latest generation frame matrix
setInterval(async () => {
    try {
        const fetchTarget = await fetch('http://127.0.0.1:5000/api/joule_stream');
        if (fetchTarget.ok) {
            const parsedData = await fetchTarget.json();
            // Assuming your presentation target text block uses id='joule-display'
            document.getElementById('joule-display').innerText = parsedData.response;
        }
    } catch (networkError) {
        console.log("Awaiting connection sequence validation update...", networkError);
    }
}, 1500);