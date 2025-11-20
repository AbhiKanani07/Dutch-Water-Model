(() => {
  const statusEl = document.getElementById("forecast-status");
  if (!statusEl) return;

  const sourceSelect = document.getElementById("source-select");
  const yearRange = document.getElementById("year-range");
  const yearValue = document.getElementById("year-value");
  const summaryEl = document.getElementById("summary");
  const chartArea = document.getElementById("chart-area");
  const tableContainer = document.getElementById("table-container");
  const resetBtn = document.getElementById("reset-filters");

  let forecastData = [];
  let allSources = [];
  let minYear = null;
  let maxYear = null;

  const setStatus = (msg, isError = false) => {
    statusEl.textContent = msg;
    statusEl.style.color = isError ? "#b00020" : "#444";
  };

  const parseCSV = (text) => {
    const lines = text.trim().split(/\r?\n/);
    const header = lines.shift().split(",");
    const idxYear = header.indexOf("year");
    const idxSource = header.indexOf("source");
    const idxPred = header.indexOf("predicted_mln_m3");
    if (idxYear === -1 || idxSource === -1 || idxPred === -1) {
      throw new Error("Missing expected columns in CSV");
    }
    return lines
      .map((line) => {
        const cols = line.split(",");
        return {
          year: Number(cols[idxYear]),
          source: cols[idxSource],
          predicted: Number(cols[idxPred]),
        };
      })
      .filter((d) => Number.isFinite(d.year) && Number.isFinite(d.predicted));
  };

  const render = () => {
    const maxY = Number(yearRange.value);
    const source = sourceSelect.value;

    const filtered = forecastData.filter(
      (d) => d.year <= maxY && (source === "all" || d.source === source)
    );

    if (!filtered.length) {
      summaryEl.textContent = "No forecast data for this filter.";
      chartArea.innerHTML = "";
      tableContainer.innerHTML = "";
      return;
    }

    const sorted = filtered.slice().sort((a, b) => {
      if (a.year !== b.year) return a.year - b.year;
      return a.source.localeCompare(b.source);
    });

    const maxVal = Math.max(...sorted.map((d) => d.predicted));
    const avgVal =
      sorted.reduce((acc, d) => acc + d.predicted, 0) / sorted.length;

    summaryEl.textContent = `Showing ${sorted.length} rows | Years ${sorted[0].year}–${sorted[sorted.length - 1].year} | Avg ${avgVal.toFixed(1)} mln m³`;

    chartArea.innerHTML = sorted
      .map((d) => {
        const pct = maxVal > 0 ? (d.predicted / maxVal) * 100 : 0;
        return `
          <div class="bar-row">
            <span>${d.year}</span>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct.toFixed(1)}%"></div>
            </div>
            <span>${d.predicted.toFixed(1)}</span>
          </div>
        `;
      })
      .join("");

    const rows = sorted
      .map(
        (d) =>
          `<tr><td>${d.year}</td><td>${d.source}</td><td>${d.predicted.toFixed(
            2
          )}</td></tr>`
      )
      .join("");

    tableContainer.innerHTML = `
      <table aria-label="Forecast table">
        <thead>
          <tr><th>Year</th><th>Source</th><th>Predicted (mln m³)</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  };

  const initControls = () => {
    allSources = Array.from(new Set(forecastData.map((d) => d.source))).sort();
    minYear = Math.min(...forecastData.map((d) => d.year));
    maxYear = Math.max(...forecastData.map((d) => d.year));

    sourceSelect.innerHTML =
      '<option value="all">All sources</option>' +
      allSources.map((s) => `<option value="${s}">${s}</option>`).join("");
    sourceSelect.disabled = false;

    yearRange.min = String(minYear);
    yearRange.max = String(maxYear);
    yearRange.value = String(maxYear);
    yearRange.disabled = false;
    yearValue.textContent = maxYear;

    resetBtn.disabled = false;

    sourceSelect.addEventListener("change", render);
    yearRange.addEventListener("input", () => {
      yearValue.textContent = yearRange.value;
      render();
    });
    resetBtn.addEventListener("click", () => {
      sourceSelect.value = "all";
      yearRange.value = String(maxYear);
      yearValue.textContent = maxYear;
      render();
    });

    render();
  };

  const loadData = async () => {
    try {
      const res = await fetch("plots/forecast_total_economy.csv", {
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(
          "File not found. Run python main.py to regenerate plots/forecast_total_economy.csv and push it."
        );
      }
      const text = await res.text();
      forecastData = parseCSV(text);
      if (!forecastData.length) {
        throw new Error("Forecast CSV is empty.");
      }
      setStatus("Forecast loaded. Adjust filters to explore.");
      initControls();
    } catch (err) {
      setStatus(err.message, true);
      sourceSelect.disabled = true;
      yearRange.disabled = true;
      resetBtn.disabled = true;
      summaryEl.textContent = "";
      chartArea.innerHTML = "";
      tableContainer.innerHTML = "";
    }
  };

  loadData();
})();
