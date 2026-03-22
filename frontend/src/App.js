import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // We'll create this for styling

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [pipeline, setPipeline] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [topK, setTopK] = useState(5);

  const search = async () => {
    // Validation
    if (!query.trim()) {
      setError("Please enter a search query");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);
    setPipeline("");

    try {
      const res = await axios.get("http://localhost:8000/search", {
        params: { 
          query: query.trim(), 
          k: topK 
        }
      });

      setPipeline(res.data.pipeline_used || "Unknown");
      setResults(res.data.results || []);
      
      if (!res.data.results || res.data.results.length === 0) {
        setError("No results found. Try a different query.");
      }
    } catch (err) {
      console.error("Search error:", err);
      
      if (err.response) {
        // Backend returned an error
        setError(`Error: ${err.response.data.detail || err.response.statusText}`);
      } else if (err.request) {
        // Request made but no response
        setError("Cannot connect to backend. Make sure the API is running on http://localhost:8000");
      } else {
        // Something else happened
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      search();
    }
  };

  const clearSearch = () => {
    setQuery("");
    setResults([]);
    setPipeline("");
    setError("");
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>NCO Semantic Search</h1>
        <p className="subtitle">
          Search for occupations in English, Hindi, Tamil, Telugu, and other Indian languages
        </p>
      </header>

      {/* Search Section */}
      <div className="search-section">
        <div className="search-bar">
          <input
            type="text"
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter job title or description (e.g., 'software engineer' or 'सिलाई ऑपरेटर')"
            disabled={loading}
          />
          
          <div className="search-controls">
            <label className="topk-label">
              Results:
              <select 
                value={topK} 
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="topk-select"
                disabled={loading}
              >
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="15">15</option>
              </select>
            </label>

            <button 
              onClick={search} 
              className="search-button"
              disabled={loading || !query.trim()}
            >
              {loading ? "Searching..." : "Search"}
            </button>

            {query && (
              <button 
                onClick={clearSearch} 
                className="clear-button"
                disabled={loading}
              >
                Clear
              </button>
            )}
          </div>
        </div>


        {/* Error Message */}
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <div className="results-section">
          <h2 className="results-title">
            Found {results.length} {results.length === 1 ? 'result' : 'results'}
          </h2>
          
          <div className="results-grid">
            {results.map((result, index) => (
              <div key={index} className="result-card">
                <div className="result-header">
                  <span className="result-rank">#{index + 1}</span>
                  <span className="result-score">
                    Score: {result.score.toFixed(4)}
                  </span>
                </div>
                
                <h3 className="result-title">
                  {result.Title}
                </h3>
                
                <div className="result-code">
                  NCO Code: <span className="code-value">{result.NCO_Code}</span>
                </div>
                
                {result.Description && (
                  <p className="result-description">
                    {result.Description}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Searching...</p>
        </div>
      )}

      {/* Example Queries */}
      {!results.length && !loading && !error && (
        <div className="examples-section">
          <h3>Try these example queries:</h3>
          <div className="example-chips">
            <button onClick={() => setQuery("software engineer")} className="example-chip">
              software engineer
            </button>
            <button onClick={() => setQuery("truck driver")} className="example-chip">
              truck driver
            </button>
            <button onClick={() => setQuery("सिलाई ऑपरेटर")} className="example-chip">
              सिलाई ऑपरेटर
            </button>
            <button onClick={() => setQuery("ట్రక్ డ్రైవర్")} className="example-chip">
              ట్రక్ డ్రైవర్ (Telugu)
            </button>
            <button onClick={() => setQuery("security guard")} className="example-chip">
              security guard
            </button>
            <button onClick={() => setQuery("दरजी")} className="example-chip">
              दरजी (Tailor)
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Powered by SBERT + IndicBERT • NCO 2015 Classification</p>
      </footer>
    </div>
  );
}

export default App;