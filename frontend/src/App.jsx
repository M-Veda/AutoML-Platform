import { useState } from "react";
import axios from "axios";
import logo from "./assets/logo.png";
import toast, { Toaster } from "react-hot-toast";

import {
  BarChart,
  Bar,
 XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from "recharts";

export default function App() {

  const [file, setFile] = useState(null);

  const [message, setMessage] = useState("");

  const [targetColumn, setTargetColumn] = useState("");

  const [trainingResult, setTrainingResult] = useState(null);

  const [featureInputs, setFeatureInputs] = useState({});

  const [prediction, setPrediction] = useState(null);

  const [loading, setLoading] = useState(false);

  const [progress, setProgress] = useState(0);

  const [darkMode, setDarkMode] = useState(true);

  // =========================
  // HANDLE FILE UPLOAD
  // =========================

  const handleUpload = async () => {

    if (!file) {

      toast.error("Please select a file");

      return;
    }

    const formData = new FormData();

    formData.append("file", file);

    try {

      const response = await axios.post(

        "http://localhost:8000/upload-dataset",

        formData
      );

      setMessage(response.data.message);
      toast.success("Dataset Uploaded Successfully");

    } catch (error) {

      console.log(error);

      toast.error("Upload Failed");
    }
  };

  // =========================
  // HANDLE MODEL TRAINING
  // =========================

  const handleTraining = async () => {    
    if (!file || !targetColumn) {

      toast.error("Please upload dataset and enter target column");

      return;
    }

    try {

      setLoading(true);

      setProgress(0);

const progressInterval = setInterval(() => {

  setProgress((prev) => {

    if (prev >= 90) {

      return prev;
    }

    return prev + 10;
  });

}, 500);

      const response = await axios.post(

        "http://localhost:8000/train-models",

        {

          filename: file.name,

          target_column: targetColumn
        }
      );

      setTrainingResult(response.data);
      toast.success("Models Trained Successfully");
      setProgress(100);

      clearInterval(progressInterval);
      setLoading(false);

    } catch (error) {

      console.log(error);

      toast.error("Training Failed");
      clearInterval(progressInterval);
      setLoading(false);
    }

    
  };

// =========================
// HANDLE INPUT CHANGE
// =========================

const handleInputChange = (feature, value) => {

  setFeatureInputs({

    ...featureInputs,

    [feature]: value
  });
};

// =========================
// HANDLE PREDICTION
// =========================

const handlePrediction = async () => {

  try {

    const response = await axios.post(

      "http://localhost:8000/predict",

      {
        features: featureInputs
      }
    );

    setPrediction(response.data.prediction);
    toast.success("Prediction Generated");


  } catch (error) {

    console.log(error);

    toast.error("Prediction Failed");
  }
};

  return (

<div 
  className={`min-h-screen overflow-hidden transition-all duration-500 ${
    darkMode
      ? "bg-slate-950 text-white"
      : "bg-slate-100 text-black"
  }`}
>
<Toaster position="top-right" />

      {/* Background Glow */}

      <div className="absolute top-0 left-0 w-96 h-96 bg-cyan-500 opacity-20 blur-3xl rounded-full"></div>

      <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500 opacity-20 blur-3xl rounded-full"></div>

      {/* Main Content */}

<div className="absolute top-6 right-6 z-50">

  <button
    onClick={() => setDarkMode(!darkMode)}
    className="bg-white/10 border border-white/20 backdrop-blur-lg px-5 py-3 rounded-2xl font-semibold hover:scale-105 transition-all duration-300"
  >

    {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}

  </button>

</div>

      <div className="relative z-10 flex flex-col items-center py-12 px-6">

        {/* Hero */}

        <div className="flex flex-col items-center">

  <img
    src={logo}
    alt="AutoML Logo"
    className="w-40 mb-6 drop-shadow-2xl"
  />

  <h1 className="hero-title">
    AutoML Platform
  </h1>

</div>

        <p className="text-slate-400 text-lg md:text-xl mb-12 text-center max-w-2xl">

          Upload any dataset, train multiple machine learning algorithms
          automatically, compare performances, and get intelligent predictions.
        </p>

        {/* Upload Card */}

        <div className="backdrop-blur-lg bg-white/10 border border-white/20 rounded-3xl p-10 w-full max-w-2xl shadow-2xl">

          <h2 className="text-3xl font-bold mb-6 text-center">

            Upload Dataset
          </h2>

          {/* File Input */}

          <label className="w-full flex flex-col items-center justify-center border-2 border-dashed border-cyan-400/40 rounded-2xl p-10 cursor-pointer hover:border-cyan-400 hover:bg-cyan-500/5 transition-all duration-300 mb-6">

  <div className="text-5xl mb-4">
    📂
  </div>

  <h3 className="text-xl font-semibold mb-2">

    Drag & Drop Dataset
  </h3>

  <p className="text-slate-400 text-sm mb-4">

    Upload CSV datasets for AI training
  </p>

  {
    file && (
      <div className="text-green-400 font-semibold">

        Selected: {file.name}
      </div>
    )
  }

  <input
    type="file"
    accept=".csv"
    className="hidden"
    onChange={(e) => setFile(e.target.files[0])}
  />

</label>

          {/* Upload Button */}

          <button
            onClick={handleUpload}
            className="w-full bg-cyan-500 hover:bg-cyan-400 transition-all duration-300 text-black font-bold py-4 rounded-xl text-lg mb-6"
          >
            Upload Dataset
          </button>

          {/* Upload Message */}

          {message && (

            <div className="mb-6 text-center text-green-400 text-lg">

              {message}
            </div>
          )}

          {/* Target Column */}

          <input
            type="text"
            placeholder="Enter Target Column Name"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-xl p-4 mb-6"
          />

          {/* Train Button */}

          <button
  onClick={handleTraining}
  disabled={loading}
  className="w-full bg-gradient-to-r from-pink-500 to-purple-500 hover:scale-105 hover:shadow-2xl hover:shadow-pink-500/30 transition-all duration-300 text-white font-bold py-4 rounded-xl text-lg"
>

  {
    loading
      ? `Training Models... ${progress}%`
      : "Train Models"
  }

</button>

{loading && (

  <div className="w-full bg-slate-800 rounded-full h-3 mt-6 overflow-hidden">

    <div
      className="bg-gradient-to-r from-cyan-400 to-purple-500 h-3 transition-all duration-500"
      style={{ width: `${progress}%` }}
    ></div>

  </div>
)}

        </div>

        {/* Results Dashboard */}

        {trainingResult && (

        <>

          {/* Dataset Analytics */}

<div className="mt-16 w-full max-w-5xl">

  <div className="grid md:grid-cols-3 gap-6">

    <div className="bg-white/10 border border-white/20 rounded-2xl p-6">

      <p className="text-slate-400">
        Rows
      </p>

      <h3 className="text-3xl font-bold text-cyan-400">

        {trainingResult.dataset_info.rows}
      </h3>
    </div>

    <div className="bg-white/10 border border-white/20 rounded-2xl p-6">

      <p className="text-slate-400">
        Columns
      </p>

      <h3 className="text-3xl font-bold text-purple-400">

        {trainingResult.dataset_info.columns}
      </h3>
    </div>

    <div className="bg-white/10 border border-white/20 rounded-2xl p-6">

      <p className="text-slate-400">
        Missing Values
      </p>

      <h3 className="text-3xl font-bold text-pink-400">

        {trainingResult.dataset_info.missing_values}
      </h3>
    </div>

  </div>

</div>

          <div className="mt-16 w-full max-w-6xl mx-auto flex flex-col items-center">

            {/* Results Section */}

<div className="mt-10 w-full max-w-5xl mx-auto">

  {/* Section Title */}

  <div className="flex items-center justify-between mb-10">

    <div>

      <h2 className="text-4xl font-bold">

        Training Results
      </h2>

      <p className="text-slate-400 mt-2">

        Compare algorithm performances intelligently
      </p>
    </div>

    <div className="flex gap-4">

      <div className="bg-white/10 border border-white/20 px-6 py-4 rounded-2xl backdrop-blur-lg">

        <p className="text-slate-400 text-sm">

          Problem Type
        </p>

        <h3 className="text-cyan-400 text-xl font-bold">

          {trainingResult.problem_type}
        </h3>
      </div>

      <div className="bg-white/10 border border-white/20 px-6 py-4 rounded-2xl backdrop-blur-lg">

        <p className="text-slate-400 text-sm">

          Best Model
        </p>

        <h3 className="text-green-400 text-xl font-bold">

          {trainingResult.best_model}
        </h3>
      </div>
    </div>
  </div>

  {/* Leaderboard */}

  <div className="bg-white/10 border border-white/20 rounded-3xl backdrop-blur-lg overflow-hidden">

    {Object.entries(trainingResult.results)

  .sort((a, b) => {

    const metricsA = a[1];

    const metricsB = b[1];

    // Regression sorting
    if (trainingResult.problem_type === "Regression") {

      return metricsA["RMSE"] - metricsB["RMSE"];
    }

    // Classification sorting
    return metricsB["F1 Score"] - metricsA["F1 Score"];
  })

  .map(

      ([model, metrics], index) => (

        <div
          key={model}
          className={`flex items-center justify-between px-8 py-6 border-b border-white/10 hover:bg-white/5 transition-all duration-300 ${
  index === 0
    ? "bg-green-500/10 border border-green-500/30"
    : ""
}`}
        >

          {/* Left */}

          <div className="flex items-center gap-6">

            <div className="w-12 h-12 rounded-2xl bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-bold text-lg">

              {
  index === 0
    ? "🥇"
    : index === 1
    ? "🥈"
    : index === 2
    ? "🥉"
    : `#${index + 1}`
}
            </div>

            <div>

              <h3 className="text-xl font-semibold">

                {model}
              </h3>

              <p className="text-slate-400 text-sm">

                Machine Learning Algorithm
              </p>
            </div>
          </div>

          {/* Right */}

          <div className="flex gap-8">

            {Object.entries(metrics).map(

              ([metric, value]) => (

                <div
                  key={metric}
                  className="text-center"
                >

                  <p className="text-slate-500 text-sm">

                    {metric}
                  </p>

                  <h4 className="text-lg font-bold">

                    {Number(value).toFixed(4)}
                  </h4>
                </div>
              )
            
            )}
          </div>
        </div>
      )
    )}
  </div>
</div>

{/* Dataset Preview */}

<div className="mt-16 w-full max-w-5xl">

  <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg overflow-x-auto">

    <h2 className="text-4xl font-bold mb-8 text-center">

      Dataset Preview
    </h2>

    <table className="w-full border-collapse">

      <thead>

        <tr className="border-b border-white/20">

          {
            Object.keys(
              trainingResult.dataset_preview[0]
            ).map((column) => (

              <th
                key={column}
                className="text-left p-4 text-cyan-400"
              >
                {column}
              </th>
            ))
          }
        </tr>

      </thead>

      <tbody>

        {
          trainingResult.dataset_preview.map(
            (row, index) => (

              <tr
                key={index}
                className="border-b border-white/10 hover:bg-white/5"
              >

                {
                  Object.values(row).map(
                    (value, i) => (

                      <td
                        key={i}
                        className="p-4 text-slate-300"
                      >
                        {String(value)}
                      </td>
                    )
                  )
                }

              </tr>
            )
          )
        }

      </tbody>

    </table>

  </div>
</div>

{/* Charts Dashboard */}

<div className="mt-16 w-full max-w-5xl">

  <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg">

    <h2 className="text-4xl font-bold mb-8 text-center">

      Performance Charts
    </h2>

    <ResponsiveContainer width="100%" height={400}>

      <BarChart
        data={Object.entries(trainingResult.results).map(
          ([name, values]) => ({
            name,
            value:
              values["Accuracy"] ||
              values["R2 Score"] ||
              0
          })
        )}
      >

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis
  dataKey="name"
  angle={-15}
  textAnchor="end"
  interval={0}
  height={70}
/>

        <YAxis />

        <Tooltip />

        <Bar
          dataKey="value"
          fill="#00d4ff"
        />

      </BarChart>

    </ResponsiveContainer>

  </div>
</div>

{/* Advanced Model Comparison */}

<div className="mt-16 w-full max-w-5xl">

  <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg">

    <h2 className="text-4xl font-bold mb-8 text-center">

      Model Comparison Analytics
    </h2>

    <ResponsiveContainer width="100%" height={500}>

      <BarChart
        layout="vertical"
        data={
          Object.entries(trainingResult.results)

          .map(([name, metrics]) => ({

            name,

            score:

              metrics["Accuracy"] ||

              metrics["R2 Score"] ||

              metrics["F1 Score"] ||

              0
          }))

          .sort((a, b) => b.score - a.score)
        }
      >

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis type="number" />

        <YAxis
          type="category"
          dataKey="name"
          width={180}
        />

        <Tooltip />

        <Legend />

        <Bar
          dataKey="score"
          fill="#00d4ff"
          radius={[0, 12, 12, 0]}
        />

      </BarChart>

    </ResponsiveContainer>

  </div>
</div>

{/* Feature Importance */}

{
  Object.keys(trainingResult.feature_importance).length > 0 && (

    <div className="mt-16 w-full max-w-5xl">

      <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg">

        <h2 className="text-4xl font-bold mb-8 text-center">

          Feature Importance
        </h2>

        <div className="space-y-6">

          {
            Object.entries(
              trainingResult.feature_importance
            )

            .sort((a, b) => b[1] - a[1])

            .map(([feature, importance]) => (

              <div key={feature}>

                <div className="flex justify-between mb-2">

                  <span className="text-lg">

                    {feature}
                  </span>

                  <span className="text-cyan-400 font-bold">

                    {(importance * 100).toFixed(2)}%
                  </span>
                </div>

                <div className="w-full bg-slate-800 rounded-full h-4 overflow-hidden">

                  <div
                    className="bg-gradient-to-r from-cyan-400 to-purple-500 h-4 rounded-full"
                    style={{
                      width: `${importance * 100}%`
                    }}
                  ></div>

                </div>

              </div>
            ))
          }

        </div>

      </div>

    </div>
  )
}

{/* AI Insights */}

<div className="mt-16 w-full max-w-5xl">

  <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg">

    <h2 className="text-4xl font-bold mb-8 text-center">

      AI Insights
    </h2>

    <div className="grid md:grid-cols-3 gap-6">

      {/* Best Model */}

      <div className="bg-slate-900/60 rounded-2xl p-6 border border-white/10">

        <p className="text-slate-400 mb-2">

          🏆 Best Model
        </p>

        <h3 className="text-2xl font-bold text-green-400">

          {trainingResult.best_model}
        </h3>
      </div>

      {/* Models Trained */}

      <div className="bg-slate-900/60 rounded-2xl p-6 border border-white/10">

        <p className="text-slate-400 mb-2">

          🤖 Models Trained
        </p>

        <h3 className="text-2xl font-bold text-cyan-400">

          {
            Object.keys(trainingResult.results).length
          }
        </h3>
      </div>

      {/* Problem Type */}

      <div className="bg-slate-900/60 rounded-2xl p-6 border border-white/10">

        <p className="text-slate-400 mb-2">

          📊 Problem Type
        </p>

        <h3 className="text-2xl font-bold text-purple-400">

          {trainingResult.problem_type}
        </h3>
      </div>
    </div>

    {/* Recommendation */}

    <div className="mt-8 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-white/10 rounded-2xl p-6">

      <p className="text-slate-400 mb-3">

        🧠 AI Recommendation
      </p>

      <h3 className="text-xl leading-relaxed">

        {trainingResult.best_model} achieved the best
        performance for this dataset and is recommended
        for production-level predictions.
      </h3>
    </div>

    <div className="mt-8 flex justify-center gap-4">

  <a
    href="http://localhost:8000/download-model"
    download
    className="bg-cyan-500 hover:bg-cyan-400 transition-all duration-300 text-black font-bold px-8 py-4 rounded-2xl text-lg"
  >
    Download Best Model
  </a>

  <a
    href="http://localhost:8000/download_report"
    download
    className="bg-purple-500 hover:bg-purple-400 transition-all duration-300 text-white font-bold px-8 py-4 rounded-2xl text-lg"
  >
    Download AI Report
  </a>

</div>
  </div>
</div>


{/* Prediction Dashboard */}

<div className="mt-16 w-full max-w-5xl">

  <div className="bg-white/10 border border-white/20 rounded-3xl p-8 backdrop-blur-lg">

    <h2 className="text-4xl font-bold mb-3">

      Make Prediction

    </h2>

    <p className="text-slate-400 mb-10">

      Enter feature values to generate AI predictions

    </p>

    {/* Dynamic Inputs */}

    <div className="grid md:grid-cols-2 gap-6">

      {trainingResult.feature_columns.map((feature) => (

        <input
          key={feature}
          type="text"
          placeholder={feature}
          className="bg-slate-900 border border-slate-700 rounded-2xl p-4"
          onChange={(e) =>
            handleInputChange(feature, e.target.value)
          }
        />
      ))}

    </div>

    {/* Predict Button */}

    <button
      onClick={handlePrediction}
      className="mt-10 w-full bg-green-500 hover:bg-green-400 transition-all duration-300 text-black font-bold py-4 rounded-2xl text-lg"
    >

      Generate Prediction

    </button>

    {/* Prediction Result */}

    {prediction && (

      <div className="mt-10 bg-green-500/10 border border-green-500/30 rounded-3xl p-8 text-center">

        <p className="text-slate-400 mb-3">

          Prediction Result

        </p>

        <h3 className="text-5xl font-extrabold text-green-400">

          {prediction}

        </h3>

      </div>
    )}

  </div>

</div>
          </div>
        </>
        )}
      </div>
    </div>
  );
}