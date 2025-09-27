import os
import base64
import requests
import json
from flask import Flask, request, jsonify, render_template, render_template_string
import datetime
import uuid
import time
from typing import Dict, Any, Tuple
import joblib 
import numpy as np 

# Initialize Flask App
app = Flask(__name__)

# Firestore integration
from db import save_analysis_to_firestore, fetch_all_analysis_records

APP_ID = "my_app_id"  # Change as needed

# Dashboard route
@app.route('/dashboard')
def dashboard():
    records = fetch_all_analysis_records(APP_ID)

    # Aggregate stats
    total_calls = len(records)
    tech_issues = sum(1 for r in records if r.get('structured_record', {}).get('model_classification', '') == 'Technical')
    business_issues = sum(1 for r in records if r.get('structured_record', {}).get('model_classification', '') == 'Non-Technical')
    trust_scores = [r.get('structured_record', {}).get('sentiment_score', 0) for r in records]
    avg_trust_score = round(sum(trust_scores) / len(trust_scores), 2) if trust_scores else 0
    ahts = [r.get('structured_record', {}).get('avgAHT', 0) for r in records if 'avgAHT' in r.get('structured_record', {})]
    avg_aht = round(sum(ahts) / len(ahts), 2) if ahts else 0
    res_times = [r.get('structured_record', {}).get('resTime', 0) for r in records if 'resTime' in r.get('structured_record', {})]
    avg_res_time = round(sum(res_times) / len(res_times), 2) if res_times else 0
    failed_esc = sum(1 for r in records if r.get('structured_record', {}).get('failedEsc', 0))
    churn_rates = [r.get('structured_record', {}).get('churnRate', 0) for r in records if 'churnRate' in r.get('structured_record', {})]
    avg_churn_rate = round(sum(churn_rates) / len(churn_rates), 2) if churn_rates else 0

    # Distributor performance data
    distributors = []
    for r in records:
        sr = r.get('structured_record', {})
        distributors.append({
            'name': r.get('filename', 'Unknown'),
            'calls': 1,
            'techPct': 100 if sr.get('model_classification', '') == 'Technical' else 0,
            'sentiment': sr.get('sentiment_score', 0),
            'riskScore': sr.get('MODEL_CONFIDENCE_SCORES', {}).get('Technical', 0),
            'csat': sr.get('sentiment_score', 0),
            'escRate': sr.get('MODEL_CONFIDENCE_SCORES', {}).get('Technical', 0),
            'scoreType': 'critical' if sr.get('model_classification', '') == 'Technical' else 'low',
            'tier': 'platinum',
            'contract': 'active',
            'product': 'cloud',
            'history': [],
            'nonTechPct': 100 if sr.get('model_classification', '') == 'Non-Technical' else 0,
            'avgAHT': sr.get('avgAHT', 0),
            'failedAudits': sr.get('failedEsc', 0),
            'lastActivity': r.get('timestamp', '')
        })

    # Pass all data to template
    return render_template(
        'dashboard.html',
        records=records,
        total_calls=total_calls,
        tech_issues=tech_issues,
        business_issues=business_issues,
        avg_trust_score=avg_trust_score,
        avg_aht=avg_aht,
        avg_res_time=avg_res_time,
        failed_esc=failed_esc,
        avg_churn_rate=avg_churn_rate,
        distributors=distributors
    )

# --- CONFIGURATION ---
# IMPORTANT: I have placed your provided key here. 
# If you still get the 'API key is not configured' error, 
# you MUST replace this value with a valid, working key from Google AI Studio.
API_KEY = "AIzaSyAeRJD-CSWRvutlIzp3FiibLQ24raWxFDo" # <--- RECTIFIED LINE

API_MODEL = "gemini-2.5-flash-preview-05-20"
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent?key="
MAX_RETRIES = 3
MAX_AUDIO_SIZE_MB = 20 

# --- HTML TEMPLATE (The combined HTML/CSS/JS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Call Intelligence Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .progress-bar {
            transition: width 0.3s ease-in-out;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .pulse-glow {
            animation: pulseGlow 2s ease-in-out infinite alternate;
        }
        @keyframes pulseGlow {
            from { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
            to { box-shadow: 0 0 30px rgba(99, 102, 241, 0.6); }
        }
        .card-hover {
            transition: all 0.3s ease;
        }
        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">

    <div class="glass-effect rounded-3xl shadow-2xl p-8 max-w-6xl w-full min-h-[600px]">
        
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-indigo-400 to-purple-600 rounded-full mb-4 pulse-glow">
                <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                </svg>
            </div>
            <h1 class="text-4xl font-bold text-white mb-2">AI Call Intelligence Platform</h1>
            <p class="text-white/80 text-lg">Advanced audio analysis with real-time transcription and sentiment insights</p>
        </div>

        <div id="upload-section" class="mb-8">
            <form id="upload-form" class="flex flex-col items-center gap-6" enctype="multipart/form-data">
                
                <div class="w-full max-w-2xl">
                    <div id="drop-zone" class="relative border-2 border-dashed border-white/30 rounded-2xl p-12 text-center hover:border-white/50 transition-all duration-300 cursor-pointer glass-effect card-hover">
                        <input type="file" id="audio-file" name="audio_file" accept="audio/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10">
                        
                        <div id="drop-content" class="flex flex-col items-center">
                            <svg class="w-16 h-16 text-white/60 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                            </svg>
                            <h3 class="text-xl font-semibold text-white mb-2">Drop your audio file here</h3>
                            <p class="text-white/70 mb-4">or click to browse</p>
                            <p class="text-sm text-white/50">Supports MP3, WAV, OGG • Max 20MB</p>
                        </div>
                        
                        <div id="file-selected" class="hidden">
                            <svg class="w-12 h-12 text-green-400 mb-3 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            <p id="selected-filename" class="text-white font-medium mb-2"></p>
                            <p class="text-white/60 text-sm">Ready to analyze</p>
                        </div>
                    </div>
                </div>

                <button type="submit" id="submit-btn" class="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold py-4 px-12 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none">
                    <span id="btn-text">Start AI Analysis</span>
                </button>
            </form>
        </div>

        <div id="progress-section" class="hidden mb-8">
            <div class="glass-effect rounded-2xl p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-white">Processing Audio</h3>
                    <span id="progress-percentage" class="text-white/80 font-medium">0%</span>
                </div>
                
                <div class="w-full bg-white/20 rounded-full h-2 mb-4">
                    <div id="progress-bar" class="progress-bar bg-gradient-to-r from-indigo-400 to-purple-500 h-2 rounded-full" style="width: 0%"></div>
                </div>
                
                <div class="flex items-center">
                    <svg id="progress-spinner" class="animate-spin h-5 w-5 text-white mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p id="progress-text" class="text-white/90">Initializing analysis...</p>
                </div>
            </div>
        </div>

        <div id="result-container" class="hidden space-y-6 fade-in">
            
            <div class="glass-effect rounded-2xl p-6 card-hover">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-2xl font-bold text-white flex items-center">
                        <svg class="w-6 h-6 mr-3 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                        </svg>
                        Call Classification
                    </h2>
                    
                    <div id="confidence-badge" class="px-4 py-2 rounded-full text-sm font-medium">
                        <span id="confidence-text"></span>
                    </div>
                </div>
                
                <div class="mb-4">
                    <div id="classification-result" class="text-3xl font-bold mb-2"></div>
                    <p class="text-white/70 text-sm">Based on AI model analysis</p>
                </div>
                
                <div class="bg-black/20 rounded-xl p-4">
                    <h4 class="text-white/80 font-medium mb-2 flex items-center">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        Key Evidence
                    </h4>
                    <p id="classification-justification" class="text-white/90 italic"></p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="glass-effect rounded-2xl p-6 card-hover">
                    <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
                        <svg class="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V1a1 1 0 011-1h2a1 1 0 011 1v16l-3.5-2-3.5 2V1a1 1 0 011-1h2a1 1 0 011 1v3"/>
                        </svg>
                        Original Transcription
                    </h3>
                    <div id="transcription-text" class="bg-black/20 rounded-xl p-4 text-white/90 text-sm leading-relaxed max-h-40 overflow-y-auto"></div>
                </div>
                
                <div class="glass-effect rounded-2xl p-6 card-hover">
                    <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
                        <svg class="w-5 h-5 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/>
                        </svg>
                        English Translation
                    </h3>
                    <div id="translation-text" class="bg-black/20 rounded-xl p-4 text-white/90 text-sm leading-relaxed max-h-40 overflow-y-auto"></div>
                </div>
            </div>

            <div class="glass-effect rounded-2xl p-6 card-hover">
                <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
                    <svg class="w-5 h-5 mr-2 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    Sentiment & Key Insights
                </h3>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div class="bg-black/20 rounded-lg p-4 text-center">
                        <p class="text-white/60 text-sm mb-1">Sentiment Score</p>
                        <p id="sentiment-score" class="text-2xl font-bold text-white"></p>
                    </div>
                    <div class="bg-black/20 rounded-lg p-4 text-center">
                        <p class="text-white/60 text-sm mb-1">Overall Sentiment</p>
                        <p id="sentiment-summary" class="text-lg font-semibold text-white capitalize"></p>
                    </div>
                    <div class="bg-black/20 rounded-lg p-4 text-center">
                        <p class="text-white/60 text-sm mb-1">Action Required</p>
                        <p id="action-indicator" class="text-lg font-semibold text-white">Yes</p>
                    </div>
                </div>
                
                <div class="bg-black/20 rounded-lg p-4">
                    <p class="text-white/70 text-sm mb-2">Recommended Action:</p>
                    <p id="action-summary" class="text-white/90"></p>
                </div>
            </div>

            <div class="glass-effect rounded-2xl p-6 card-hover">
                <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
                    <svg class="w-5 h-5 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/>
                    </svg>
                    Phrase Analysis
                </h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <div class="flex items-center mb-3">
                            <div class="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                            <h4 class="font-medium text-white">Technical Phrases</h4>
                        </div>
                        <div id="technical-phrases-container" class="bg-black/20 rounded-lg p-4 min-h-[100px]">
                            <div id="technical-phrases" class="space-y-2"></div>
                        </div>
                    </div>
                    
                    <div>
                        <div class="flex items-center mb-3">
                            <div class="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                            <h4 class="font-medium text-white">Non-Technical Phrases</h4>
                        </div>
                        <div id="non-technical-phrases-container" class="bg-black/20 rounded-lg p-4 min-h-[100px]">
                            <div id="non-technical-phrases" class="space-y-2"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="text-center pt-4">
                <button id="download-btn" class="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105">
                    <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    Download Analysis Report
                </button>
            </div>
        </div>

        <div id="error-container" class="hidden">
            <div class="bg-red-500/20 border border-red-500/30 rounded-2xl p-6 text-center">
                <svg class="w-12 h-12 text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.996-.833-2.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"/>
                </svg>
                <h3 class="text-xl font-semibold text-white mb-2">Processing Error</h3>
                <p id="error-message" class="text-white/80"></p>
            </div>
        </div>

        <div class="text-center mt-8">
            <button onclick="window.location.href='/dashboard'" class="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105">
                Go to Dashboard
            </button>
        </div>
    </div>

    <script>
        let currentAnalysisData = null;

        // File handling
        const audioFile = document.getElementById('audio-file');
        const dropZone = document.getElementById('drop-zone');
        const dropContent = document.getElementById('drop-content');
        const fileSelected = document.getElementById('file-selected');
        const selectedFilename = document.getElementById('selected-filename');

        // Progress tracking
        const progressSteps = [
            { text: "Initializing analysis...", progress: 10 },
            { text: "Transcribing audio...", progress: 35 },
            { text: "Running classification model...", progress: 65 },
            { text: "Generating insights...", progress: 90 },
            { text: "Finalizing results...", progress: 100 }
        ];

        audioFile.addEventListener('change', handleFileSelect);
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('drop', handleDrop);

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                displaySelectedFile(file);
            }
        }

        function handleDragOver(e) {
            e.preventDefault();
            dropZone.classList.add('border-white/70');
        }

        function handleDrop(e) {
            e.preventDefault();
            dropZone.classList.remove('border-white/70');
            const file = e.dataTransfer.files[0];
            if (file) {
                audioFile.files = e.dataTransfer.files;
                displaySelectedFile(file);
            }
        }

        function displaySelectedFile(file) {
            selectedFilename.textContent = file.name;
            dropContent.classList.add('hidden');
            fileSelected.classList.remove('hidden');
        }

        function updateProgress(stepIndex) {
            const step = progressSteps[stepIndex];
            document.getElementById('progress-text').textContent = step.text;
            document.getElementById('progress-bar').style.width = step.progress + '%';
            document.getElementById('progress-percentage').textContent = step.progress + '%';
        }

        function showError(message) {
            document.getElementById('progress-section').classList.add('hidden');
            document.getElementById('error-message').textContent = message;
            document.getElementById('error-container').classList.remove('hidden');
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('btn-text').textContent = 'Start AI Analysis';
        }

        function displayResults(data) {
            const record = data.structured_record;
            
            // Classification
            const classification = record.model_classification;
            const classificationEl = document.getElementById('classification-result');
            
            if (classification === 'Technical') {
                classificationEl.textContent = 'Technical Call';
                classificationEl.className = 'text-3xl font-bold mb-2 text-red-400';
            } else {
                classificationEl.textContent = 'Non-Technical Call';
                classificationEl.className = 'text-3xl font-bold mb-2 text-green-400';
            }

            // Confidence badge
            const techScore = record.MODEL_CONFIDENCE_SCORES?.Technical || 0;
            const nonTechScore = record.MODEL_CONFIDENCE_SCORES?.['Non-Technical'] || 0;
            const maxScore = Math.max(techScore, nonTechScore);
            
            const confidenceBadge = document.getElementById('confidence-badge');
            const confidenceText = document.getElementById('confidence-text');
            
            if (maxScore > 0.8) {
                confidenceBadge.className = 'px-4 py-2 rounded-full text-sm font-medium bg-green-500/30 text-green-300';
                confidenceText.textContent = 'High Confidence';
            } else if (maxScore > 0.6) {
                confidenceBadge.className = 'px-4 py-2 rounded-full text-sm font-medium bg-yellow-500/30 text-yellow-300';
                confidenceText.textContent = 'Medium Confidence';
            } else {
                confidenceBadge.className = 'px-4 py-2 rounded-full text-sm font-medium bg-red-500/30 text-red-300';
                confidenceText.textContent = 'Low Confidence';
            }

            // Other data
            document.getElementById('classification-justification').textContent = record.CLASSIFICATION_JUSTIFICATION || 'No justification available';
            document.getElementById('transcription-text').textContent = data.transcription_original || 'N/A';
            document.getElementById('translation-text').textContent = data.translation || 'N/A';
            
            // Sentiment
            const sentimentScore = record.sentiment_score || 0;
            document.getElementById('sentiment-score').textContent = sentimentScore.toFixed(2);
            document.getElementById('sentiment-summary').textContent = record.sentiment_summary || 'neutral';
            document.getElementById('action-summary').textContent = record.ACTION_SUMMARY || 'No specific action required';

            // Phrases
            displayPhrases('technical-phrases', record.TECHNICAL_PHRASES || [], 'red');
            displayPhrases('non-technical-phrases', record.NON_TECHNICAL_PHRASES || [], 'green');
            
            currentAnalysisData = record;
            document.getElementById('progress-section').classList.add('hidden');
            document.getElementById('result-container').classList.remove('hidden');
        }

        function displayPhrases(containerId, phrases, colorClass) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            if (phrases.length === 0) {
                container.innerHTML = '<p class="text-white/50 text-sm">No specific phrases identified</p>';
                return;
            }
            
            phrases.forEach(phrase => {
                const phraseEl = document.createElement('div');
                phraseEl.className = `inline-block bg-${colorClass}-500/20 text-${colorClass}-300 px-3 py-1 rounded-full text-sm mr-2 mb-2`;
                phraseEl.textContent = phrase;
                container.appendChild(phraseEl);
            });
        }

        function downloadAnalysis() {
            if (!currentAnalysisData) return;
            
            const jsonString = JSON.stringify([currentAnalysisData], null, 2);
            const blob = new Blob([jsonString], { type: 'application/json' });
            const a = document.createElement('a');
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            a.download = `call_analysis_${currentAnalysisData.call_id}_${timestamp}.json`;
            a.href = URL.createObjectURL(blob);
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(a.href);
        }

        // Main form submission
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = audioFile.files[0];
            if (!file) {
                showError('Please select an audio file to upload.');
                return;
            }
            
            // Reset UI
            document.getElementById('result-container').classList.add('hidden');
            document.getElementById('error-container').classList.add('hidden');
            document.getElementById('progress-section').classList.remove('hidden');
            document.getElementById('submit-btn').disabled = true;
            document.getElementById('btn-text').textContent = 'Processing...';
            
            // Simulate progress
            let currentStep = 0;
            updateProgress(currentStep);
            
            const progressInterval = setInterval(() => {
                currentStep++;
                if (currentStep < progressSteps.length - 1) {
                    updateProgress(currentStep);
                }
            }, 2000);
            
            const formData = new FormData();
            formData.append('audio_file', file);
            
            try {
                const response = await fetch('/process-audio', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                clearInterval(progressInterval);
                
                if (response.ok) {
                    updateProgress(progressSteps.length - 1);
                    setTimeout(() => displayResults(result), 500);
                } else {
                    showError(result.error || 'An unknown error occurred on the server.');
                }
            } catch (error) {
                clearInterval(progressInterval);
                showError('An unexpected network error occurred. Please try again.');
            } finally {
                setTimeout(() => {
                    document.getElementById('submit-btn').disabled = false;
                    document.getElementById('btn-text').textContent = 'Start AI Analysis';
                }, 500);
            }
        });

        document.getElementById('download-btn').addEventListener('click', downloadAnalysis);
    </script>
</body>
</html>
"""


# --- MODEL LOADING (Loads the local .pkl file) ---
# Ensure this file is in the same directory as app.py
CLASSIFIER_MODEL_PATH = "final_advanced_classifier_model (1).pkl"
CLASSIFIER_MODEL = None

try:
    # This line requires you to have the file 'final_advanced_classifier_model (1).pkl' in the same directory.
    CLASSIFIER_MODEL = joblib.load(CLASSIFIER_MODEL_PATH) 
    print(f"✅ Successfully loaded classification model from {CLASSIFIER_MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ WARNING: Classification model file not found at {CLASSIFIER_MODEL_PATH}. Classification will be disabled.")
except ImportError:
    print("⚠️ WARNING: Required libraries (scikit-learn/joblib) not installed. Classification will be disabled.")
except Exception as e:
    print(f"❌ ERROR: Failed to load classification model: {e}. Classification is disabled.")

# --- CLASSIFICATION HELPER FUNCTION ---
def classify_text(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Uses the loaded scikit-learn model to classify text and returns the class
    with the highest probability (Technical or Non-Technical).
    Returns: (Primary Classification, Confidence Scores)
    """
    default_scores = {"Technical": 0.0, "Non-Technical": 0.0}
    if CLASSIFIER_MODEL is None:
        return "Classification Model Not Loaded/Error", default_scores
    
    try:
        # 1. Get probability scores
        proba = CLASSIFIER_MODEL.predict_proba([text])[0]
        classes = CLASSIFIER_MODEL.classes_
        
        # Map class labels (0, 1) to names
        if len(classes) != 2:
            raise ValueError("Model must have exactly two classes (Technical/Non-Technical).")

        # Determine indices for 'Technical' (1) and 'Non-Technical' (0)
        technical_idx = np.where(classes == 1)[0][0]
        non_technical_idx = np.where(classes == 0)[0][0]
        
        tech_proba = proba[technical_idx]
        non_tech_proba = proba[non_technical_idx]
        
        confidence_scores = {
            "Technical": float(f"{tech_proba:.3f}"),
            "Non-Technical": float(f"{non_tech_proba:.3f}"),
        }
        
        # 2. Simplified Classification Logic: Always pick the highest score
        if tech_proba > non_tech_proba:
            primary_class = "Technical"
        else:
            primary_class = "Non-Technical" 
            
        return primary_class, confidence_scores
        
    except Exception as e:
        app.logger.error(f"Error during model prediction: {e}")
        return f"Classification Failed (Error: {e.__class__.__name__})", default_scores

# --- JSON SCHEMA DEFINITION ---
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sentiment_score": {
            "type": "NUMBER",
            "description": "A score between -1.0 (very negative) and 1.0 (very positive)."
        },
        "sentiment_summary": {
            "type": "STRING",
            "description": "One word summary of sentiment (e.g., negative, positive, neutral)."
        },
        "key_phrases": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "The most important phrases from the conversation."
        },
        "POSITIVE_INTENT": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of words or short phrases indicating positive intent or successful outcome."
        },
        "NEGATIVE_INTENT": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of words or short phrases indicating a problem, complaint, or negative intent."
        },
        "ACTION_SUMMARY": {
            "type": "STRING",
            "description": "A brief summary of the required business action or resolution based on the call."
        },
        "CLASSIFICATION_JUSTIFICATION": { 
            "type": "STRING",
            "description": "The specific sentences or key phrases from the transcript that clearly justify the model's classification (Technical or Non-Technical)."
        },
        "TECHNICAL_PHRASES": { 
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "A list of phrases identified by the LLM as highly technical or domain-specific, supporting the 'Technical' aspect of the call."
        },
        "NON_TECHNICAL_PHRASES": { 
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "A list of general, conversational, or non-technical phrases identified by the LLM, supporting the 'Non-Technical' aspect of the call."
        },
        "MODEL_CONFIDENCE_SCORES": {
            "type": "OBJECT",
            "properties": {
                "Technical": {"type": "NUMBER", "description": "Probability score for the Technical class."},
                "Non-Technical": {"type": "NUMBER", "description": "Probability score for the Non-Technical class."}
            },
            "description": "Raw probability scores from the local machine learning model."
        }
    },
    "required": [
        "sentiment_score",
        "sentiment_summary",
        "key_phrases",
        "POSITIVE_INTENT",
        "NEGATIVE_INTENT",
        "ACTION_SUMMARY",
        "CLASSIFICATION_JUSTIFICATION", 
        "TECHNICAL_PHRASES", 
        "NON_TECHNICAL_PHRASES", 
        "MODEL_CONFIDENCE_SCORES" 
    ],
}

def call_gemini_api(payload: Dict[str, Any], is_structured: bool = False) -> Dict[str, Any]:
    """Handles all Gemini API requests with exponential backoff and error handling."""
    
    # Check if API key is configured
    if not API_KEY or API_KEY == "AIzaSyAeRJD-CSWRvutlIzp3FiibLQ24raWxFDo" and not API_KEY.startswith("AIzaSy"):
        # This check is added to ensure the user replaces the key if the provided one fails.
        raise ValueError("API key is not configured. Please replace the placeholder API_KEY in the code with your actual key before running.")

    headers = {'Content-Type': 'application/json'}
    
    if is_structured:
        # Configure the request to output structured JSON based on the schema
        payload['generationConfig'] = {
            "responseMimeType": "application/json",
            "responseSchema": JSON_SCHEMA,
        }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{API_ENDPOINT}{API_KEY}", 
                headers=headers, 
                data=json.dumps(payload),
                timeout=90
            )
            response.raise_for_status()
            
            response_json = response.json()
            
            if 'error' in response_json:
                raise requests.exceptions.HTTPError(f"API returned error: {response_json['error']['message']}")
                
            return response_json
            
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            
            time.sleep(2 ** attempt)
            
    raise Exception("API request failed after all retries.")

@app.route('/')
def index():
    """Renders the main page using the in-memory HTML string."""
    # Uses render_template_string to serve the HTML content directly
    return render_template_string(HTML_TEMPLATE)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """
    Handles audio file upload, performs the three-step analysis pipeline, 
    and returns the complete structured JSON log.
    """
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # --- PREP: Audio Bytes, Mime Type, and Size Check ---
        file_stream = file.stream
        file_stream.seek(0, os.SEEK_END)
        file_size = file_stream.tell()
        file_stream.seek(0, 0) # Reset stream position to beginning
        
        if file_size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds the limit of {MAX_AUDIO_SIZE_MB}MB. Please upload a smaller file.")
            
        audio_bytes = file.read()
        
        # Determine MIME type based on file extension
        filename_lower = file.filename.lower()
        if filename_lower.endswith((".wav", ".wave")):
            mime_type = "audio/wav"
        elif filename_lower.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif filename_lower.endswith((".mp3", ".mpeg", ".mpg")):
            mime_type = "audio/mpeg"
        else:
            # Fallback for unknown/inferred type
            mime_type = file.mimetype or "audio/unknown"

        if mime_type == "audio/unknown":
            raise ValueError(f"Unsupported or unknown audio file type: {filename_lower.split('.')[-1]}")
        
        audio_data_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # --------------------------------------------
        # STEP 1: Audio Transcription and Translation (LLM)
        # --------------------------------------------
        prompt_step1 = (
            "Transcribe this audio clip exactly in the language it was spoken, "
            "and then provide an accurate English translation. "
            "Format your entire response strictly as two labeled lines: "
            "1. Transcription: [The transcribed text in the original language] "
            "2. Translation: [The English translation]"
        )

        payload_step1 = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_step1},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": audio_data_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        step1_result = call_gemini_api(payload_step1)
        api_text = step1_result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        transcription = "N/A (Transcription Failed)"
        translation = "N/A (Translation Failed)"

        # Robust parsing of the two required lines
        lines = api_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("1. Transcription:"):
                transcription = line.split(":", 1)[-1].strip()
            elif line.startswith("2. Translation:"):
                translation = line.split(":", 1)[-1].strip()

        # Fallback: if translation failed but transcription succeeded, use transcription as translation
        if translation in ["N/A (Translation Failed)", "Could not find translation.", ""]:
            if transcription and transcription not in ["N/A (Transcription Failed)"]:
                translation = transcription 
            else:
                raise ValueError("Could not extract a valid English translation from the audio content.")

        # --------------------------------------------
        # STEP 2: Advanced Model Classification (PKL Model)
        # --------------------------------------------
        loading_text = "Processing audio (Step 2 of 3: Local Model Classification)..."
        app.logger.info(loading_text)
        model_classification, confidence_scores = classify_text(translation) 
        app.logger.info(f"Model Classification Result: {model_classification}")
        app.logger.info(f"Model Confidence Scores: {confidence_scores}")


        # --------------------------------------------
        # STEP 3: Structured Analysis from English Text (LLM) - Segregates Phrases
        # --------------------------------------------
        loading_text = "Processing audio (Step 3 of 3: Structured LLM Analysis)..."
        app.logger.info(loading_text)

        prompt_step2 = (
            f"Analyze the following call transcript for sentiment, key phrases, required business action, and extract the single most relevant sentence or short phrase that justifies the classification."
            f"Critically examine the transcript and create two distinct lists of key phrases: one for **TECHNICAL_PHRASES** and one for **NON_TECHNICAL_PHRASES**. These lists must be populated regardless of the primary topic classification, ensuring all relevant domain-specific terms are in TECHNICAL_PHRASES and all general/emotional/customer-service phrases are in NON_TECHNICAL_PHRASES. "
            f"The pre-computed primary model topic classification is: {model_classification}. "
            f"The model confidence scores are: Technical={confidence_scores.get('Technical', 0.0)}, Non-Technical={confidence_scores.get('Non-Technical', 0.0)}. "
            f"Strictly use the provided JSON schema for output. "
            f"Transcript: \"{translation}\""
        )

        payload_step2 = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_step2}]
                }
            ]
        }
        
        # Call the API, enforcing the structured JSON output
        step2_result = call_gemini_api(payload_step2, is_structured=True)
        
        # Extract and parse the JSON response
        raw_json_string = step2_result['candidates'][0]['content']['parts'][0]['text']
        analysis_data = json.loads(raw_json_string)

        # --------------------------------------------
        # Final Log Construction
        # --------------------------------------------
        call_id = f"call_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        distributor_id = "dist_42" 

        # Combine all data into the final structured record
        structured_record = {
            "call_id": call_id,
            "distributor_id": distributor_id,
            "model_classification": model_classification, 
            **analysis_data # Merges the LLM-generated JSON into the record
        }
        
        # Ensure the confidence scores from the PKL model overwrite the LLM-generated object
        structured_record["MODEL_CONFIDENCE_SCORES"] = confidence_scores

        # Save output JSON to Firestore
        output_json = {
            "call_id": structured_record["call_id"],
            "transcription": transcription,
            "translation": translation,
            "detected_language": "en",  # You may want to extract this from the API
            "timestamp": time.time(),
            "filename": file.filename,
            "structured_record": structured_record
        }
        save_analysis_to_firestore(APP_ID, output_json)
        # Return all data to the frontend
        return jsonify({
            "transcription_original": transcription,
            "translation": translation,
            "structured_record": structured_record
        })

    except ValueError as e:
        app.logger.error(f"Configuration/Parsing Error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        app.logger.error(f"API Request Failure: {str(e)}")
        error_detail = getattr(e.response, 'text', str(e))
        # Log the specific response content if available for better debugging
        try:
            error_json = json.loads(error_detail)
            error_message = error_json.get('error', {}).get('message', error_detail)
        except json.JSONDecodeError:
            error_message = error_detail
            
        return jsonify({"error": f"External API request failed (503): {error_message}"}), 503
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        app.logger.error(f"AI Response Structure Error: {e}")
        return jsonify({"error": "Failed to parse AI response. Check model output structure."}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # NOTE: In environments like Colab or notebooks, you might need to use 
    # `from flask_ngrok import run_with_ngrok; run_with_ngrok(app)` 
    # or an equivalent local tunnel solution instead of app.run(debug=True)
    # for the frontend to be accessible.
    app.run(debug=True)