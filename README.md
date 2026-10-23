# 🛡️ Sentinel Edge
### An Intelligent Conversational AI for Humanoid Robots & Edge Devices

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Ollama](https://img.shields.io/badge/Local%20LLM-Phi3%203.8B-green)
![Platform](https://img.shields.io/badge/Edge-Jetson%20Orin%20Nano%20Super-76B900)

> A hybrid AI assistant that runs mostly on your own hardware and calls
> the cloud only when a question truly needs it.

<!-- Add a demo GIF or screenshot of the UI here -->

## Overview

Most AI assistants send every query to a remote server. That means latency,
internet dependence, privacy concerns, and ongoing cost. Sentinel Edge takes
a different approach.

It runs a small local model (Phi3 3.8B via Ollama) for everyday questions
and uses **ClawRouter**, a rule-based routing engine, to escalate complex,
reasoning-heavy, or time-sensitive queries to a stronger cloud model. A
layered memory system keeps conversations context-aware without slowing
the local model down.

The project is designed for low-cost deployment on edge devices and
humanoid robots, and was tested on the NVIDIA Jetson Orin Nano Super.

## Key Features

- **Hybrid local-cloud inference**: simple queries stay on-device; hard ones go to the cloud
- **ClawRouter**: scores every query across 15 dimensions with no extra model inference
- **Three-layer memory**: user profile, semantic RAG memory, and rolling chat summary
- **Voice conversation**: local Whisper speech recognition and Kokoro text-to-speech
- **Multiple chat sessions**: ChatGPT-style sidebar with persistent history
- **Privacy-friendly**: most data never leaves your device
