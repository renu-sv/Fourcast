const express = require("express");
const cors = require("cors");
const axios = require("axios");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

app.post("/ask", async (req, res) => {
  const userMessage = req.body.message;

  try {
    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "openai/gpt-3.5-turbo", // or try other models like "mistralai/mistral-7b-instruct"
        messages: [
          {
            role: "system",
            content:
              "You are TurtleBot 🐢, a friendly and knowledgeable water quality expert. Your role is to educate users about water parameters such as pH, BOD, DO, turbidity, and other indicators in a clear and simple way. Use emojis to make your replies engaging and easy to understand. Your core focus is on the water quality of various ecosystems like rivers, canals, lakes, seas, marine environments, and groundwater. Always be helpful, scientific, and approachable — like a smart environmentalist bestie. 🌊💧🐢"
             },
          { role: "user", content: userMessage },
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "http://localhost:3001", // required
          "X-Title": "TurtleBot Chat", // optional title
        },
      }
    );

    res.json({ reply: response.data.choices[0].message.content });
  } catch (error) {
    console.error("OpenRouter error:", error.response?.data || error.message);
    res.status(500).json({
      error: "Failed to get a response from OpenRouter",
    });
  }
});

app.listen(3001, () => console.log("🟢 TurtleBot backend running on port 3001"));
