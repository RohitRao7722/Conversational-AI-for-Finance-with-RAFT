# Conversational-AI-for-Finance-with-RAFT


This application uses a conversational AI model powered by Groq, Google Generative AI embeddings, and RAFT (Feedback-Driven Retrieval and Analysis for Financial Documents). It enables users to upload PDF documents containing financial data and ask finance-related questions, with the model using feedback to refine its document retrieval and response generation process.

## Features

- **PDF Upload:** Users can upload multiple PDF files containing financial data.
- **Chat History:** The system maintains conversation history for continuous dialogue.
- **Contextual Question Reformulation:** Reformulates questions based on prior conversation to provide more accurate responses.
- **Feedback-Driven Adjustment:** Users can provide feedback on the answers, which is used to simulate model improvements (RAFT).
- **Q&A with Context:** Answers are generated based on the context retrieved from uploaded PDFs.

## Technologies Used

- **Groq:** Used for advanced AI model inference and processing.
- **Google Generative AI:** Embeddings for understanding the financial context.
- **Qdrant:** Vector search for document retrieval.
- **Langchain:** Chain creation, document processing, and prompt management.
- **Streamlit:** Interface for interacting with the AI system.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/finance-ai-raft.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd finance-ai-raft
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file in the project root with the following:

```
GOOGLE_API_KEY=<your_google_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

2. **Upload your PDF documents:** Upload one or more PDFs containing the financial documents you'd like to analyze.

3. **Enter your Groq API Key:** Provide the API key for Groq to enable model inference.

4. **Ask a question:** Type a finance-related question about the uploaded documents.

5. **Provide Feedback:** After receiving an answer, you can provide feedback on whether it was helpful. The system uses this feedback to simulate model refinement.

6. **View Chat History:** Review your entire conversation history.

## How RAFT Works

**Feedback-Driven Retrieval:** After each interaction, users can provide feedback (Yes/No) about the answer's usefulness. If feedback is negative ("No"), the system simulates improvements by adjusting retrieval methods or model behavior.

**Improvement Process:** In the full implementation, feedback would trigger dynamic adjustments to the retrieval system or fine-tuning of the model based on user input.

## Example

1. **Upload:** Upload a financial PDF document.
2. **Ask a Question:** "What is the latest quarterly revenue for the company?"
3. **Answer:** The system retrieves relevant data and provides an answer.
4. **Feedback:** If the answer is not accurate, provide feedback, and the system will adjust based on this feedback.

