# Testing Documentation

## Test Environment
- Model: phi-3.5-mini (Microsoft Foundry Local)
- Embedding Model: all-MiniLM-L6-v2
- Test Document: University transcript (PDF)

---

## Test Cases

| # | Question | Expected Answer | Actual Answer | Result |
|---|----------|----------------|---------------|--------|
| 1 | What is my GPA? | 2.72 | 2.72 | ✅ Pass |
| 2 | What is the letter grade of SEN1011? | A | A | ✅ Pass |
| 3 | When do robotics club meetings happen? | Every Tuesday at 5 PM | Every Tuesday at 5 PM | ✅ Pass |
| 4 | When does the application period open? | March | March | ✅ Pass |
| 5 | What is the campus district? | Besiktas | Besiktas | ✅ Pass |
| 6 | What is the weather today? | Should say it doesn't know | Gave irrelevant answer | ❌ Fail |

---

## Observations

- Line-based chunking significantly improved accuracy for structured documents (transcripts, tables)
- Period-based chunking caused decimal numbers like "3.85" to split incorrectly — fixed by switching to line-based chunking
- The model occasionally adds unnecessary explanation beyond the direct answer
- Out-of-scope questions (e.g. weather) are not always handled gracefully — this is a known limitation of the current system prompt

---

## Known Limitations

- Model sometimes answers out-of-scope questions instead of saying "I don't know"
- Response quality depends on chunking quality — poorly formatted PDFs may give weaker results
- No conversation memory between questions

---

## Conclusion

Core RAG functionality works correctly across multiple document types and question categories. Retrieval accuracy is high for structured documents. Main area for improvement is handling out-of-scope queries more robustly.