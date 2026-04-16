# Milestone 1: Qualitative Evaluation Report

## 1. Query Set and Diversity
To evaluate our retrieval systems, we developed a set of 10 queries categorized by difficulty and intent. This set includes keyword-specific searches, synonym-heavy natural language, and abstract thematic queries to test the limits of both BM25 and Semantic search.

| ID | Query | Difficulty | Type |
| :--- | :--- | :--- | :--- |
| 1 | Glee season 4 musical comedy | Easy | Exact Keyword |
| 2 | Taylor Cole Jack Turner chalet wedding | Easy | Named Entities |
| 3 | Texas sheriff border corruption movie | Easy | Topic/Thematic |
| 4 | funny tv series about high school singing competitions | Medium | Synonym/Paraphrase |
| 5 | romantic movie about a couple trying to plan a small getaway marriage | Medium | Natural Language |
| 6 | police officer thriller investigating murders and dark secrets | Medium | Intent-based |
| 7 | comedy about a guy going to wild parties in Las Vegas to find a girlfriend | Medium | Scene Description |
| 8 | independent film about mental illness with an ending that lacks closure | Complex | Abstract/Thematic |
| 9 | family friendly shows that young kids will absolutely love to watch | Complex | Sentiment/Subjective |
| 10 | highly rated show that amazon needs to buy and bring back | Complex | Meta-commentary |

---

## 2. Comparison of Results

The retrieval results for all test queries can be found in results_log.txt.
### BM25 Strengths
BM25 demonstrated high precision when exact tokens were present in the metadata. 
For Query 10 ("highly rated show that amazon needs to buy and bring back"), BM25 correctly identified Sneaky Pete as the top result with a high score of **14.91**, likely due to exact matches in user reviews. 
* It also handled Query 8 ("independent film about mental illness...") well, placing Silver Linings Playbook at the top with a score of 10.00. 
* BM25 is the superior choice when the user knows the exact title or specific unique keywords.

### Semantic Search Strengths
Semantic retrieval showed superior performance on natural language queries where the exact keywords were missing. 
* In Query 6 ("police officer thriller..."), while BM25 returned *The Descendants* , the Semantic search successfully retrieved **CSI: Crime Scene Investigation** and **Columbo**, recognizing the "investigation" and "police" intent through embeddings. 
* For Query 9 ("family friendly shows..."), Semantic search correctly prioritized **Creative Galaxy** (score **0.5685**), which is a show explicitly for children, whereas BM25 relied on generic keywords in unrelated titles.

---

## 3. Failure Analysis

### Where BM25 Fails but Semantic Succeeds
* Query 4: For "funny tv series about high school singing competitions," BM25 returned **SOAP - The Complete Series** and **Spartacus**, likely due to high weights on the word "series" or "season". 
* Semantic search successfully identified **The Kicks** and **My Blueberry Nights**, which better capture the "competition" and "TV series" intent through vector similarity.

### Where Semantic Retrieval Fails
* Query 1: For the specific keyword "Glee," Semantic search failed to find the exact show, instead returning **Thou Shalt Laugh 4**. This highlights a failure where the embedding focuses on the "comedy" genre but loses the specific "Glee" entity.
* Query 2: For the specific actors "Taylor Cole" and "Jack Turner," Semantic search returned **Seven Brides for Seven Brothers**. It captured the general "wedding" concept but failed to respect the specific named entities that BM25 is better equipped to handle.

---

## 4. Summary of Insights
* **Keyword vs. Intent:** BM25 remains essential for queries containing specific titles, actors, or unique nouns. However, it is brittle when users use synonyms, such as searching for "police" when a document uses the word "sheriff".
* **The "Fuzzy" Gap:** Semantic search is excellent at finding "vibe" and "genre," but can sometimes be "too fuzzy," missing exact matches in favor of items that are conceptually similar but factually incorrect.
* **Recommendation:** For a production system, a Hybrid Search approach—combining both BM25 and Semantic scores—would be most effective to balance keyword exactness with natural language understanding.
